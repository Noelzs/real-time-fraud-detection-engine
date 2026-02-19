# real_model.py
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import deque
import asyncio
from model_utils import load_model_version

class RealFraudEngine:
    def __init__(self, model_version="v5_xg_20251109_154848"):
        print(f"🧠 Loading real XGBoost model: {model_version}")
        self.model, self.features, self.performance = load_model_version(model_version)
        self.demo_data = pd.read_csv(f"model_versions/{model_version}/demo_data.csv")
        self.is_running = False
        self.current_batch = 0
        self.total_processed = 0
        self.stats = {
            "total_processed": 0,
            "fraud_detected": 0,
            "missed_fraud": 0,
            "false_alarms": 0,
            "detection_rate": 0.0,
            "processing_speed": 0.0,
            "alert_rate": 0.0,
            "threshold": 0.063,
            "model_auc": self.performance['roc_auc'],
            "batch_size": 1000
        }
        self.event_buffer = deque(maxlen=100)
        self.last_calc_time = time.time()
        self.last_count = 0
    
    def calculate_real_speed(self):
        """Calculate actual processing speed"""
        current_time = time.time()
        time_diff = current_time - self.last_calc_time
        
        if time_diff >= 0.5:
            processed_since = self.stats["total_processed"] - self.last_count
            if processed_since > 0:
                self.stats["processing_speed"] = processed_since / time_diff
            
            self.last_calc_time = current_time
            self.last_count = self.stats["total_processed"]
        
        return self.stats["processing_speed"]
    
    async def process_batch(self, batch_size=100, threshold=0.063):
        """Process a batch of real transactions through the model"""
        if self.current_batch >= len(self.demo_data):
            self.current_batch = 0  # Loop back to start
        
        batch_end = min(self.current_batch + batch_size, len(self.demo_data))
        batch_data = self.demo_data.iloc[self.current_batch:batch_end]
        
        # Prepare features
        feature_names = self.features['feature_names']
        batch_features = batch_data[feature_names].values
        
        # REAL MODEL PREDICTION
        batch_probs = self.model.predict_proba(batch_features)[:, 1]
        
        # Get actual fraud labels
        if 'isFraud' in batch_data.columns:
            batch_labels = batch_data['isFraud'].values
        else:
            batch_labels = np.zeros(len(batch_data), dtype=int)
        
        events = []
        
        for i, fraud_prob in enumerate(batch_probs):
            self.stats["total_processed"] += 1
            self.total_processed += 1
            
            is_flagged = fraud_prob > threshold
            is_actual_fraud = batch_labels[i] == 1
            
            if is_actual_fraud:
                if is_flagged:
                    event_type = "detected_fraud"
                    self.stats["fraud_detected"] += 1
                else:
                    event_type = "missed_fraud"
                    self.stats["missed_fraud"] += 1
            else:
                if is_flagged:
                    event_type = "false_alarm"
                    self.stats["false_alarms"] += 1
                else:
                    event_type = "legitimate"
            
            # Update derived stats
            total_frauds = self.stats["fraud_detected"] + self.stats["missed_fraud"]
            if total_frauds > 0:
                self.stats["detection_rate"] = (self.stats["fraud_detected"] / total_frauds) * 100
            
            total_alerts = self.stats["fraud_detected"] + self.stats["false_alarms"]
            if self.stats["total_processed"] > 0:
                self.stats["alert_rate"] = (total_alerts / self.stats["total_processed"]) * 100
            
            # Calculate speed
            self.calculate_real_speed()
            
            # Create event
            amount = float(batch_data.iloc[i]['TransactionAmt'])
            
            event = {
                "id": f"TX-REAL-{self.total_processed}",
                "amount": amount,
                "fraud_prob": float(fraud_prob),
                "is_flagged": bool(is_flagged),
                "type": event_type,
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "interesting": event_type != "legitimate",
                "model": "XGBoost v5",
                "batch": self.current_batch // batch_size
            }
            
            if event["interesting"]:
                self.event_buffer.append(event)
            
            events.append(event)
        
        self.current_batch = batch_end
        
        return events