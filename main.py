# main.py - FIXED VERSION (with correct WebSocket endpoints)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from collections import deque
import asyncio
import random
import json
import time
from datetime import datetime
from real_model import RealFraudEngine

# ==================== FRAUD ENGINE (Simulation) ====================
class FraudEngine:
    def __init__(self):
        self.stats = {
            "total_processed": 0,
            "fraud_detected": 0,
            "missed_fraud": 0,
            "false_alarms": 0,
            "detection_rate": 0.0,
            "processing_speed": 0,
            "alert_rate": 0.0,
            "threshold": 0.063
        }
        self.active_connections = []
        self.event_buffer = deque(maxlen=100)
        self.is_running = False
        # For dynamic speed calculation
        self.last_calc_time = time.time()
        self.last_count = 0
        
        # Set random seed for consistent demos (optional)
        random.seed(42)  # Remove this line for random demos each time
    
    def calculate_real_speed(self):
        """Calculate actual processing speed in real-time"""
        current_time = time.time()
        time_diff = current_time - self.last_calc_time
        
        # Update speed every 0.5 seconds for smoother display
        if time_diff >= 0.5:
            processed_since = self.stats["total_processed"] - self.last_count
            if processed_since > 0:
                self.stats["processing_speed"] = processed_since / time_diff
            
            self.last_calc_time = current_time
            self.last_count = self.stats["total_processed"]
        
        return self.stats["processing_speed"]
    
    async def generate_transaction(self):
        """Generate a realistic transaction with fraud probability"""
        transaction_id = random.randint(100000, 999999)
        amount = random.uniform(500, 50000)
        
        # Realistic fraud rate (1.5% - matches IEEE-CIS dataset)
        is_fraud = random.random() < 0.015
        
        # =========== FIXED: Match your XGBoost model performance ===========
        # Your real model: 39.7% detection rate, 8.8% false alarm rate
        
        if is_fraud:
            # FRAUD transactions
            # 39.7% get caught (above threshold), 60.3% missed (below threshold)
            if random.random() < 0.397:  # 39.7% detection rate
                fraud_prob = random.uniform(0.064, 0.8)  # Above threshold (will be flagged)
            else:  # 60.3% missed
                fraud_prob = random.uniform(0.001, 0.062)  # Below threshold (won't be flagged)
        else:
            # LEGITIMATE transactions  
            # 91.2% correctly ignored (below threshold), 8.8% false alarms (above threshold)
            if random.random() < 0.088:  # 8.8% false alarm rate
                fraud_prob = random.uniform(0.064, 0.3)  # Above threshold (false alarm)
            else:  # 91.2% correctly ignored
                fraud_prob = random.uniform(0.001, 0.062)  # Below threshold (correctly ignored)
        
        is_flagged = fraud_prob > self.stats["threshold"]
        
        # Update statistics
        self.stats["total_processed"] += 1
        
        if is_fraud:
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
        
        # Calculate real-time speed
        self.calculate_real_speed()
        
        # Create event
        event = {
            "id": f"TX-SIM-{self.stats['total_processed']}",
            "amount": round(amount, 2),
            "fraud_prob": round(fraud_prob, 4),
            "is_flagged": is_flagged,
            "type": event_type,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "interesting": event_type != "legitimate",
            "model": "Simulation"
        }
        
        if event["interesting"]:
            self.event_buffer.append(event)
        
        return event

# ==================== LIFESPAN MANAGER ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    print("🚀 Starting Fraud Detection API...")
    
    # Initialize both engines
    app.state.simulation_engine = FraudEngine()
    app.state.real_engine = RealFraudEngine()
    
    print("✅ Simulation engine ready")
    print(f"✅ Real model loaded (AUC: {app.state.real_engine.stats['model_auc']:.3f})")
    print("✅ Backend ready! Visit http://localhost:8000")
    yield
    print("🛑 Shutting down...")

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Fraud Detection API",
    description="Hybrid: Simulation + Real XGBoost Model",
    version="2.0",
    lifespan=lifespan
)

# ==================== CORS CONFIGURATION ====================
# IMPORTANT: This fixes the 403 Forbidden errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== WEB SOCKETS ====================

@app.websocket("/ws/simulation")
async def websocket_simulation(websocket: WebSocket):
    """WebSocket for simulation mode"""
    await websocket.accept()
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "mode": "simulation",
            "message": "Connected to Simulation Mode",
            "performance": "39.7% detection @ 8.8% false alarms (matching XGBoost)"
        })
        
        batch = []
        batch_size = 50
        
        # Keep connection alive while engine is running
        while True:
            try:
                # Check if engine is running
                if not app.state.simulation_engine.is_running:
                    await asyncio.sleep(0.001)
                    continue
                
                # Generate transaction
                transaction = await app.state.simulation_engine.generate_transaction()
                batch.append(transaction)
                
                # Send batch when full
                if len(batch) >= batch_size:
                    try:
                        await websocket.send_json({
                            "type": "batch",
                            "transactions": batch,
                            "stats": app.state.simulation_engine.stats,
                            "mode": "simulation",
                            "batch_size": len(batch),
                            "performance": "39.7% detection @ 8.8% false alarms"
                        })
                        batch = []
                    except Exception as e:
                        print(f"Error sending batch: {e}")
                        break
                
                # NO DELAY for simulation - let it run as fast as possible!
                # The bottleneck will be WebSocket/network speed, not our code
                await asyncio.sleep(0.00001)  # Tiny sleep to prevent CPU hogging
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in simulation loop: {e}")
                break
                
    except WebSocketDisconnect:
        print("Simulation WebSocket client disconnected")
    except Exception as e:
        print(f"Simulation WebSocket error: {e}")
    finally:
        print("Simulation WebSocket connection closed")

@app.websocket("/ws/real-model")
async def websocket_real_model(websocket: WebSocket):
    """WebSocket for real XGBoost model predictions"""
    await websocket.accept()
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "mode": "real_model",
            "message": "Connected to Real XGBoost Model",
            "model_auc": app.state.real_engine.stats["model_auc"],
            "features": len(app.state.real_engine.features['feature_names']),
            "performance": "39.7% detection @ 8.8% false alarms"
        })
        
        batch = []
        batch_size = 100
        
        while True:
            try:
                # Check if engine is running
                if not app.state.real_engine.is_running:
                    await asyncio.sleep(0.001)
                    continue
                
                # Process real batch
                transactions = await app.state.real_engine.process_batch(
                    batch_size=batch_size,
                    threshold=app.state.real_engine.stats["threshold"]
                )
                
                batch.extend(transactions)
                
                # Send batch when full
                if len(batch) >= batch_size:
                    try:
                        await websocket.send_json({
                            "type": "batch",
                            "transactions": batch,
                            "stats": app.state.real_engine.stats,
                            "mode": "real_model",
                            "batch_size": len(batch),
                            "model": "XGBoost v5",
                            "performance": "39.7% detection @ 8.8% false alarms"
                        })
                        batch = []
                    except Exception as e:
                        print(f"Error sending real model batch: {e}")
                        break
                
                # Control speed - similar to simulation
                await asyncio.sleep(0.00001)  
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in real model loop: {e}")
                break
                
    except WebSocketDisconnect:
        print("Real model WebSocket client disconnected")
    except Exception as e:
        print(f"Real model WebSocket error: {e}")
    finally:
        print("Real model WebSocket connection closed")

# ==================== REST API ENDPOINTS ====================
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Fraud Detection API",
        "status": "running",
        "performance": "39.7% fraud detection @ 8.8% false alarms",
        "web_sockets": {
            "simulation": "ws://localhost:8000/ws/simulation",
            "real_model": "ws://localhost:8000/ws/real-model"
        },
        "rest_api": {
            "modes": "/api/modes",
            "start_mode": "/api/mode/{mode_id}/start",
            "stats": "/api/stats/{mode_id}"
        }
    }

@app.get("/api/modes")
async def get_available_modes():
    """Get available detection modes"""
    return {
        "modes": [
            {
                "id": "simulation",
                "name": "Simulation Mode",
                "description": "Real-time simulation matching XGBoost performance",
                "speed": "~6,000 tx/sec (demo speed)",
                "performance": "39.7% detection @ 8.8% false alarms",
                "type": "demo",
                "websocket": "ws://localhost:8000/ws/simulation"
            },
            {
                "id": "real_model", 
                "name": "Real XGBoost Model",
                "description": "Actual XGBoost v5 predictions on real data",
                "speed": "~6,000 tx/sec (demo) / 50,000+ tx/sec (batch)",
                "performance": "39.7% detection @ 8.8% false alarms",
                "auc": app.state.real_engine.stats["model_auc"],
                "type": "production",
                "websocket": "ws://localhost:8000/ws/real-model"
            }
        ]
    }

@app.post("/api/simulation/{action}")
async def control_simulation_legacy(action: str):
    """Legacy endpoint for backward compatibility"""
    if action == "start":
        app.state.simulation_engine.is_running = True
        return {"status": "started", "message": "Simulation running"}
    elif action == "stop":
        app.state.simulation_engine.is_running = False
        return {"status": "stopped", "message": "Simulation paused"}
    else:
        return {"error": "Invalid action. Use 'start' or 'stop'"}

@app.post("/api/mode/{mode_id}/start")
async def start_mode(mode_id: str):
    """Start a specific mode"""
    if mode_id == "simulation":
        app.state.simulation_engine.is_running = True
        return {"mode": "simulation", "status": "started"}
    elif mode_id == "real_model":
        app.state.real_engine.is_running = True
        return {"mode": "real_model", "status": "started", "model": "XGBoost v5"}
    else:
        return {"error": "Invalid mode"}

@app.post("/api/mode/{mode_id}/stop")
async def stop_mode(mode_id: str):
    """Stop a specific mode"""
    if mode_id == "simulation":
        app.state.simulation_engine.is_running = False
        return {"mode": "simulation", "status": "stopped"}
    elif mode_id == "real_model":
        app.state.real_engine.is_running = False
        return {"mode": "real_model", "status": "stopped"}
    else:
        return {"error": "Invalid mode"}

@app.get("/api/stats/{mode_id}")
async def get_mode_stats(mode_id: str):
    """Get statistics for a specific mode"""
    if mode_id == "simulation":
        return app.state.simulation_engine.stats
    elif mode_id == "real_model":
        return app.state.real_engine.stats
    else:
        return {"error": "Invalid mode"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "simulation_running": app.state.simulation_engine.is_running,
        "real_model_running": app.state.real_engine.is_running,
        "performance": "39.7% detection @ 8.8% false alarms"
    }

@app.post("/api/reset/{mode_id}")
async def reset_mode_stats(mode_id: str):
    """Reset statistics on the backend"""
    if mode_id == "simulation":
        # Reset simulation engine stats
        app.state.simulation_engine.stats = {
            "total_processed": 0,
            "fraud_detected": 0,
            "missed_fraud": 0,
            "false_alarms": 0,
            "detection_rate": 0.0,
            "processing_speed": 0,
            "alert_rate": 0.0,
            "threshold": 0.063
        }
        # Reset internal counters
        app.state.simulation_engine.last_calc_time = time.time()
        app.state.simulation_engine.last_count = 0
        app.state.simulation_engine.event_buffer.clear()
        
        return {"status": "reset", "mode": "simulation", "message": "Simulation stats reset to zero"}
        
    elif mode_id == "real_model":
        # Reset real engine stats
        app.state.real_engine.stats = {
            "total_processed": 0,
            "fraud_detected": 0,
            "missed_fraud": 0,
            "false_alarms": 0,
            "detection_rate": 0.0,
            "processing_speed": 0.0,
            "alert_rate": 0.0,
            "threshold": 0.063,
            "model_auc": app.state.real_engine.stats["model_auc"],  # Keep AUC
            "batch_size": 1000
        }
        # Reset internal counters
        app.state.real_engine.total_processed = 0
        app.state.real_engine.current_batch = 0
        app.state.real_engine.last_calc_time = time.time()
        app.state.real_engine.last_count = 0
        app.state.real_engine.event_buffer.clear()
        
        return {"status": "reset", "mode": "real_model", "message": "Real model stats reset to zero"}
    else:
        return {"error": "Invalid mode"}

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )