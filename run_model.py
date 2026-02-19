import pandas as pd
import numpy as np
import time
from datetime import datetime
from model_utils import load_model_version, list_model_versions

def live_fraud_demo(model, features, transaction_data, max_transactions=10000, speed='fast', threshold=0.01):
    """Running a live simulation"""
    print("\n" + "="*60)
    print("LIVE FRAUD DETECTION DEMO")
    print("="*60)
    
    fraud_count = 0
    total_count = 0
    detected_frauds = 0
    missed_frauds = 0
    
    # Speed settings
    speed_config = {
        'slow': 1.0,     # 1 seconds between transactions  
        'medium': 0.1,   # 0.1 seconds between transactions
        'fast': 0.01     # 0.01 seconds between transactions (1000 in 10 seconds)
    }
    
    delay = speed_config.get(speed, 0.1)
    
    print(f"Starting transaction stream... (Speed: {speed}, Delay: {delay}s)")
    print(f"Processing {min(max_transactions, len(transaction_data))} transactions...\n")
    
    start_time = time.time()
    
    for idx, row in transaction_data.iterrows():
        if total_count >= max_transactions:
            break
            
        # Prepare transaction for prediction
        transaction_features = row[features['feature_names']].values.reshape(1, -1)
        
        # Get fraud probability
        fraud_prob = model.predict_proba(transaction_features)[0][1]
        
        # Display transaction with color coding
        is_fraud = fraud_prob > threshold
        is_actual_fraud = row['isFraud'] == 1
        
        if is_fraud:
            fraud_count += 1
            alert_icon = "🚨"
        else:
            alert_icon = "✅"
            
        total_count += 1
        
        # Only show details in slow/medium mode, or when fraud is detected
        if speed in ['slow', 'medium'] or is_fraud or is_actual_fraud:
            print(f"{alert_icon} Transaction {total_count:04d} | "
                  f"Amount: {row['TransactionAmt']:8.2f} | "
                  f"Fraud Prob: {fraud_prob:.3f} | "
                  f"Time: {datetime.now().strftime('%H:%M:%S')}")
            
            if is_fraud:
                print(f"   ⚠️  ALERT! Unusual patterns detected!")
            
            # Show actual fraud status
            if is_actual_fraud and is_fraud:
                print(f"   🎯 CORRECT! This was actual fraud!")
                detected_frauds += 1
            elif is_actual_fraud and not is_fraud:
                print(f"   ❌ MISSED! This was actual fraud but not detected!")
                missed_frauds += 1
            
            if speed in ['slow', 'medium']:
                print("-" * 70)
        
        # Progress indicator for fast mode
        if speed == "fast" and total_count % 100 == 0:
            print(f"📊 Processed {total_count} transactions...")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Demo summary
    print("\n" + "="*60)
    print("📊 DEMO SUMMARY")
    print("="*60)
    print(f"Total transactions processed: {total_count}")
    print(f"Fraud alerts triggered: {fraud_count}")
    print(f"Actual frauds detected: {detected_frauds}")
    print(f"Actual frauds missed: {missed_frauds}")
    print(f"Alert rate: {(fraud_count/total_count)*100:.1f}%")
    print(f"Processing speed: {total_count/processing_time:.1f} transactions/second")
    print(f"Total time: {processing_time:.1f} seconds")
    print(f"Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Detection threshold: {threshold}")
    print(f"Fraud detection rate: {detected_frauds/(detected_frauds + missed_frauds)*100:.1f}%")

def main():
    """Main function - simplified version selection"""
    print("🛡️  FRAUD DETECTION DEMO LAUNCHER")
    print("="*50)
    
    # Show available versions
    list_model_versions()
    
    # Get version name
    version_name = input("\nEnter version name (e.g., v5_xg_20251109_154848): ").strip()
    print(f"\nLoading version: {version_name}")
    
    result = load_model_version(version_name)
    if result is None:
        return
    
    model, features, performance = result
    version_info = f"Version: {version_name} (AUC: {performance['roc_auc']:.3f})"
    
    # Load demo data
    try:
        demo_data_path = f"model_versions/{version_name}/demo_data.csv"
        demo_data = pd.read_csv(demo_data_path)
    except Exception as e:
        print(f"❌ Error loading demo data: {e}")
        return
    
    print(f"\n✅ Successfully loaded {version_info}")
    print(f"📁 Features: {len(features['feature_names'])}")
    print(f"📊 Demo transactions available: {len(demo_data)}")
    
    # Speed choice (default to fast)
    print("\nChoose processing speed:")
    print("1. Slow (1s delay - good for presentations)")
    print("2. Medium (0.1s delay - balanced)") 
    print("3. Fast (0.01s delay - 1000 transactions in ~10 seconds) [DEFAULT]")
    
    speed_choice = input("\nEnter speed choice (1-3, default 3): ").strip()
    speed_map = {'1': 'slow', '2': 'medium', '3': 'fast'}
    speed = speed_map.get(speed_choice, 'fast')
    
    # Transaction count (default to 10,000)
    try:
        max_tx = int(input(f"\nEnter number of transactions to process (1-{len(demo_data)}, default 10000): ") or "10000")
        max_tx = min(max_tx, len(demo_data))
    except:
        max_tx = 10000
    
    # Transaction threshold
    try:
        threshold = float(input("Enter custom threshold (0.01-0.99, default 0.063): ") or "0.063")
    except:
        threshold = 0.063
    
    print(f"🎯 Using detection threshold: {threshold}")
    
    # Start demo with threshold
    input(f"\nPress Enter to start live fraud detection demo (threshold: {threshold})...")
    live_fraud_demo(model, features, demo_data, max_transactions=max_tx, speed=speed, threshold=threshold)
    
if __name__ == "__main__":
    main()
