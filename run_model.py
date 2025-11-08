import pandas as pd
import numpy as np
import time
from datetime import datetime
from model_utils import load_model_version, load_current_model, list_model_versions

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
                  f"Amount: ₹{row['TransactionAmt']:8.2f} | "
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
    """Main function with user choice"""
    print("🛡️  FRAUD DETECTION DEMO LAUNCHER")
    print("="*50)
    
    # Show available versions
    list_model_versions()
    
    print("\nChoose loading option:")
    print("1. Load current/latest model")
    print("2. Load specific version")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\nLoading current model...")
        model, features = load_current_model()
        print(f"✅ Loaded model type: {model.__class__.__name__}")
        version_info = "Current Model"
        
    elif choice == "2":
        version_name = input("Enter version name (e.g., v1_20241215_143022): ").strip()
        print(f"\nLoading version: {version_name}")
        result = load_model_version(version_name)
        if result is None:
            return
        model, features, performance = result
        version_info = f"Version: {version_name} (AUC: {performance['roc_auc']:.3f})"
        
    elif choice == "3":
        print("Goodbye!")
        return
    else:
        print("Invalid choice!")
        return
    
    # Load demo data
    try:
        if choice == "1":
            import glob
            version_dirs = glob.glob("model_versions/v*")
            if version_dirs:
                latest_version = sorted(version_dirs)[-1]
                demo_data_path = f"{latest_version}/demo_data.csv"
                demo_data = pd.read_csv(demo_data_path)
            else:
                print("❌ No demo data found! Train a model first.")
                return
        else:
            demo_data_path = f"model_versions/{version_name}/demo_data.csv"
            demo_data = pd.read_csv(demo_data_path)
    except Exception as e:
        print(f"❌ Error loading demo data: {e}")
        return
    
    print(f"\n✅ Successfully loaded {version_info}")
    print(f"📁 Features: {len(features['feature_names'])}")
    print(f"📊 Demo transactions available: {len(demo_data)}")
    
    # Speed choice
    print("\nChoose processing speed:")
    print("1. Slow (1s delay - good for presentations)")
    print("2. medium (0.1s delay - balanced)") 
    print("3. fast (0.01s delay - 1000 transactions in ~10 seconds)")
    
    speed_choice = input("\nEnter speed choice (1-3): ").strip()
    speed_map = {'1': 'slow', '2': 'medium', '3': 'fast'}
    speed = speed_map.get(speed_choice, 'fast')
    
    # Transaction count
    try:
        max_tx = int(input(f"\nEnter number of transactions to process (1-{len(demo_data)}): ") or "1000")
        max_tx = min(max_tx, len(demo_data))
    except:
        max_tx = 1000

    '''
    # Add threshold choice
    print("\nChoose detection sensitivity:")
    print("1. High precision (threshold: 0.5) - Fewer false alarms")
    print("2. Balanced (threshold: 0.3) - Good mix")
    print("3. High recall (threshold: 0.2) - Catch more fraud")
    print("4. Custom threshold")
    
    threshold_choice = input("\nEnter choice (1-4): ").strip()
    
    if threshold_choice == "1":
        threshold = 0.5
    elif threshold_choice == "2":
        threshold = 0.3  
    elif threshold_choice == "3":
        threshold = 0.2
    elif threshold_choice == "4":'''
    try:
        threshold = float(input("Enter custom threshold (0.01-0.99): "))
    except:
        threshold = 0.01
    '''else:
        threshold = 0.3'''
    
    print(f"🎯 Using detection threshold: {threshold}")
    
    # Start demo with threshold
    input(f"\nPress Enter to start live fraud detection demo (threshold: {threshold})...")
    live_fraud_demo(model, features, demo_data, max_transactions=max_tx, speed=speed, threshold=threshold)
    
if __name__ == "__main__":
    main()