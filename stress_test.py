import pandas as pd
import numpy as np
import random
import joblib
import os
import json
from datetime import datetime
from model_utils import load_model_version, load_current_model, list_model_versions

def load_current_model():
    """Load the current/latest model - copied from your model_utils"""
    return joblib.load('current_model.joblib'), joblib.load('current_features.joblib')

def create_stress_test_cases():
    """Generate challenging edge cases that break typical fraud models"""
    print(" CREATING ADVERSARIAL STRESS TEST CASES...")
    
    stress_cases = []
    
    # Scenario 1: SLEEPER ACCOUNTS (long-term legitimate then exploited)
    for i in range(50):
        stress_cases.append({
            'TransactionAmt': random.randint(50000, 200000),  # Very high amount
            'amount_deviation_ratio': 10.0,  # 10x user's normal spending
            'tx_count_user': 100,  # Established user (makes it harder to detect)
            'is_new_merchant': 0,  # Known merchant
            'unusual_time': 0,     # Normal hours
            'hour_of_day': 14,
            'is_weekend': 0,
            'is_night': 0,
            'card1': 1000 + i, 'card2': 1, 'card3': 150, 'card4': 1, 'card5': 100,
            'ProductCD': 2,  # Common product
            'addr1': 100, 'addr2': 1,
            'DeviceType': 1,
            'isFraud': 1,  # THIS IS FRAUD - but looks legitimate!
            'scenario': 'sleeper_account_exploit'
        })
    
    # Scenario 2: FALSE POSITIVE NIGHTMARE (everything suspicious but legitimate)
    for i in range(50):
        stress_cases.append({
            'TransactionAmt': random.randint(50000, 100000),  # High amount
            'amount_deviation_ratio': 8.0,  # High deviation
            'tx_count_user': 2,  # New user
            'is_new_merchant': 1,  # New merchant
            'unusual_time': 1,     # Unusual time
            'hour_of_day': 3,      # 3 AM
            'is_weekend': 1,       # Weekend
            'is_night': 1,         # Night time
            'card1': 2000 + i, 'card2': 1, 'card3': 150, 'card4': 1, 'card5': 100,
            'ProductCD': 0,  # High-risk product
            'addr1': 500, 'addr2': 2,  # Unusual location
            'DeviceType': 0,  # New device
            'isFraud': 0,  # LEGITIMATE - but everything screams fraud!
            'scenario': 'false_positive_nightmare'
        })
    
    # Scenario 3: LOW-AND-SLOW ATTACK (small amounts that fly under radar)
    for i in range(100):
        stress_cases.append({
            'TransactionAmt': random.randint(100, 500),  # Small amounts
            'amount_deviation_ratio': 1.2,  # Slightly above normal
            'tx_count_user': 5,  # Some history
            'is_new_merchant': 0,  # Known merchants
            'unusual_time': 0,     # Normal patterns
            'hour_of_day': random.randint(9, 18),
            'is_weekend': random.choice([0, 1]),
            'is_night': 0,
            'card1': 3000 + i, 'card2': 1, 'card3': 150, 'card4': 1, 'card5': 100,
            'ProductCD': random.randint(0, 4),
            'addr1': random.randint(50, 200), 'addr2': 1,
            'DeviceType': 1,
            'isFraud': 1,  # FRAUD - but designed to avoid detection
            'scenario': 'low_and_slow_attack'
        })
    
    # Scenario 4: PERFECT STORM (combination attack)
    for i in range(30):
        stress_cases.append({
            'TransactionAmt': random.randint(1000, 5000),
            'amount_deviation_ratio': 2.5,
            'tx_count_user': 20,
            'is_new_merchant': 1,
            'unusual_time': 1,
            'hour_of_day': random.choice([2, 3, 4]),  # Very early morning
            'is_weekend': 0,
            'is_night': 1,
            'card1': 4000 + i, 'card2': 2, 'card3': 200, 'card4': 0, 'card5': 50,
            'ProductCD': 0,  # High-risk product category
            'addr1': 300, 'addr2': 2,
            'DeviceType': 0,  # Mobile device
            'isFraud': 1,
            'scenario': 'perfect_storm_attack'
        })
    
    return pd.DataFrame(stress_cases)

def run_stress_test(model, features, threshold=0.11):
    """Run the stress test and see how the model breaks"""
    print("" * 20)
    print(" ADVERSARIAL STRESS TEST")
    print("" * 20)
    
    print(f" Testing model: {model.__class__.__name__}")
    print(f" Using threshold: {threshold}")
    
    # Generate stress test cases
    stress_data = create_stress_test_cases()
    print(f" Generated {len(stress_data)} adversarial test cases")
    
    # Extract features and true labels
    X_stress = stress_data[features['feature_names']]
    y_true = stress_data['isFraud']
    scenarios = stress_data['scenario']
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_stress)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Overall performance
    print(f"\n📈 OVERALL STRESS TEST PERFORMANCE")
    print(f"Total test cases: {len(stress_data)}")
    print(f"Actual frauds: {y_true.sum()}")
    print(f"Detected frauds: {y_pred.sum()}")
    print(f"Stress Test Recall: {(y_pred[y_true == 1].sum() / y_true.sum() * 100):.1f}%")
    
    # Performance by scenario
    print(f"\n🔍 BREAKDOWN BY ATTACK SCENARIO:")
    for scenario in stress_data['scenario'].unique():
        scenario_mask = scenarios == scenario
        scenario_true = y_true[scenario_mask]
        scenario_pred = y_pred[scenario_mask]
        
        recall = (scenario_pred[scenario_true == 1].sum() / scenario_true.sum() * 100) if scenario_true.sum() > 0 else 0
        false_positives = scenario_pred[scenario_true == 0].sum()
        
        print(f"   {scenario}:")
        print(f"      Cases: {scenario_mask.sum()}, Frauds: {scenario_true.sum()}")
        print(f"      Recall: {recall:.1f}%, False Positives: {false_positives}")
    
    # Detailed analysis of failures
    print(f"\n❌ ANALYSIS OF FAILED DETECTIONS:")
    failed_frauds = stress_data[(y_true == 1) & (y_pred == 0)]
    if len(failed_frauds) > 0:
        print("Undetected fraud patterns:")
        for scenario in failed_frauds['scenario'].unique():
            count = (failed_frauds['scenario'] == scenario).sum()
            print(f"   {scenario}: {count} cases missed")
    
    # False positive analysis
    print(f"\n⚠️ ANALYSIS OF FALSE POSITIVES:")
    false_positives = stress_data[(y_true == 0) & (y_pred == 1)]
    if len(false_positives) > 0:
        print("Legitimate transactions flagged as fraud:")
        for scenario in false_positives['scenario'].unique():
            count = (false_positives['scenario'] == scenario).sum()
            print(f"   {scenario}: {count} false alarms")
    
    # Most challenging cases
    print(f"\n🎯 MOST CHALLENGING CASES (Probability close to threshold):")
    borderline_cases = stress_data[(y_pred_proba > threshold-0.03) & (y_pred_proba < threshold+0.03)]
    for idx, row in borderline_cases.head(5).iterrows():
        prob = y_pred_proba[stress_data.index == idx][0]
        actual = row['isFraud']
        scenario = row['scenario']
        print(f"   {scenario}: prob={prob:.3f}, actual={'FRAUD' if actual else 'LEGIT'}")
    
    return stress_data, y_pred_proba, y_true

def stress_test_demo(model, features, threshold=0.11):
    """Interactive stress test demonstration"""
    print(" STRESS TEST DEMO MODE")
    print("Testing model against sophisticated attack scenarios...\n")
    
    stress_data, probabilities, true_labels = run_stress_test(model, features, threshold)
    
    # Show some specific challenging cases
    print(f"\n" + "="*60)
    print(" CASE STUDIES: WHY THESE ARE HARD TO DETECT")
    print("="*60)
    
    # Find some interesting cases
    hard_frauds = stress_data[(true_labels == 1) & (probabilities < threshold)]
    false_alarms = stress_data[(true_labels == 0) & (probabilities > threshold)]
    
    if len(hard_frauds) > 0:
        print(f"\n HARD-TO-DETECT FRAUDS (Model probability < {threshold}):")
        for idx, row in hard_frauds.head(3).iterrows():
            prob = probabilities[stress_data.index == idx][0]
            print(f"   {row['scenario']}:")
            print(f"      Amount: ₹{row['TransactionAmt']}")
            print(f"      User History: {row['tx_count_user']} transactions")
            print(f"      Model Confidence: {prob:.3f}")
            print(f"      Why it's hard: Looks like normal user behavior!")
    
    if len(false_alarms) > 0:
        print(f"\n FALSE ALARMS (Legitimate but flagged):")
        for idx, row in false_alarms.head(3).iterrows():
            prob = probabilities[stress_data.index == idx][0]
            print(f"   {row['scenario']}:")
            print(f"      Amount: ₹{row['TransactionAmt']}")
            print(f"      Model Confidence: {prob:.3f}")
            print(f"      Why it's flagged: Multiple risk factors combined!")

def main():
    """Main function with model selection - just like run_model.py"""
    print(" ADVERSARIAL STRESS TEST LAUNCHER")
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
        version_name = input("Enter version name (e.g., v1_rf_20241215_143022): ").strip()
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
    
    print(f"\n✅ Successfully loaded {version_info}")
    print(f"📁 Features: {len(features['feature_names'])}")
    
    # Threshold selection
    try:
        threshold = float(input(f"\nEnter detection threshold (default 0.11): ") or "0.11")
    except:
        threshold = 0.11
    
    print(f"🎯 Using detection threshold: {threshold}")
    
    # Start stress test
    input(f"\nPress Enter to start adversarial stress test...")
    stress_test_demo(model, features, threshold)

if __name__ == "__main__":
    main()