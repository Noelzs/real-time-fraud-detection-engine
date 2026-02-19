import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from model_utils import save_model_version


# Load the data
print("Loading data...")
train_trans = pd.read_csv('train_transaction.csv')
train_identity = pd.read_csv('train_identity.csv')

print(f"Transaction data: {train_trans.shape}")
print(f"Identity data: {train_identity.shape}")

# Merge them
df = pd.merge(train_trans, train_identity, on='TransactionID', how='left')
print(f"Merged data: {df.shape}")

# Note from Noel: TransactionDT is TIMEDELTA NOT TIMESTAMP!
df['TransactionDT'] = df['TransactionDT'].astype(int)

# Create time-based features
df['hour_of_day'] = (df['TransactionDT'] // 3600) % 24
df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_night'] = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] <= 5)).astype(int)

print("Time features created!")
print(df[['TransactionDT', 'hour_of_day', 'day_of_week', 'is_weekend']].head())

def create_real_time_features(df):
    """Create features that simulate real-time detection"""
    
    # Sort by time for rolling calculations
    df = df.sort_values('TransactionDT').reset_index(drop=True)
   
    # Feature 1: Amount deviation (user-level)
    user_median_amount = df.groupby('card1')['TransactionAmt'].transform('median')
    df['amount_deviation_ratio'] = df['TransactionAmt'] / (user_median_amount + 1)
    
    # Feature 2: User transaction count (simpler and more reliable)
    df['tx_count_user'] = df.groupby('card1')['TransactionID'].transform('count')
    
    # Feature 3: New merchant flag (simplified)
    common_merchants = df.groupby('card1')['ProductCD'].agg(lambda x: x.value_counts().index[0] if len(x) > 0 else 'X')
    df['common_merchant'] = df['card1'].map(common_merchants)
    df['is_new_merchant'] = (df['ProductCD'] != df['common_merchant']).astype(int)
    
    # Feature 4: Unusual time for user
    user_common_hour = df.groupby('card1')['hour_of_day'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 12)
    df['user_common_hour'] = df['card1'].map(user_common_hour)
    df['unusual_time'] = (abs(df['hour_of_day'] - df['user_common_hour']) > 4).astype(int)

     # User's normal spending patterns (NEW - replaces old approach)
    df['user_amount_95th'] = df.groupby('card1')['TransactionAmt'].transform(
    lambda x: x.quantile(0.95) if len(x) > 10 else x.median())

    df['amount_vs_normal'] = df['TransactionAmt'] / df['user_amount_95th']

    # Better time deviation (NEW - replaces unusual_time)
    df['hour_deviation'] = abs(df['hour_of_day'] - df['user_common_hour'])
    
    # Feature 5: High-risk product categories
    df['high_risk_product'] = df['ProductCD'].isin(['W', 'S']).astype(int)

    # Feature 6: Time since last transaction (HUGE for fraud)
    df['time_since_last_tx'] = df.groupby('card1')['TransactionDT'].diff()

    # Instead of time-based rolling, use count-based rolling
    df['tx_frequency_change'] = df.groupby('card1')['TransactionAmt'].transform(
    lambda x: x.rolling(5, min_periods=1).std() / (x.rolling(5, min_periods=1).mean() + 1))

    # Merchant risk profiling
    merchant_fraud_rates = df.groupby('ProductCD')['isFraud'].mean()
    df['merchant_risk_score'] = df['ProductCD'].map(merchant_fraud_rates)
    
    return df

# Add this BEFORE calling create_real_time_features()
df = df.sample(100000, random_state=42)  # Use 100K samples for training
print(f"Sampled data: {df.shape}")

print("Creating real-time features...")
df = create_real_time_features(df)

# Select features that make sense for real-time detection
feature_columns = [
    # Core transaction features
    'TransactionAmt',
    
    # Our engineered real-time features
    'amount_deviation_ratio',
    #Note from Noel:'tx_velocity_24h'(The complex velocity feature was mathematically elegant but conceptually flawed for this dataset since TransactionDT is timedelta and not a timestamp.), 
    'tx_count_user',
    'is_new_merchant',
    'unusual_time',
    
    # Time patterns
    'hour_of_day',
    'is_weekend',
    'is_night',
    
    # Simple card features (available in real-time)
    'card1', 'card2', 'card3', 'card4', 'card5',
    
    # Product context
    'ProductCD',
    
    # Geographic (simplified)
    'addr1', 'addr2',
    
    # Device info (if available)
    'DeviceType',
]

# Filter to available columns
available_features = [col for col in feature_columns if col in df.columns]
print(f"Using {len(available_features)} features: {available_features}")

# Prepare X and y
X = df[available_features].copy()
y = df['isFraud'].copy()

# Handle categorical variables
categorical_cols = ['card4', 'ProductCD', 'DeviceType']
for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].fillna('Missing')
        X[col] = X[col].astype('category').cat.codes
    else:
        print(f"{col} not available, skipping")

# Fill remaining NaN
X = X.fillna(-999)

print(f"Final feature matrix: {X.shape}")
print(f"Fraud rate: {y.mean():.3f}")

# Split by time to simulate real-world scenario
# Use first 80% for training, last 20% for testing
split_point = int(0.8 * len(df))
train_mask = df.index < split_point
test_mask = df.index >= split_point

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train set: {X_train.shape}, Fraud rate: {y_train.mean():.3f}")
print(f"Test set: {X_test.shape}, Fraud rate: {y_test.mean():.3f}")

# Simple Random Forest (good for POC)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# Predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# XGBOOST COMPARISON

print("\n" + "="*50)
print("🧪 XGBOOST COMPARISON")
print("="*50)

from xgboost import XGBClassifier

# Train XGBoost on same features
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)

# XGBoost predictions
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_proba_xgb > 0.5).astype(int)

print(f"Random Forest AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"XGBoost AUC: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")

print("\n" + "="*60)
print("🔍 MODEL DIAGNOSTICS")
print("="*60)

# 1. Check what probabilities XGBoost is outputting
fraud_indices = y_test[y_test == 1].index
fraud_probabilities = y_pred_proba_xgb[y_test == 1]
non_fraud_probabilities = y_pred_proba_xgb[y_test == 0]

print(f"Actual fraud cases in test set: {len(fraud_probabilities)}")
print(f"Actual non-fraud cases in test set: {len(non_fraud_probabilities)}")

print(f"\n📊 Fraud transaction probabilities:")
print(f"   Min: {fraud_probabilities.min():.6f}")
print(f"   Max: {fraud_probabilities.max():.6f}") 
print(f"   Mean: {fraud_probabilities.mean():.6f}")
print(f"   Median: {np.median(fraud_probabilities):.6f}")

print(f"\n📊 Legit transaction probabilities:")
print(f"   Min: {non_fraud_probabilities.min():.6f}")
print(f"   Max: {non_fraud_probabilities.max():.6f}")
print(f"   Mean: {non_fraud_probabilities.mean():.6f}")

# 2. Check how many frauds would be caught at different thresholds
print(f"\n🎯 Fraud detection at different thresholds:")
for threshold in [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]:
    frauds_caught = (fraud_probabilities > threshold).sum()
    false_positives = (non_fraud_probabilities > threshold).sum()
    print(f"   Threshold {threshold:.4f}: {frauds_caught}/{len(fraud_probabilities)} frauds caught, {false_positives} false alarms")

# 3. Check if model is just predicting everything as non-fraud
print(f"\n🤖 Model prediction distribution:")
unique_preds, counts = np.unique(y_pred_xgb, return_counts=True)
for pred, count in zip(unique_preds, counts):
    print(f"   Predict {pred}: {count} transactions")

# 4. Check a few specific fraud cases
print(f"\n🔎 Examining specific fraud cases (first 5):")
fraud_cases = X_test.loc[y_test == 1].head(5)
for i, (idx, row) in enumerate(fraud_cases.iterrows()):
    prob = xgb_model.predict_proba([row])[0][1]
    actual_fraud = y_test.loc[idx]
    print(f"   Fraud case {i+1}: probability = {prob:.6f}, actual = {actual_fraud}")

# 5. Check model training performance
print(f"\n📈 Model training performance:")
print(f"   Training score: {xgb_model.score(X_train, y_train):.4f}")
print(f"   Test score: {xgb_model.score(X_test, y_test):.4f}")

# 6. Check feature importance
print(f"\n📋 Top 10 feature importances:")
xgb_importance = pd.DataFrame({
    'feature': available_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(xgb_importance.head(10))

# Compare performance
improvement = roc_auc_score(y_test, y_pred_proba_xgb) - roc_auc_score(y_test, y_pred_proba)
print(f"Improvement: {improvement:+.4f}")


print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# See what features matter most
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'].head(10)[::-1], 
         feature_importance['importance'].head(10)[::-1])
plt.xlabel('Feature Importance')
plt.title('Top 10 Fraud Detection Features')
plt.tight_layout()
plt.show()

# Create the feature_info dictionary
feature_info = {
    'feature_names': available_features,
    'categorical_columns': categorical_cols
}

# Get XGBoost feature importance
xgb_feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Create performance metrics for XGBoost
xgb_performance = {
    'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb),
    'classification_report': classification_report(y_test, y_pred_xgb, output_dict=True),
    'confusion_matrix': confusion_matrix(y_test, y_pred_xgb).tolist()
}

# Create performance metrics for Random Forest  
rf_performance = {
    'roc_auc': roc_auc_score(y_test, y_pred_proba),
    'classification_report': classification_report(y_test, y_pred, output_dict=True),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
}

# Create demo data sample - USE PROCESSED FEATURES
demo_data = X_test.sample(10000, random_state=42).copy()
demo_data['isFraud'] = y_test[demo_data.index]  # Add target for demo tracking

# Save both models
print("\n" + "="*50)
print("💾 SAVING BOTH MODELS")
print("="*50)

# Save Random Forest
rf_version_note = f"RandomForest- AUC: {roc_auc_score(y_test, y_pred_proba):.4f}"
rf_version_name = save_model_version(
    model=model,                   
    features=feature_info,
    performance=rf_performance,        
    demo_data=demo_data,
    feature_importance=feature_importance, 
    version_note=rf_version_note,
    model_type="rf"
)

# Save XGBoost  
xgb_version_note = f"XGBoost- AUC: {roc_auc_score(y_test, y_pred_proba_xgb):.4f} (+{improvement:.4f} improvement)"
xgb_version_name = save_model_version(
    model=xgb_model,                   
    features=feature_info,
    performance=xgb_performance,        
    demo_data=demo_data,
    feature_importance=xgb_feature_importance, 
    version_note=xgb_version_note,
    model_type="xg"
)

print(f"\n🎉 Both models saved successfully!")
print(f"🌲 Random Forest: {rf_version_name}")
print(f"🤖 XGBoost: {xgb_version_name}")
