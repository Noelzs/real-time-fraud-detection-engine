# Real-Time Fraud Detection Engine

A proof-of-concept fraud detection system that identifies suspicious credit card transactions in real-time. Built to demonstrate how machine learning can catch fraud without annoying legitimate customers.

> Note: This is a proof-of-concept, not a production system. It's designed to show what's possible, not handle millions of transactions.

## What's This All About?

Imagine you're a bank. People use your credit cards for shopping. Most transactions are legit, but some are fraud. This system helps you:

## What's in This Project?

fraud-detection/
├── requirements.txt    # What you need to install
├── model_utils.py      # Save/load different model versions
├── train_model.py      # Train the brain (ML models)
├── model_versions/     # All your trained models live here
├── run_model.py        # Live fraud detection demo
└── stress_test.py      # Test against tricky fraud attempts

## Quick Start

### 1. **Get the Data**
First, grab the dataset from [IEEE-CIS Fraud Detection on Kaggle](https://www.kaggle.com/c/ieee-fraud-detection).
You'll need these files:
- train_transaction.csv
- train_identity.csv

Put them in your project folder.

### 2. **Setup**
# Install what you need
pip install -r requirements.txt

### 3. **Train Your First Model**
python train_model.py

This creates two models:

Random Forest - Reliable, explainable
XGBoost - Fast, sophisticated

### 4. **See It in Action!**
python run_model.py

Here you get to choose:
    Speed: Slow (see every transaction) or Fast (1000s per second)

    Model: Pick which model to use

    Sensitivity: How strict should we be?(threshold)

### 5. Stress Test It

python stress_test.py

## Additional features:

Every time you train, a new version is saved. Compare:

    v1_rf_20241215_143022 - Random Forest from Dec 15

    v2_xg_20241215_143025 - XGBoost from 3 seconds later

## Adversarial Testing

We generate fraud scenarios such as:

    Sleeper Agents: Good customers gone bad

    Slow Drips: Tiny frauds that add up

    Perfect Storms: Everything looks wrong but it's legit

## What to Expect

On the test data, you'll typically see:

    Detection Rate: ~40-45% of actual fraud caught

    False Alarms: ~5-6% of legit transactions flagged

    Speed: 1,500+ transactions per second

    You may ask- wait only 40%? Yes - We tried to balance catching fraud vs annoying customers. The stress test shows we can hit 90%+ detection if we're willing to flag more transactions.

## Important Notes

    This uses synthetic data patterns based on real fraud

    TransactionDT is time delta, not timestamp! (I came to know it the hard way, DT here doesnt refer to DateTime)

    Performance depends heavily on your feature engineering

    Real production systems would need more data and features

## License


This project is for educational purposes. Make sure you have the right to use the IEEE-CIS dataset from Kaggle.
