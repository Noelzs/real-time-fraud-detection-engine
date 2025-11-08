import os
import datetime
import joblib
import json

def save_model_version(model, features, performance, demo_data, feature_importance, 
                      version_note="", version_name=None, model_type=""):
    """Save everything with automatic versioning and model type"""
    
    # Create versions directory if it doesn't exist
    os.makedirs('model_versions', exist_ok=True)
    
    # Generate version name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if version_name:
        # Use custom version name with model type
        version_name = f"{version_name}_{model_type}_{timestamp}"
    else:
        # Auto-generate version name with model type
        existing_versions = len([d for d in os.listdir('model_versions') 
                               if os.path.isdir(os.path.join('model_versions', d))])
        version_name = f"v{existing_versions + 1}_{model_type}_{timestamp}"
    
    # Create version directory
    version_dir = f"model_versions/{version_name}"
    os.makedirs(version_dir, exist_ok=True)
    
    # Save everything in the version directory
    joblib.dump(model, f'{version_dir}/model.joblib')
    joblib.dump(features, f'{version_dir}/features.joblib')
    joblib.dump(performance, f'{version_dir}/performance.joblib')
    demo_data.to_csv(f'{version_dir}/demo_data.csv', index=False)
    feature_importance.to_csv(f'{version_dir}/feature_importance.csv', index=False)
    
    # Save version info
    version_info = {
        'version': version_name,
        'timestamp': timestamp,
        'note': version_note,
        'model_type': model_type,
        'features_count': len(features['feature_names']),
        'performance': performance['roc_auc'],
        'full_performance': performance  # Save all metrics
    }
    
    with open(f'{version_dir}/version_info.json', 'w') as f:
        json.dump(version_info, f, indent=2)
    
    print(f"✅ Version {version_name} saved!")
    print(f"   📁 Location: {version_dir}/")
    print(f"   🤖 Model: {model_type}")
    print(f"   📊 ROC-AUC: {performance['roc_auc']:.4f}")
    print(f"   🔢 Features: {len(features['feature_names'])}")
    print(f"   💡 Note: {version_note}")
    
    return version_name
    
def load_model_version(version_name):
    """Load a specific model version"""
    version_dir = f"model_versions/{version_name}"
    
    if not os.path.exists(version_dir):
        print(f"❌ Version {version_name} not found!")
        return None
    
    model = joblib.load(f'{version_dir}/model.joblib')
    features = joblib.load(f'{version_dir}/features.joblib')
    performance = joblib.load(f'{version_dir}/performance.joblib')
    
    print(f"✅ Loaded {version_name}")
    print(f"📊 ROC-AUC: {performance['roc_auc']:.4f}")
    
    return model, features, performance

def load_current_model():
    """Load the current/latest model"""
    return joblib.load('current_model.joblib'), joblib.load('current_features.joblib')

def list_model_versions():
    """List all saved model versions with model types"""
    if not os.path.exists('model_versions'):
        print("No versions saved yet!")
        return
    
    versions = os.listdir('model_versions')
    versions.sort()
    
    print("\n" + "="*60)
    print("📚 MODEL VERSIONS")
    print("="*60)
    
    for version in versions:
        version_path = f"model_versions/{version}"
        if os.path.exists(f"{version_path}/version_info.json"):
            with open(f"{version_path}/version_info.json", 'r') as f:
                info = json.load(f)
            
            model_icon = "🤖" if info.get('model_type') == 'xg' else "🌲"
            
            print(f"{model_icon} {version}")
            print(f"   ⏰ {info['timestamp']}")
            print(f"   📊 ROC-AUC: {info['performance']:.4f}")
            print(f"   🔢 Features: {info['features_count']}")
            print(f"   💡 {info.get('note', 'No note')}")
            print()