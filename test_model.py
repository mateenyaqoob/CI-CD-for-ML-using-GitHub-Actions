import joblib
import numpy as np
import os
import glob

# Setup absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def test_prediction():
    # Find all model files
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    assert len(model_files) > 0, "No trained models found in the models directory."
    
    # Automatically grab the latest model file
    latest_model_path = max(model_files, key=os.path.getctime)
    model = joblib.load(latest_model_path)
    
    # Test the prediction
    pred = model.predict(np.array([[5, 6]]))
    
    # Assert that the output is one of the expected classes
    assert pred[0] in [0, 1], f"Unexpected prediction result: {pred[0]}"