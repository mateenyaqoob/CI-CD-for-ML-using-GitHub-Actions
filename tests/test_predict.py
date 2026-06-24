import subprocess
import sys


def setup_module():
    # Ensure processed data and model exist
    subprocess.run([sys.executable, "src/preprocess.py"], check=True)
    subprocess.run([sys.executable, "src/train.py"], check=True)


def test_predict_script_outputs_prediction():
    res = subprocess.run([sys.executable, "src/predict.py"], capture_output=True, text=True, check=True)
    assert "Prediction:" in res.stdout
