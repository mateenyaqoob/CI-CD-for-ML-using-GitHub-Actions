import subprocess
import sys


def setup_module():
    # Ensure processed data and model exist
    subprocess.run([sys.executable, "src/preprocess.py"], check=True)
    subprocess.run([sys.executable, "src/train.py"], check=True)


def test_api_predict_endpoint():
    import app

    client = app.app.test_client()
    res = client.post("/predict", json={"features": [5, 6]})
    assert res.status_code == 200
    data = res.get_json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)
