# ML CI/CD with GitHub Actions

This repository demonstrates:

✅ Data preprocessing  
✅ Model training  
✅ Model evaluation  
✅ Automated testing  
✅ Docker image build  
✅ CI/CD pipeline using GitHub Actions
✅ Model versioning
✅ MLflow experiment tracking

## Run Locally

pip install -r requirements.txt

python src/preprocess.py
python src/train.py
python src/evaluate.py

python app.py

## Model Versioning

Training always saves the latest model to models/model.pkl and a versioned
artifact to models/model_<version>.pkl. Set MODEL_VERSION to control the
version tag; CI uses the workflow run number and writes the current version
to models/model_version.txt.

## MLflow Tracking

If mlflow is installed, training logs parameters, metrics, and the model to a
local MLflow store (mlruns/). You can override these values:

- MLFLOW_TRACKING_URI (default: file:./mlruns)
- MLFLOW_EXPERIMENT_NAME (default: ml-cicd)
- MLFLOW_REGISTERED_MODEL_NAME (optional)

## GitHub Actions

Push code to main branch → pipeline auto runs.

## Structure
ml-cicd-github-actions/

│── .github/

│   └── workflows/

│       └── ml-pipeline.yml

│

│── data/

│   └── sample.csv

│

│── src/

│   ├── preprocess.py

│   ├── train.py

│   ├── evaluate.py

│   └── predict.py

│

│── models/

│   └── (saved model here after training)

│

│── requirements.txt

│── Dockerfile

│── README.md

│── app.py

│── test_model.py

│── .gitignore

