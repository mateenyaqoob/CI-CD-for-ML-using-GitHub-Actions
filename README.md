# ML CI/CD with GitHub Actions

This repository demonstrates:

✅ Data preprocessing  
✅ Model training  
✅ Model evaluation  
✅ Automated testing  
✅ Docker image build  
✅ CI/CD pipeline using GitHub Actions

## Run Locally

pip install -r requirements.txt

python src/preprocess.py
python src/train.py
python src/evaluate.py

python app.py

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
# Work in progress
# Work in progress
