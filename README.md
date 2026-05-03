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

## Submitting a pull request (assignment)

Fork [mateenyaqoob/CI-CD-for-ML-using-GitHub-Actions](https://github.com/mateenyaqoob/CI-CD-for-ML-using-GitHub-Actions) on GitHub, push a branch with your fixes to your fork, then open a pull request against the upstream `main` branch. On macOS, if `git` reports an Xcode license error, run `sudo xcodebuild -license` once to accept the license, or use [GitHub Desktop](https://desktop.github.com/) to clone your fork and push without the command-line tools conflict.

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

