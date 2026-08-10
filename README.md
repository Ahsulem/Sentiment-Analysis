# Sentiment-Analysis
MLOps pipeline for Sentiment Analysis on Youtube comments

# Commands

conda create -n youtube python=3.14 -y
conda activate youtube
pip install -r requirements.txt

dvc init
dvc repro
dvc dag

aws configure