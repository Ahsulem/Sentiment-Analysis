# Sentiment-Analysis
MLOps pipeline for Sentiment Analysis on Youtube comments

## Flask API

This repository includes a minimal Flask API that calls a sentiment model for YouTube comments.

### Run locally

```bash
pip install -r requirements.txt
python app.py
```

### Endpoints

- `GET /health` - health check
- `POST /predict` - run sentiment prediction on a comment

Example request:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this video"}'
```
