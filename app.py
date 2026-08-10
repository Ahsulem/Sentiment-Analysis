from flask import Flask, jsonify, request

from model import YouTubeSentimentModel


def create_app() -> Flask:
    app = Flask(__name__)
    model = YouTubeSentimentModel()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        text = payload.get("text")
        if not isinstance(text, str):
            return jsonify({"error": "Expected JSON payload with string field 'text'"}), 400

        result = model.predict(text)
        return jsonify({"text": text, "label": result.label, "score": result.score})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
