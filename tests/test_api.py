import unittest

from app import create_app


class APITestCase(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.testing = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_predict_positive(self):
        response = self.client.post("/predict", json={"text": "I love this awesome video"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["label"], "positive")
        self.assertGreater(payload["score"], 0)

    def test_predict_validation(self):
        response = self.client.post("/predict", json={"comment": "missing text field"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
