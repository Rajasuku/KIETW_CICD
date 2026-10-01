# test_app.py
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_endpoint(client):
    response = client.get("/")  # Test the home route
    assert response.status_code == 200
    assert b"Iris Classifier API is Running!" in response.data

def test_predict_endpoint_valid_input(client):
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}  # Valid input for setosa
    )
    assert response.status_code == 200
    assert response.json == {"prediction": "setosa"}

def test_predict_endpoint_invalid_input(client):
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4]}  # Invalid: missing one feature
    )
    assert response.status_code == 400
    assert "error" in response.json
