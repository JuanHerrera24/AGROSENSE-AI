from tests.conftest import TOKEN

# Exactly what BACKEND-AGROSENSE sends (AiClientService.predictIrrigation).
BACKEND_BODY = {"id_crop": 7, "soil_moisture": 31.5, "air_temperature": 24.0, "ph": 6.4,
                "humidity_min": 40, "humidity_max": 80, "ph_min": 5.5, "ph_max": 7.0}


def test_backend_payload_returns_the_stored_fields(client):
    response = client.post("/predict/irrigation", json=BACKEND_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["id_crop"] == 7
    assert body["recommendation"]
    assert 0 <= body["confidence"] <= 1
    assert body["model_used"].startswith("agrosense-irrigation")


def test_only_soil_moisture_is_required(client):
    response = client.post("/predict/irrigation", json={"soil_moisture": 55})
    assert response.status_code == 200
    assert response.json()["limits"]["source"] == "default"


def test_unknown_fields_are_ignored(client):
    assert client.post("/predict/irrigation", json={**BACKEND_BODY, "other": 1}).status_code == 200


def test_invalid_requests_return_400_with_message(client):
    cases = [
        {"air_temperature": 20},
        {"soil_moisture": 130},
        {"soil_moisture": "a lot"},
        {"soil_moisture": 50, "humidity_min": 80, "humidity_max": 40},
        {"soil_moisture": 50, "humidity_min": 40},
    ]
    for body in cases:
        response = client.post("/predict/irrigation", json=body)
        assert response.status_code == 400, body
        assert response.json()["message"]


def test_malformed_json_returns_400(client):
    response = client.post("/predict/irrigation", content=b"{no", headers={"Content-Type": "application/json"})
    assert response.status_code == 400


def test_large_body_returns_413(client):
    response = client.post("/detect/anomalies", content=b"1" * 70_000, headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_unknown_route_and_wrong_method(client):
    assert client.get("/missing").json() == {"message": "Not found"}
    assert client.get("/predict/irrigation").status_code == 405


def test_health_supports_get_and_head(client):
    assert client.get("/health").json() == {"status": "UP"}
    assert client.head("/health").status_code == 200
    assert client.head("/").status_code == 200


def test_token_is_required_when_configured(secured):
    assert secured.post("/predict/irrigation", json=BACKEND_BODY).status_code == 401
    wrong = {"Authorization": "Bearer other"}
    assert secured.post("/predict/irrigation", json=BACKEND_BODY, headers=wrong).status_code == 401
    right = {"Authorization": f"Bearer {TOKEN}"}
    assert secured.post("/predict/irrigation", json=BACKEND_BODY, headers=right).status_code == 200
    for path in ("/predict/moisture-forecast", "/detect/anomalies", "/predict/disease-risk"):
        assert secured.post(path, json={}).status_code == 401
    assert secured.get("/health").status_code == 200
    assert secured.get("/models").status_code == 200


def test_models_and_crops_endpoints(client):
    assert set(client.get("/models").json()) == {"irrigation", "forecast", "anomaly", "disease"}
    codes = {c["code"] for c in client.get("/crops").json()}
    assert {"generic", "coffee", "plantain", "potato"} <= codes


def test_demo_and_docs_are_served(client):
    assert "AgroSense" in client.get("/demo").text
    assert client.get("/docs").status_code == 200


def test_every_model_endpoint_answers(client):
    assert client.post("/predict/moisture-forecast", json={"soil_moisture": 60}).status_code == 200
    assert client.post("/detect/anomalies",
                       json={"sensor_type": "PH", "readings": [6.1, 6.2, 6.1, 6.2]}).status_code == 200
    assert client.post("/predict/disease-risk",
                       json={"crop_type": "coffee", "air_temperature": 22, "relative_humidity": 90}).status_code == 200
    assert client.post("/detect/anomalies", json={"sensor_type": "RADAR", "readings": [1, 2, 3]}).status_code == 400
