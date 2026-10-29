from fastapi.testclient import TestClient
from fastapi import FastAPI

import api.main as api_module
from api.main import app
from src.api_security import LocalAdminKeyMiddleware, _is_loopback


client = TestClient(app, headers={"X-API-Key": "test-only-api-key"})


def test_admin_key_required_for_api_routes_but_not_health():
    health = client.get("/", headers={})
    forbidden = client.get("/models", headers={"X-API-Key": "wrong"})
    authorized = client.get("/models")

    assert health.status_code == 200
    assert forbidden.status_code == 401
    assert authorized.status_code == 200


def test_unconfigured_key_policy_only_allows_loopback_hosts():
    assert _is_loopback("127.0.0.1")
    assert _is_loopback("::1")
    assert not _is_loopback("192.0.2.1")
    assert not _is_loopback("testclient")


def test_non_loopback_access_is_rejected_when_key_is_not_configured():
    test_app = FastAPI()
    test_app.add_middleware(
        LocalAdminKeyMiddleware,
        key_provider=lambda: None,
    )
    test_app.get("/protected")(lambda: {"ok": True})

    response = TestClient(test_app).get("/protected")

    assert response.status_code == 403


def test_request_body_size_guard(monkeypatch):
    monkeypatch.setattr(api_module, "MAX_REQUEST_BODY_BYTES", 8)

    response = client.post(
        "/predict/csv",
        files={"file": ("input.csv", b"feature\n1\n", "text/csv")},
        data={"model_id": "unused"},
    )

    assert response.status_code == 413
    assert "22 MB" in response.json()["detail"]
