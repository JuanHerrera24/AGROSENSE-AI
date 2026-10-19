import pytest
from fastapi.testclient import TestClient

from agrosense_ai.api import create_app
from agrosense_ai.registry import ModelRegistry

TOKEN = "test-token"


@pytest.fixture(scope="session")
def models():
    return ModelRegistry()


@pytest.fixture(scope="session")
def client(models):
    with TestClient(create_app(models, token="")) as c:
        yield c


@pytest.fixture(scope="session")
def secured(models):
    with TestClient(create_app(models, token=TOKEN)) as c:
        yield c
