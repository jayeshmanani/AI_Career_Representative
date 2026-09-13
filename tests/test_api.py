from fastapi.testclient import TestClient
from backend.main import app, compute_file_sha256, RESUME_PATH, CACHE_PATH

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert data["resume_path_exists"] is True


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Candidate AI Resume Assistant" in response.text
    assert "text/html" in response.headers.get("content-type", "")


def test_api_profile():
    response = client.get("/api/profile")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Jayesh Manani"
    assert "email" in data
    assert isinstance(data["skills"], list)
    assert len(data["skills"]) > 0
    assert "Python" in data["skills"] or any("python" in s.lower() for s in data["skills"])


def test_api_resume():
    response = client.get("/api/resume")
    assert response.status_code == 200
    assert "application/pdf" in response.headers.get("content-type", "")
    assert len(response.content) > 10000


def test_chat_validation_empty():
    response = client.post("/chat", json={"question": ""})
    # Either 422 Unprocessable Entity (Pydantic min_length=1) or 400
    assert response.status_code in (400, 422)


def test_chat_validation_whitespace():
    response = client.post("/chat", json={"question": "    "})
    assert response.status_code in (400, 422)


def test_resume_cache_sha256():
    assert RESUME_PATH.exists()
    assert CACHE_PATH.exists()
    file_sha = compute_file_sha256(RESUME_PATH)
    import json
    with open(CACHE_PATH, "r") as f:
        cache = json.load(f)
    assert cache.get("sha256") == file_sha
    assert "Jayesh" in cache.get("data", {}).get("name", "")


def test_chat_refuses_out_of_scope():
    response = client.post(
        "/chat",
        json={"question": "can you give me python script says hello", "history": []},
    )
    assert response.status_code == 200
    # Must refuse to write the script and mention domain restriction
    answer = response.text.lower()
    assert "only answer questions directly related" in answer or "cannot" in answer or "exclusive" in answer
    assert "def main()" not in response.text


def test_chat_transferable_skills_for_unlisted_tech():
    response = client.post(
        "/chat",
        json={"question": "does he know JS?", "history": []},
    )
    assert response.status_code == 200
    answer = response.text.lower()
    # Must mention that JS is not listed, but highlight foundation in Python / C or transferable learning
    assert "not" in answer or "explicitly" in answer
    assert "python" in answer or "c" in answer or "transferable" in answer or "adapt" in answer or "learn" in answer


