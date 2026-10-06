"""HTTP-level tests for POST /plans using FastAPI's in-process client."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, "examples", "request.json")) as fh:
    EXAMPLE = json.load(fh)
with open(os.path.join(ROOT, "examples", "response.json")) as fh:
    DOC_RESPONSE = json.load(fh)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_plan_matches_documented_example():
    r = client.post("/plans", json=EXAMPLE)
    assert r.status_code == 200
    assert r.json() == DOC_RESPONSE


def test_infeasible_deadline():
    p = json.loads(json.dumps(EXAMPLE)); p["deadline"] = 6  # P2 alone needs 7
    r = client.post("/plans", json=p)
    assert r.status_code == 200
    assert r.json()["feasible"] is False and r.json()["reason"] == "deadline"


def test_semantic_422():
    p = json.loads(json.dumps(EXAMPLE))
    p["paths"][1]["tensors"][0]["alignment"] = 16  # inconsistent with P1
    r = client.post("/plans", json=p)
    assert r.status_code == 422
    assert "inconsistent" in r.json()["detail"][0]["msg"]


def test_schema_422():
    p = json.loads(json.dumps(EXAMPLE)); p["deadline"] = -1
    r = client.post("/plans", json=p)
    assert r.status_code == 422


def test_malformed_json_422():
    r = client.post("/plans", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_unknown_field_422():
    p = json.loads(json.dumps(EXAMPLE)); p["bogus"] = 1
    r = client.post("/plans", json=p)
    assert r.status_code == 422


def test_infeasible_answers_stay_http_200():
    # A proved-infeasible request is a normal answer with the verdict in the
    # body — not a 4xx/5xx.
    p = json.loads(json.dumps(EXAMPLE)); p["deadline"] = 6
    r = client.post("/plans", json=p)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    assert body["feasible"] is False and body["reason"] == "deadline"


def test_undecided_answer_is_http_503():
    # Every CP-SAT solve forced to a zero time budget returns UNKNOWN. The
    # verdict must be explicitly undecided (HTTP 503 + status "unknown"),
    # never a 200 dressed up as proved infeasible.
    keys = (
        "PLANNER_STAGE1_SECONDS", "PLANNER_STAGE_SECONDS",
        "PLANNER_OP_STAGE_SECONDS", "PLANNER_PROBE_SECONDS",
    )
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "0"
    try:
        r = client.post("/plans", json=EXAMPLE)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert r.status_code == 503
    body = r.json()
    assert body == {
        "status": "unknown",
        "feasible": None,
        "optimal": False,
        "reason": None,
        "capacity_upper_bound": 0,
        "timeline": [],
        "addresses": [],
    }


def test_feasible_body_has_status_field():
    r = client.post("/plans", json=EXAMPLE)
    assert r.status_code == 200
    assert r.json()["status"] == "feasible"
    assert r.json()["feasible"] is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
