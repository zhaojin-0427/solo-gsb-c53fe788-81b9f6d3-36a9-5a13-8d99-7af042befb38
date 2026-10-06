"""HTTP-level tests for POST /plans using FastAPI's in-process client."""
from __future__ import annotations

import json
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app import planner
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


def test_undetermined_returns_503_not_infeasible_200():
    # Script every CP-SAT call as timed-out (UNKNOWN) with no incumbent: the
    # first solve is not decided, and neither probe finds a proof. The API
    # must say "undecided" with 503 instead of fabricating a feasible=false
    # reason with 200.
    def fake(model, tag, time_limit):
        solver = planner._new_solver(1e-12)
        solver.Solve(model)  # starved -> UNKNOWN, no incumbent
        return planner.UNKNOWN, solver, planner.cp_model.UNKNOWN

    with mock.patch.object(planner, "run_solver", fake):
        r = client.post("/plans", json=EXAMPLE)
    assert r.status_code == 503
    body = r.json()
    assert body == {
        "feasible": False,
        "optimal": False,
        "reason": "undetermined",
        "capacity_upper_bound": 0,
        "timeline": [],
        "addresses": [],
    }


def test_unknown_first_solve_with_proven_probe_is_still_200():
    # First solve timed out, but the deadline-relaxed probe proves the memory
    # limit impossible: a definite infeasibility stays HTTP 200 with the exact
    # proven reason (an undecided first solve must not hide a real proof).
    labels = iter([
        planner.UNKNOWN,          # first solve
        planner.FEASIBLE,         # memory-relaxed probe: feasible
        planner.INFEASIBLE,       # deadline-relaxed probe: impossible
    ])

    def fake(model, tag, time_limit):
        label = next(labels)
        if label == planner.FEASIBLE:
            solver = planner._new_solver(5.0)
            status = solver.Solve(model)
            return label, solver, status
        if label == planner.INFEASIBLE:
            probe = planner.cp_model.CpModel()
            v = probe.NewBoolVar("x")
            probe.Add(v == 0); probe.Add(v == 1)
            solver = planner._new_solver(5.0)
            solver.Solve(probe)
            return label, solver, planner.cp_model.INFEASIBLE
        solver = planner._new_solver(1e-12)
        solver.Solve(model)
        return label, solver, planner.cp_model.UNKNOWN

    with mock.patch.object(planner, "run_solver", fake):
        r = client.post("/plans", json=EXAMPLE)
    assert r.status_code == 200
    assert r.json()["feasible"] is False
    assert r.json()["reason"] == "memory_limit"


def test_real_solver_timeout_header_yields_503():
    # End-to-end without mocks: an effectively zero time budget on the large
    # instance leaves CP-SAT undecided. The instance is actually feasible
    # under the default budgets (see test_planner stress test), so this is a
    # genuine timeout rather than an infeasibility.
    with open(os.path.join(ROOT, "examples", "stress_payload.json")) as fh:
        hard = json.load(fh)
    r = client.post(
        "/plans", json=hard, headers={"X-Solver-Timeout-Seconds": "0.00001"}
    )
    assert r.status_code == 503
    assert r.json()["reason"] == "undetermined"


def test_bad_timeout_header_is_422():
    for bad in ("abc", "-1", "0", "nan", "inf"):
        r = client.post(
            "/plans", json=EXAMPLE, headers={"X-Solver-Timeout-Seconds": bad}
        )
        assert r.status_code == 422, bad


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
