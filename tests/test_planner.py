"""Correctness tests for the planner.

Runs directly (``python tests/test_planner.py``) or under pytest.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import ValidationError

from app import planner
from app.planner import PlannerError, plan, prepare
from app.schemas import PlanRequest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextmanager
def _time_budgets(**overrides):
    # Force CP-SAT time limits per stage; a 0s limit makes every solve return
    # UNKNOWN, which deterministically reproduces real time-out evidence.
    defaults = {"stage1": 10.0, "stage": 0.3, "op_stage": 0.08, "probe": 5.0}
    defaults.update(overrides)
    saved = planner._time_limits
    planner._time_limits = lambda: dict(defaults)
    try:
        yield
    finally:
        planner._time_limits = saved


def _load_example() -> dict:
    with open(os.path.join(ROOT, "examples", "request.json")) as fh:
        return json.load(fh)


def _plan(payload: dict):
    return plan(PlanRequest.model_validate(payload))


def test_example_is_optimal_and_matches_documented_response():
    r = _plan(_load_example())
    assert r.feasible and r.optimal
    assert r.capacity_upper_bound == 96

    by_id = {a.tensor_id: a.address for a in r.addresses}
    assert by_id == {"w": 64, "x": 0, "y": 0, "yv": 16}
    # views are placed inside their root
    assert by_id["yv"] == by_id["y"] + 16

    p1, p2 = r.timeline
    assert (p1.start, p1.end, p1.peak_bytes) == (0, 5, 64)
    assert (p2.start, p2.end, p2.peak_bytes) == (0, 7, 96)
    # mutually exclusive paths are planned independently: each meets the
    # deadline on its own and their timelines may overlap
    assert p2.start < p1.end


def test_mutually_exclusive_paths_overlap_and_reuse_addresses():
    # Each path finishes in 7 < deadline 8, but serialized they would need
    # 14: the planner must not serialize mutually-exclusive paths.
    r = _plan({
        "deadline": 8, "memory_limit": 256,
        "paths": [
            {"id": "A",
             "operators": [{"id": "a", "duration": 4},
                           {"id": "b", "duration": 3, "depends_on": ["a"]}],
             "tensors": [{"id": "u", "bytes": 64, "alignment": 16,
                          "producer": "a", "consumers": ["b"]}]},
            {"id": "B",
             "operators": [{"id": "a", "duration": 4},
                           {"id": "b", "duration": 3, "depends_on": ["a"]}],
             "tensors": [{"id": "v", "bytes": 64, "alignment": 16,
                          "producer": "a", "consumers": ["b"]}]},
        ],
    })
    assert r.feasible and r.optimal
    pa, pb = r.timeline
    assert (pa.start, pa.end) == (0, 7) and (pb.start, pb.end) == (0, 7)
    # overlapping timelines still share the global buffer: u and v are never
    # live in the same path, so they reuse the same address
    addr = {a.tensor_id: a.address for a in r.addresses}
    assert addr["u"] == addr["v"] == 0
    assert r.capacity_upper_bound == 64


def test_overlapping_paths_still_protect_same_path_lifetimes():
    # Timelines overlap, but tensors simultaneously live in the SAME path
    # must still get disjoint bytes.
    r = _plan({
        "deadline": 6, "memory_limit": 256,
        "paths": [
            {"id": "A",
             "operators": [{"id": "a", "duration": 2},
                           {"id": "b", "duration": 2, "depends_on": ["a"]}],
             "tensors": [
                 {"id": "u", "bytes": 64, "alignment": 16, "producer": "a",
                  "consumers": ["b"]},
                 {"id": "w", "bytes": 64, "alignment": 16, "producer": "a",
                  "consumers": ["b"]},
             ]},
            {"id": "B",
             "operators": [{"id": "a", "duration": 3}],
             "tensors": [{"id": "v", "bytes": 128, "alignment": 16,
                          "producer": "a", "consumers": []}]},
        ],
    })
    assert r.feasible and r.optimal
    addr = {a.tensor_id: a.address for a in r.addresses}
    # u and w are co-live in A -> disjoint; v (path B) may reuse their bytes
    assert {addr["u"], addr["w"]} == {0, 64}
    assert addr["v"] == 0
    assert r.capacity_upper_bound == 128


def test_deadline_infeasible():
    p = _load_example(); p["deadline"] = 6  # P2 alone needs 7
    r = _plan(PlanRequest.model_validate(p))
    assert not r.feasible and r.reason == "deadline"


def test_memory_infeasible():
    p = _load_example(); p["memory_limit"] = 95  # minimum capacity is 96
    r = _plan(PlanRequest.model_validate(p))
    assert not r.feasible and r.reason == "memory_limit"


def test_both_infeasible():
    p = _load_example(); p["deadline"] = 6; p["memory_limit"] = 64
    r = _plan(PlanRequest.model_validate(p))
    assert not r.feasible and r.reason == "deadline_and_memory_limit"


def test_parallel_fork_join_coexists():
    r = _plan({
        "deadline": 4, "memory_limit": 256,
        "paths": [{
            "id": "P",
            "operators": [
                {"id": "p", "duration": 1},
                {"id": "q", "duration": 2, "depends_on": ["p"]},
                {"id": "r", "duration": 2, "depends_on": ["p"]},
                {"id": "j", "duration": 1, "depends_on": ["q", "r"]},
            ],
            "tensors": [
                {"id": "x", "bytes": 64, "alignment": 16, "producer": "p", "consumers": ["q", "j"]},
                {"id": "y", "bytes": 48, "alignment": 16, "producer": "q", "consumers": ["j"]},
                {"id": "z", "bytes": 32, "alignment": 16, "producer": "r", "consumers": ["j"]},
            ],
        }],
    })
    assert r.feasible and r.optimal
    # all three live together at join -> 64+48+32
    assert r.capacity_upper_bound == 144
    assert r.timeline[0].peak_bytes == 144
    # q and r may run in parallel
    assert r.timeline[0].end == 4


def test_half_open_intervals_reuse_address():
    r = _plan({
        "deadline": 5, "memory_limit": 256,
        "paths": [{
            "id": "P",
            "operators": [
                {"id": "a", "duration": 2},
                {"id": "b", "duration": 2, "depends_on": ["a"]},
            ],
            "tensors": [
                {"id": "t1", "bytes": 40, "alignment": 8, "producer": "a", "consumers": ["b"]},
                {"id": "t2", "bytes": 40, "alignment": 8, "producer": "b", "consumers": []},
            ],
        }],
    })
    addr = {a.tensor_id: a.address for a in r.addresses}
    # t1 dies exactly when b starts; t2 is produced then: reuse is allowed
    assert addr["t1"] == addr["t2"] == 0
    assert r.capacity_upper_bound == 40


def test_view_extends_root_lifetime():
    r = _plan({
        "deadline": 6, "memory_limit": 256,
        "paths": [{
            "id": "P",
            "operators": [
                {"id": "a", "duration": 1},
                {"id": "b", "duration": 1, "depends_on": ["a"]},
                {"id": "c", "duration": 1, "depends_on": ["b"]},
            ],
            "tensors": [
                {"id": "r", "bytes": 64, "alignment": 16, "producer": "a", "consumers": ["b"]},
                {"id": "v", "bytes": 8, "alignment": 8, "producer": "c", "output": True,
                 "base_tensor": "r", "view_offset": 0},
                {"id": "s", "bytes": 64, "alignment": 16, "producer": "b", "consumers": ["c"]},
            ],
        }],
    })
    addr = {a.tensor_id: a.address for a in r.addresses}
    # r is kept alive until path end by output view v, so s cannot alias it
    assert addr["r"] != addr["s"]
    assert r.capacity_upper_bound == 128
    assert addr["v"] == addr["r"]


def test_cross_path_reuse():
    r = _plan({
        "deadline": 8, "memory_limit": 64,
        "paths": [
            {"id": "A",
             "operators": [{"id": "a", "duration": 2},
                           {"id": "b", "duration": 2, "depends_on": ["a"]}],
             "tensors": [{"id": "u", "bytes": 64, "alignment": 16,
                          "producer": "a", "consumers": ["b"]}]},
            {"id": "B",
             "operators": [{"id": "a", "duration": 2},
                           {"id": "b", "duration": 2, "depends_on": ["a"]}],
             "tensors": [{"id": "v", "bytes": 64, "alignment": 16,
                          "producer": "a", "consumers": ["b"]}]},
        ],
    })
    assert r.feasible and r.optimal
    addr = {a.tensor_id: a.address for a in r.addresses}
    assert addr["u"] == addr["v"] == 0
    assert r.capacity_upper_bound == 64
    # both paths start at 0: no artificial serialization
    assert all(p.start == 0 for p in r.timeline)


def test_inconsistent_same_name_tensor_rejected():
    p = _load_example()
    p["paths"][1]["tensors"][0]["alignment"] = 16
    try:
        plan(PlanRequest.model_validate(p))
    except PlannerError as e:
        assert "inconsistent" in str(e)
    else:
        raise AssertionError("expected PlannerError")


def test_semantic_errors_are_422():
    cases = [
        # dependency cycle
        {"paths": [{"id": "P",
                    "operators": [{"id": "a", "duration": 1, "depends_on": ["b"]},
                                  {"id": "b", "duration": 1, "depends_on": ["a"]}],
                    "tensors": [{"id": "x", "bytes": 8, "alignment": 8,
                                 "producer": "a", "output": True}]}]},
        # unknown consumer
        {"paths": [{"id": "P",
                    "operators": [{"id": "a", "duration": 1}],
                    "tensors": [{"id": "x", "bytes": 8, "alignment": 8,
                                 "producer": "a", "consumers": ["zzz"]}]}]},
        # view outside root
        {"paths": [{"id": "P",
                    "operators": [{"id": "a", "duration": 1},
                                  {"id": "b", "duration": 1, "depends_on": ["a"]}],
                    "tensors": [
                        {"id": "r", "bytes": 8, "alignment": 8, "producer": "a",
                         "consumers": ["b"]},
                        {"id": "v", "bytes": 8, "alignment": 8, "producer": "b",
                         "output": True, "base_tensor": "r", "view_offset": 64}]}]},
        # non power-of-two alignment / non-positive duration are schema errors
        None,
    ]
    for extra in cases[:-1]:
        payload = {"deadline": 10, "memory_limit": 1024, **extra}
        try:
            plan(PlanRequest.model_validate(payload))
        except PlannerError:
            pass
        else:
            raise AssertionError("expected PlannerError: " + json.dumps(payload))

    for bad in (
        {"deadline": 0, "memory_limit": 1024, "paths": []},
        {"deadline": 10, "memory_limit": -1, "paths": []},
        {"paths": [{"id": "P",
                    "operators": [{"id": "a", "duration": 0}],
                    "tensors": [{"id": "x", "bytes": 8, "alignment": 3,
                                 "producer": "a"}]}]},
    ):
        try:
            PlanRequest.model_validate(bad)
        except ValidationError:
            pass
        else:
            raise AssertionError("expected ValidationError: " + json.dumps(bad))


def test_never_overclaims_optimality_on_hard_instance():
    # a larger instance must still return quickly with a feasible plan, and
    # must only claim optimality when CP-SAT proved it
    payload = json.loads(
        open(os.path.join(ROOT, "examples", "stress_payload.json")).read()
    )
    t0 = time.time()
    r = _plan(payload)
    assert time.time() - t0 < 30
    assert r.feasible
    # invariants: views inside roots, everything aligned, capacity respected
    prep = prepare(PlanRequest.model_validate(payload))
    amap = {a.tensor_id: a.address for a in r.addresses}
    for tid, info in prep.tensors.items():
        assert amap[tid] == amap[info.root] + info.cum_offset
        assert amap[tid] % info.alignment == 0
        assert amap[tid] >= 0
    assert r.capacity_upper_bound <= payload["memory_limit"]


def test_view_can_share_base_producer_operator():
    # a view may be produced inside the same operator that produces its base
    r = _plan({
        "deadline": 5, "memory_limit": 1024,
        "paths": [{
            "id": "P",
            "operators": [
                {"id": "p", "duration": 1},
                {"id": "q", "duration": 1, "depends_on": ["p"]},
            ],
            "tensors": [
                {"id": "z", "bytes": 64, "alignment": 16, "producer": "p",
                 "consumers": ["q"]},
                {"id": "a", "bytes": 16, "alignment": 16, "producer": "p",
                 "consumers": ["q"], "base_tensor": "z", "view_offset": 0},
            ],
        }],
    })
    assert r.feasible
    addr = {t.tensor_id: t.address for t in r.addresses}
    assert addr["a"] == addr["z"]


def test_address_tiebreak_uses_smallest_tensor_id_of_root():
    # root 'z' has a view 'a' whose tensor id sorts first, so root z must be
    # placed before root b even though b < z by id
    r = _plan({
        "deadline": 10, "memory_limit": 1024,
        "paths": [{
            "id": "P",
            "operators": [
                {"id": "p", "duration": 1},
                {"id": "q", "duration": 1, "depends_on": ["p"]},
            ],
            "tensors": [
                {"id": "z", "bytes": 64, "alignment": 16, "producer": "p",
                 "consumers": ["q"]},
                {"id": "a", "bytes": 16, "alignment": 16, "producer": "p",
                 "consumers": ["q"], "base_tensor": "z", "view_offset": 0},
                {"id": "b", "bytes": 64, "alignment": 16, "producer": "p",
                 "consumers": ["q"]},
            ],
        }],
    })
    assert r.optimal
    addr = {t.tensor_id: t.address for t in r.addresses}
    assert addr["z"] == 0 and addr["a"] == 0 and addr["b"] == 64


# --------------------------------------------------------------------------- #
# Three-valued evidence: FEASIBLE / INFEASIBLE / UNKNOWN are never conflated.
# --------------------------------------------------------------------------- #
def _assert_legal_feasible_plan(payload: dict, r) -> None:
    assert r.status == "feasible"
    assert r.feasible is True and r.reason is None
    assert r.timeline and r.addresses
    prep = prepare(PlanRequest.model_validate(payload))
    amap = {a.tensor_id: a.address for a in r.addresses}
    # views stay inside their root, every address obeys its own alignment
    for tid, info in prep.tensors.items():
        assert amap[tid] == amap[info.root] + info.cum_offset
        assert amap[tid] % info.alignment == 0
        assert 0 <= amap[tid]
    assert 0 < r.capacity_upper_bound <= payload["memory_limit"]
    # each path window obeys the deadline and operator windows are consistent
    for i, ps in enumerate(r.timeline):
        assert ps.start <= ps.end <= payload["deadline"]
        dur = {oid: d for oid, (d, _) in prep.path_ops[ps.path_id].items()}
        for op in ps.operators:
            assert op.end - op.start == dur[op.operator_id]
        assert ps.start == min(o.start for o in ps.operators)
        assert ps.end == max(o.end for o in ps.operators)
        assert ps.peak_bytes <= r.capacity_upper_bound


def test_feasible_response_shape():
    r = _plan(_load_example())
    assert r.status == "feasible" and r.feasible is True and r.reason is None


def test_infeasible_response_shape_and_reasons():
    p = _load_example()
    for mods, reason in (
        ({"deadline": 6}, "deadline"),
        ({"memory_limit": 95}, "memory_limit"),
        ({"deadline": 6, "memory_limit": 64}, "deadline_and_memory_limit"),
    ):
        q = json.loads(json.dumps(p)); q.update(mods)
        r = _plan(q)
        assert r.status == "infeasible"
        assert r.feasible is False and r.reason == reason
        assert not r.timeline and not r.addresses
        assert r.capacity_upper_bound == 0 and not r.optimal


def test_timeout_on_first_solve_is_reported_unknown_not_infeasible():
    # Every solve (first + both probes) times out -> nothing is proved. The
    # answer must say undecided, not fabricate infeasibility or a reason.
    p = _load_example()
    with _time_budgets(stage1=0.0, probe=0.0):
        r = _plan(p)
    assert r.status == "unknown"
    assert r.feasible is None and r.reason is None
    assert not r.optimal and r.capacity_upper_bound == 0
    assert r.timeline == [] and r.addresses == []


def test_unknown_first_solve_still_unknown_on_truly_infeasible_input():
    # The request is genuinely deadline-infeasible, but if no solve proves
    # anything in time, that impossibility must not be asserted.
    p = _load_example(); p["deadline"] = 6
    with _time_budgets(stage1=0.0, probe=0.0):
        r = _plan(p)
    assert r.status == "unknown"
    assert r.feasible is None and r.reason is None


def test_first_unknown_is_rescued_by_feasible_probe_with_real_plan():
    # First solve times out, but the relaxed probes find a witness that also
    # satisfies both original bounds. The answer is a concrete legal plan;
    # optimality cannot be claimed after an unresolved first stage.
    p = _load_example()
    with _time_budgets(stage1=0.0):
        r = _plan(p)
    _assert_legal_feasible_plan(p, r)
    assert r.optimal is False


def test_proved_infeasible_first_solve_with_timed_out_probes_has_no_reason():
    # The first solve proves infeasibility, but both diagnostic probes time
    # out. Infeasibility is established; the exact resource cause is not, so
    # reason must stay null rather than guess deadline/memory_limit.
    p = _load_example(); p["deadline"] = 6
    with _time_budgets(probe=0.0):
        r = _plan(p)
    assert r.status == "infeasible"
    assert r.feasible is False and r.reason is None


def test_one_probe_unknown_never_invents_resource_reason():
    # First solve proves infeasible; force exactly one of the two probes to
    # report UNKNOWN. No precise reason may be claimed regardless of what the
    # other probe proved.
    p = _load_example(); p["deadline"] = 6
    call = {"n": 0}
    real_solve = planner.cp_model.CpSolver.Solve

    def scripted_solve(self, model):
        call["n"] += 1
        # call 1 = first solve (proves INFEASIBLE quickly with full budget);
        # calls 2,3 = deadline/memory probes: probe 2 forced UNKNOWN.
        if call["n"] == 2:
            return planner.cp_model.UNKNOWN
        return real_solve(self, model)

    planner.cp_model.CpSolver.Solve = scripted_solve
    try:
        r = _plan(p)
    finally:
        planner.cp_model.CpSolver.Solve = real_solve
    assert r.status == "infeasible"
    assert r.feasible is False and r.reason is None


def test_deadline_probe_proves_obstruction_when_other_probe_unknown():
    # First solve UNKNOWN; deadline probe proves INFEASIBLE even with abundant
    # memory (obstruction survives extra memory) while the memory probe times
    # out: the request is proved infeasible, but the exact reason is not set.
    p = _load_example(); p["deadline"] = 6
    call = {"n": 0}
    real_solve = planner.cp_model.CpSolver.Solve

    def scripted_solve(self, model):
        call["n"] += 1
        if call["n"] == 1 or call["n"] == 3:
            return planner.cp_model.UNKNOWN
        return real_solve(self, model)  # call 2: deadline probe, proves it

    planner.cp_model.CpSolver.Solve = scripted_solve
    try:
        r = _plan(p)
    finally:
        planner.cp_model.CpSolver.Solve = real_solve
    assert r.status == "infeasible"
    assert r.feasible is False and r.reason is None


def test_feasible_incumbent_survives_later_stage_timeout_and_is_never_optimal():
    # Stage 1 finds the optimum, but force a later lexicographic stage to time
    # out: the pinned-prefix incumbent must still be returned as a legal plan,
    # and optimality must not be claimed.
    p = _load_example()
    call = {"n": 0}
    real_solve = planner.cp_model.CpSolver.Solve

    def scripted_solve(self, model):
        call["n"] += 1
        if call["n"] == 2:  # first address stage
            return planner.cp_model.UNKNOWN
        return real_solve(self, model)

    planner.cp_model.CpSolver.Solve = scripted_solve
    try:
        r = _plan(p)
    finally:
        planner.cp_model.CpSolver.Solve = real_solve
    _assert_legal_feasible_plan(p, r)
    assert r.optimal is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
