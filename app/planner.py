"""Buffer / schedule planning with OR-Tools CP-SAT.

The planner jointly solves:

* a non-preemptive schedule of each path's operator DAG (ready operators may
  run in parallel), where every path is planned independently on the shared
  timeline and must individually finish before the deadline — mutually
  exclusive paths may overlap in time because only one of them runs;
* placement of every root tensor in one shared byte-addressable buffer under
  power-of-two alignment, where tensors that are never simultaneously live in
  a path may share bytes and mutually-exclusive paths may reuse them even
  when their timelines overlap.

Optimization is strictly lexicographic:

1. minimize the address-space upper bound (capacity);
2. minimize tensor addresses in tensor-id order;
3. minimize path starts in path-id order;
4. minimize operator starts in operator-id order (operators tied across paths
   are ordered by path id).

Only when CP-SAT proves every lexicographic stage optimal is the result marked
``optimal``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ortools.sat.python import cp_model

from .schemas import (
    OpSchedule,
    PathSchedule,
    PlanRequest,
    PlanResponse,
    TensorAddress,
)

# Outcome labels shared by every CP-SAT invocation: a solve either finds a
# feasible solution (FEASIBLE/OPTIMAL), proves impossibility (INFEASIBLE), or
# cannot decide within its time budget (UNKNOWN and every other status).
FEASIBLE = "feasible"
INFEASIBLE = "infeasible"
UNKNOWN = "unknown"

# Reason returned when feasibility itself could not be decided; the HTTP layer
# maps it to 503 rather than presenting it as a proven result.
UNDETERMINED = "undetermined"


class PlannerError(ValueError):
    """Semantic validation error; mapped to HTTP 422."""


@dataclass
class Occurrence:
    producer: str
    consumers: list[str]
    output: bool


@dataclass
class TInfo:
    id: str
    size: int
    alignment: int
    base: Optional[str]
    offset: int
    root: str = ""
    cum_offset: int = 0
    occ: dict[str, Occurrence] = field(default_factory=dict)


@dataclass
class Prepared:
    path_ids: list[str]
    path_ops: dict[str, dict]       # path -> {op_id: (duration, deps)}
    path_op_order: dict[str, list[str]]
    tensors: dict[str, TInfo]
    roots: list[str]
    root_size: dict[str, int]
    root_align: dict[str, int]
    live_roots: set[str]            # roots with a non-empty lifetime somewhere
    H: int
    S: int


@dataclass
class ModelArtifacts:
    model: cp_model.CpModel
    s: dict          # (pid, oid) -> start var (global timeline)
    f: dict          # (pid, oid) -> finish var
    sp: dict         # pid -> path start var
    ep: dict         # pid -> path end var
    addr: dict       # root id -> address var
    cap: object
    lstart: dict     # (pid, root) -> liveness start expr
    lend: dict       # (pid, root) -> liveness end expr
    presence: dict   # (pid, root) -> bool var, True iff non-empty lifetime


# --------------------------------------------------------------------------- #
# Validation / preparation
# --------------------------------------------------------------------------- #
def prepare(req: PlanRequest) -> Prepared:
    path_ids: list[str] = []
    path_ops: dict[str, dict] = {}
    path_op_order: dict[str, list[str]] = {}
    canon: dict[str, tuple] = {}
    tensors: dict[str, TInfo] = {}

    if len({p.id for p in req.paths}) != len(req.paths):
        raise PlannerError("duplicate path ids are not allowed")

    for path in req.paths:
        pid = path.id
        path_ids.append(pid)

        if len({o.id for o in path.operators}) != len(path.operators):
            raise PlannerError(f"path {pid}: duplicate operator ids")
        ops = {o.id: (o.duration, list(o.depends_on)) for o in path.operators}
        path_ops[pid] = ops
        path_op_order[pid] = [o.id for o in path.operators]

        local = {t.id: t for t in path.tensors}
        if len(local) != len(path.tensors):
            raise PlannerError(f"path {pid}: duplicate tensor ids")

        for o in path.operators:
            for d in o.depends_on:
                if d == o.id:
                    raise PlannerError(
                        f"path {pid}: operator {o.id!r} cannot depend on itself"
                    )
                if d not in ops:
                    raise PlannerError(
                        f"path {pid}: operator {o.id!r} depends on unknown operator {d!r}"
                    )

        for t in path.tensors:
            if t.producer not in ops:
                raise PlannerError(
                    f"path {pid}: tensor {t.id!r} producer {t.producer!r} is not an operator"
                )
            for c in t.consumers:
                if c not in ops:
                    raise PlannerError(
                        f"path {pid}: tensor {t.id!r} consumer {c!r} is not an operator"
                    )
            if t.base_tensor is not None:
                if t.base_tensor == t.id:
                    raise PlannerError(f"path {pid}: tensor {t.id!r} cannot view itself")
                if t.base_tensor not in local:
                    raise PlannerError(
                        f"path {pid}: tensor {t.id!r} base_tensor {t.base_tensor!r} not found"
                    )

            key = (t.bytes, t.alignment, t.base_tensor, t.view_offset)
            if t.id in canon and canon[t.id] != key:
                raise PlannerError(
                    f"tensor {t.id!r} has inconsistent definitions across paths"
                )
            canon[t.id] = key

            info = tensors.get(t.id)
            if info is None:
                info = TInfo(
                    id=t.id, size=t.bytes, alignment=t.alignment,
                    base=t.base_tensor, offset=t.view_offset,
                )
                tensors[t.id] = info
            info.occ[pid] = Occurrence(t.producer, list(t.consumers), t.output)

    # resolve view chains
    for tid, info in tensors.items():
        chain_ids: list[str] = []
        cur = info
        cum = 0
        while cur.base is not None:
            if cur.id in chain_ids:
                raise PlannerError(f"tensor {tid!r}: cyclic view chain via {cur.id!r}")
            chain_ids.append(cur.id)
            nxt = tensors.get(cur.base)
            if nxt is None:  # pragma: no cover - guarded per path above
                raise PlannerError(f"tensor {tid!r}: unknown base tensor {cur.base!r}")
            cum += cur.offset
            cur = nxt
        info.root = cur.id
        info.cum_offset = cum
        if cum % info.alignment != 0:
            raise PlannerError(
                f"tensor {tid!r}: cumulative view offset {cum} is not aligned to "
                f"its alignment {info.alignment}"
            )
        if info.root != info.id and cum + info.size > cur.size:
            raise PlannerError(
                f"tensor {tid!r}: view [offset={cum}, bytes={info.size}) exceeds "
                f"root tensor {cur.id!r} of {cur.size} bytes"
            )

    # effective root alignment: strictest alignment in every view chain
    root_align: dict[str, int] = {}
    for info in tensors.values():
        chain_max = info.alignment
        cur = info
        while cur.base is not None:
            cur = tensors[cur.base]
            chain_max = max(chain_max, cur.alignment)
        root_align[info.root] = max(root_align.get(info.root, 1), chain_max)

    # DAG check over explicit deps + producer/consumer + view-producer edges
    for pid in path_ids:
        ops = path_ops[pid]
        local_ids = {tid for tid, info in tensors.items() if pid in info.occ}
        succ: dict[str, list[str]] = {o: [] for o in ops}
        for oid, (_, deps) in ops.items():
            for d in deps:
                succ[d].append(oid)
        for tid in local_ids:
            info = tensors[tid]
            oc = info.occ[pid]
            for c in oc.consumers:
                if c != oc.producer:  # same-operator access imposes no order
                    succ[oc.producer].append(c)
            if info.base is not None and info.base in local_ids:
                bp = tensors[info.base].occ[pid].producer
                if bp != oc.producer:  # view produced within the base producer
                    succ[bp].append(oc.producer)
        _assert_acyclic(succ, pid)

    roots = sorted({info.root for info in tensors.values()})
    root_size = {r: tensors[r].size for r in roots}
    live_roots = {
        info.root
        for info in tensors.values()
        for oc in info.occ.values()
        if oc.output or oc.consumers
    }
    # A path's makespan never exceeds its own total work, so the per-path
    # maximum is a sufficient global horizon for every operator/liveness var.
    H = max(
        (sum(d for d, _ in ops.values()) for ops in path_ops.values()),
        default=0,
    )
    # Sufficient-memory bound for the diagnostic probes: placing live roots
    # sequentially at aligned addresses uses at most sum(size + align - 1)
    # bytes (the slack accounts for every alignment gap).
    S = sum(
        root_size[r] + root_align.get(r, 1) - 1
        for r in roots
        if r in live_roots
    )
    return Prepared(
        path_ids=path_ids,
        path_ops=path_ops,
        path_op_order=path_op_order,
        tensors=tensors,
        roots=roots,
        root_size=root_size,
        root_align=root_align,
        live_roots=live_roots,
        H=max(H, 1),
        S=max(S, 1),
    )


def _assert_acyclic(succ: dict[str, list[str]], pid: str) -> None:
    color = {o: 0 for o in succ}

    def dfs(u: str) -> None:
        color[u] = 1
        for v in succ[u]:
            if color[v] == 1:
                raise PlannerError(
                    f"path {pid}: operator dependency cycle involving {v!r}"
                )
            if color[v] == 0:
                dfs(v)
        color[u] = 2

    for o in succ:
        if color[o] == 0:
            dfs(o)


# --------------------------------------------------------------------------- #
# CP-SAT model
# --------------------------------------------------------------------------- #
def build_model(prep: Prepared, deadline: int, memory_limit: int) -> ModelArtifacts:
    m = cp_model.CpModel()
    H = prep.H
    s: dict = {}
    f: dict = {}

    for pid in prep.path_ids:
        for oid, (dur, deps) in prep.path_ops[pid].items():
            sv = m.NewIntVar(0, H, f"s_{pid}_{oid}")
            fv = m.NewIntVar(0, H, f"f_{pid}_{oid}")
            m.Add(fv == sv + dur)
            s[(pid, oid)] = sv
            f[(pid, oid)] = fv
            for d in deps:
                m.Add(sv >= f[(pid, d)])

    # data precedences: producer -> consumers, base producer -> view producer.
    # Same-operator reads/views impose no ordering (tensor available within the
    # producing operator) and would be degenerate self constraints.
    for info in prep.tensors.values():
        for pid, oc in info.occ.items():
            for c in oc.consumers:
                if c != oc.producer:
                    m.Add(s[(pid, c)] >= f[(pid, oc.producer)])
            if info.base is not None and pid in prep.tensors[info.base].occ:
                bp = prep.tensors[info.base].occ[pid].producer
                if bp != oc.producer:
                    m.Add(s[(pid, oc.producer)] >= f[(pid, bp)])

    # path windows: each path is scheduled independently and must individually
    # finish before the deadline. Mutually-exclusive paths are NOT serialized:
    # only one of them runs at runtime, so their windows may overlap on the
    # shared timeline.
    sp, ep = {}, {}
    for pid in prep.path_ids:
        op_ids = list(prep.path_ops[pid])
        sv = m.NewIntVar(0, H, f"S_{pid}")
        ev = m.NewIntVar(0, H, f"E_{pid}")
        m.AddMaxEquality(ev, [f[(pid, o)] for o in op_ids])
        m.AddMinEquality(sv, [s[(pid, o)] for o in op_ids])
        m.Add(ev <= deadline)
        sp[pid], ep[pid] = sv, ev

    # root tensor addresses in the shared buffer
    addr: dict = {}
    cap = m.NewIntVar(0, memory_limit, "capacity")
    zero = m.NewConstant(0)
    y_iv: dict = {}
    for r in prep.roots:
        if r not in prep.live_roots:
            # empty lifetime everywhere: reserves nothing, pinned at address 0
            av = m.NewConstant(0)
            y_iv[r] = None
        else:
            av = m.NewIntVar(0, memory_limit, f"a_{r}")
            m.Add(av + prep.root_size[r] <= cap)
            m.AddModuloEquality(zero, av, prep.root_align[r])
            y_iv[r] = m.NewFixedSizeIntervalVar(
                av, prep.root_size[r], f"yiv_{r}"
            )
        addr[r] = av
    m.Add(cap <= memory_limit)

    # liveness intervals of roots within each path; each non-empty lifetime
    # becomes a (time x bytes) rectangle. AddNoOverlap2D is applied per path:
    # in one constraint it makes simultaneously-live roots of the same path
    # disjoint in bytes and keeps every rectangle inside the global capacity.
    # All paths share the same address variables (one global buffer), but
    # rectangles of different paths never appear in one constraint, so
    # mutually-exclusive paths may reuse the same bytes even when their
    # timelines overlap.
    lstart: dict = {}
    lend: dict = {}
    presence: dict = {}
    x_rects: dict[str, list] = {pid: [] for pid in prep.path_ids}
    y_rects: dict[str, list] = {pid: [] for pid in prep.path_ids}

    for pid in prep.path_ids:
        members: dict[str, list[TInfo]] = {}
        for info in prep.tensors.values():
            if pid in info.occ:
                members.setdefault(info.root, []).append(info)
        for r, infos in members.items():
            starts, ends = [], []
            for info in infos:
                oc = info.occ[pid]
                pf = f[(pid, oc.producer)]
                starts.append(pf)
                if oc.output:
                    ends.append(ep[pid])
                for c in oc.consumers:
                    ends.append(f[(pid, c)])
            if not ends:
                # half-open empty interval [t, t): no rectangle, never occupies
                base = starts[0]
                presence[(pid, r)] = m.NewConstant(0)
                lstart[(pid, r)] = base
                lend[(pid, r)] = base
                continue
            ls = m.NewIntVar(0, H, f"ls_{pid}_{r}")
            le = m.NewIntVar(0, H, f"le_{pid}_{r}")
            # occupied as soon as any view of the root is produced ...
            if len(starts) == 1:
                m.Add(ls == starts[0])
            else:
                m.AddMinEquality(ls, starts)
            # ... until the last consumer (of any view) or path end for outputs
            m.AddMaxEquality(le, ends)
            wvar = m.NewIntVar(0, H, f"lw_{pid}_{r}")
            m.Add(wvar == le - ls)
            iv = m.NewIntervalVar(ls, wvar, le, f"xiv_{pid}_{r}")
            presence[(pid, r)] = m.NewConstant(1)
            lstart[(pid, r)] = ls
            lend[(pid, r)] = le
            if y_iv[r] is not None:
                x_rects[pid].append(iv)
                y_rects[pid].append(y_iv[r])

    for pid in prep.path_ids:
        if x_rects[pid]:
            m.AddNoOverlap2D(x_rects[pid], y_rects[pid])

    return ModelArtifacts(m, s, f, sp, ep, addr, cap, lstart, lend, presence)


# --------------------------------------------------------------------------- #
# Solving (lexicographic)
# --------------------------------------------------------------------------- #
_TIME_STAGE1 = 10.0
_TIME_STAGE = 0.3
_TIME_OP_STAGE = 0.08
_TIME_PROBE = 5.0

PROOF_STATUSES = (cp_model.OPTIMAL, cp_model.FEASIBLE)


def _new_solver(time_limit: float) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42
    return solver


def _label(status) -> str:
    # Tri-state mapping: OPTIMAL/FEASIBLE carry a solution, INFEASIBLE is a
    # proof of impossibility, and UNKNOWN (timeout) — together with any other
    # status — is undecided evidence that must never be treated as a proof.
    if status in PROOF_STATUSES:
        return FEASIBLE
    if status == cp_model.INFEASIBLE:
        return INFEASIBLE
    return UNKNOWN


def run_solver(model: cp_model.CpModel, tag: str, time_limit: float) -> tuple:
    """Single seam for CP-SAT calls; tests script outcomes through ``tag``.

    Returns ``(status_label, solver, raw_status)``. The solver is always
    returned: callers decide whether an UNKNOWN solve happens to carry a
    readable (unproven) incumbent via :func:`_try_incumbent`.
    """
    solver = _new_solver(time_limit)
    status = solver.Solve(model)
    label = _label(status)
    return label, solver, status


def _try_incumbent(
    prep: Prepared, art: ModelArtifacts, solver: cp_model.CpSolver
) -> Optional[dict]:
    """Return a snapshot iff the solver holds a readable complete solution.

    A timed-out (UNKNOWN) solve sometimes has an unproven incumbent. Reading
    values raises when none exists, so the attempt is best-effort: any failure
    (or partial data) yields ``None`` rather than a half-populated response.
    """
    try:
        snap = _snapshot(prep, art, solver)
    except Exception:  # OR-Tools raises when there is no solution at all
        return None
    cap = snap["cap"]
    if cap < 0 or cap > art.cap.Proto().domain[-1]:
        return None
    return snap


def _snapshot(
    prep: Prepared, art: ModelArtifacts, solver: cp_model.CpSolver
) -> dict:
    """Capture every value needed to render a response from one incumbent."""
    snap = {
        "cap": solver.Value(art.cap),
        "addr": {r: solver.Value(art.addr[r]) for r in prep.roots},
        "s": {k: solver.Value(v) for k, v in art.s.items()},
        "f": {k: solver.Value(v) for k, v in art.f.items()},
        "sp": {pid: solver.Value(art.sp[pid]) for pid in prep.path_ids},
        "ep": {pid: solver.Value(art.ep[pid]) for pid in prep.path_ids},
        "liveness": {},
    }
    for key, pvar in art.presence.items():
        present = solver.Value(pvar) == 1
        if present:
            snap["liveness"][key] = (
                solver.Value(art.lstart[key]),
                solver.Value(art.lend[key]),
            )
    return snap


def plan(req: PlanRequest, override_seconds: Optional[float] = None) -> PlanResponse:
    prep = prepare(req)
    art = build_model(prep, req.deadline, req.memory_limit)
    m = art.model

    def budget(default: float) -> float:
        # A request-scoped override (the X-Solver-Timeout-Seconds header)
        # makes timeout/UNKNOWN outcomes reproducible for verification.
        return default if override_seconds is None else override_seconds

    # Stage 1: minimize the capacity upper bound.
    m.Minimize(art.cap)
    label, solver, raw = run_solver(m, "first", budget(_TIME_STAGE1))
    if label == FEASIBLE:
        snapshot = _snapshot(prep, art, solver)
        all_optimal = raw == cp_model.OPTIMAL
    elif label == UNKNOWN:
        # An UNKNOWN solve may still carry an unproven incumbent: keep its
        # legal timeline/addresses and answer feasible (never optimal), instead
        # of discarding a perfectly usable plan. No later stage could restore
        # an optimality proof, so the incumbent is returned as-is.
        snapshot = _try_incumbent(prep, art, solver)
        if snapshot is not None:
            return _build_response(prep, art, snapshot, optimal=False)
        return _diagnose(prep, req, first=label, override_seconds=override_seconds)
    else:
        return _diagnose(prep, req, first=label, override_seconds=override_seconds)

    # the capacity is held fixed; later stages only tie-break inside it
    m.Add(art.cap == snapshot["cap"])

    def hint_current() -> None:
        # Warm-start the next solve with the incumbent solution so that the
        # lexicographic stages only need to prove a tiny improvement.
        m.ClearHints()

        def hint(var) -> None:
            # only genuine, as-yet-unfixed decision variables can be hinted
            if isinstance(var, cp_model.IntVar):
                proto = var.Proto()
                if proto.domain and proto.domain[0] < proto.domain[1]:
                    m.AddHint(var, solver.Value(var))

        for var in art.addr.values():
            hint(var)
        hint(art.cap)
        for pid in prep.path_ids:
            hint(art.sp[pid])
            hint(art.ep[pid])
            for oid in prep.path_ops[pid]:
                hint(art.s[(pid, oid)])
                hint(art.f[(pid, oid)])
        for (pid, r), pvar in art.presence.items():
            if isinstance(pvar, cp_model.IntVar) and solver.Value(pvar) == 1:
                hint(art.lstart[(pid, r)])
                hint(art.lend[(pid, r)])

    # A per-root priority that later lexicographic stages are sequenced in.
    root_priority: dict[str, str] = {}
    for tid, info in prep.tensors.items():
        r = info.root
        if r in prep.live_roots:
            root_priority[r] = min(root_priority.get(r, tid), tid)
    root_order = sorted(root_priority, key=lambda x: root_priority[x])
    op_keys = sorted(art.s.keys(), key=lambda k: (k[1], k[0]))

    def stage(var, time_limit: float, tag: str) -> bool:
        nonlocal all_optimal, solver, snapshot
        hint_current()
        m.Minimize(var)
        lbl, incumbent, raw_status = run_solver(m, tag, budget(time_limit))
        if lbl == FEASIBLE:
            solver = incumbent
            snapshot = _snapshot(prep, art, solver)
        if lbl != FEASIBLE or raw_status != cp_model.OPTIMAL:
            # FEASIBLE-but-not-proved, or UNKNOWN: the last incumbent remains a
            # valid feasible plan, but lexicographic optimality cannot be
            # claimed. Fix the incumbent value when we have one and stop.
            if lbl == FEASIBLE:
                m.Add(var == solver.Value(var))
            all_optimal = False
            return False
        m.Add(var == solver.Value(var))
        return True

    # 2. tensor addresses in tensor-id order ...
    for r in root_order:
        if not stage(art.addr[r], _TIME_STAGE, f"addr:{r}"):
            break
    else:
        # 3. ... path starts in path-id order ...
        for pid in prep.path_ids:
            if not stage(art.sp[pid], _TIME_STAGE, f"pstart:{pid}"):
                break
        else:
            # 4. ... operator starts by (operator id, path id).
            for key in op_keys:
                if not stage(art.s[key], _TIME_OP_STAGE, f"ostart:{key[1]}:{key[0]}"):
                    break
    return _build_response(prep, art, snapshot, optimal=all_optimal)


def _build_response(
    prep: Prepared, art: ModelArtifacts, snapshot: dict, optimal: bool
) -> PlanResponse:
    root_addr = snapshot["addr"]

    timeline: list[PathSchedule] = []
    for pid in prep.path_ids:
        ops = [
            OpSchedule(
                operator_id=oid,
                start=snapshot["s"][(pid, oid)],
                end=snapshot["f"][(pid, oid)],
            )
            for oid in prep.path_op_order[pid]
        ]
        peak = _path_peak(prep, snapshot, pid)
        timeline.append(
            PathSchedule(
                path_id=pid,
                start=snapshot["sp"][pid],
                end=snapshot["ep"][pid],
                peak_bytes=peak,
                operators=ops,
            )
        )

    addresses = [
        TensorAddress(
            tensor_id=tid,
            address=root_addr[info.root] + info.cum_offset,
            bytes=info.size,
        )
        for tid, info in sorted(prep.tensors.items())
    ]
    return PlanResponse(
        feasible=True,
        optimal=optimal,
        capacity_upper_bound=snapshot["cap"],
        timeline=timeline,
        addresses=addresses,
    )


def _path_peak(prep: Prepared, snapshot: dict, pid: str) -> int:
    events: list[tuple[int, int, int]] = []  # (time, kind, delta); end before start
    for r in prep.roots:
        span = snapshot["liveness"].get((pid, r))
        if span is None:
            continue
        ls, le = span
        if le <= ls:
            continue
        events.append((ls, 1, prep.root_size[r]))   # acquire
        events.append((le, 0, -prep.root_size[r]))  # release first
    events.sort()
    cur = peak = 0
    for _, _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def _empty_response(reason: Optional[str]) -> PlanResponse:
    return PlanResponse(
        feasible=False,
        optimal=False,
        reason=reason,
        capacity_upper_bound=0,
        timeline=[],
        addresses=[],
    )


def _diagnose(
    prep: Prepared,
    req: PlanRequest,
    first: str,
    override_seconds: Optional[float] = None,
) -> PlanResponse:
    """Adjudicate a first solve that found no feasible solution.

    Evidence is three independent tri-state outcomes:

    * ``first`` — the original problem (requested deadline AND memory);
    * memory-relaxed probe — same deadline, memory raised to its guaranteed
      sufficient bound ``S``; INFEASIBLE there proves the deadline alone can
      never be met, regardless of how much memory is allowed (reason
      ``deadline``);
    * deadline-relaxed probe — same memory, deadline raised to its guaranteed
      sufficient bound ``H``; INFEASIBLE there proves no placement fits in the
      requested memory, regardless of how long the schedule may run (reason
      ``memory_limit``).

    A probe that returns UNKNOWN proves nothing about its resource, so it can
    never pin down an exact resource reason. When the first solve is itself
    UNKNOWN and neither relaxation yields a proof, feasibility remains open
    and the verdict is UNDECIDED (HTTP 503) — never a proven infeasibility.
    """
    seconds = _TIME_PROBE if override_seconds is None else override_seconds

    mem_relaxed, _, _ = run_solver(
        build_model(prep, req.deadline, prep.S).model, "probe:relax_memory", seconds
    )
    dl_relaxed, _, _ = run_solver(
        build_model(prep, prep.H, req.memory_limit).model, "probe:relax_deadline", seconds
    )

    # Proven impossibility of a relaxed problem implies impossibility of the
    # original request; the failing relaxation names the binding resource.
    if mem_relaxed == INFEASIBLE and dl_relaxed == INFEASIBLE:
        reason: Optional[str] = "deadline_and_memory_limit"
        decided = True
    elif mem_relaxed == INFEASIBLE:
        reason = "deadline"
        decided = True
    elif dl_relaxed == INFEASIBLE:
        reason = "memory_limit"
        decided = True
    else:
        # Neither relaxation was proven impossible. If the original problem
        # itself was proven infeasible, impossibility is certain but no exact
        # resource can be named; with an UNKNOWN first solve the feasibility
        # question itself is still open.
        reason = None
        decided = first == INFEASIBLE

    if not decided:
        # Explicitly undecided: feasibility could not be proven either way
        # within the time budget. Must never be presented as "no solution".
        return _empty_response(UNDETERMINED)
    return _empty_response(reason)
