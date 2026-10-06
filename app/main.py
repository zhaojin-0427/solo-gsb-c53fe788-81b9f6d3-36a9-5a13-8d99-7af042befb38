"""FastAPI application exposing POST /plans."""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse

from .planner import PlannerError, UNDETERMINED, plan
from .schemas import PlanRequest, PlanResponse

app = FastAPI(
    title="Inference Graph Buffer Planner",
    version="1.0.0",
    description=(
        "Plans operator schedules and tensor buffer placement for mutually "
        "exclusive inference paths."
    ),
)


@app.exception_handler(PlannerError)
async def planner_error_handler(_request, exc: PlannerError) -> JSONResponse:
    # Semantic validation errors share the 422 contract used for schema errors.
    return JSONResponse(
        status_code=422,
        content={"detail": [{"msg": str(exc), "type": "value_error.semantic"}]},
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/plans",
    response_model=PlanResponse,
    responses={503: {"model": PlanResponse, "description": "Result undecided (solver timeout)"}},
)
async def create_plan(
    request: PlanRequest,
    x_solver_timeout_seconds: Optional[str] = Header(default=None),
) -> JSONResponse:
    """Plan a schedule and buffer layout.

    Status codes:

    * **200** — a definite result: either a feasible plan or a *proven*
      infeasibility carrying its resource ``reason``;
    * **503** — feasibility could not be decided within the time budget
      (``reason="undetermined"``). The request may be retried; this is never a
      proof that no plan exists;
    * **422** — the request body is structurally or semantically invalid.
    """
    override: Optional[float] = None
    if x_solver_timeout_seconds is not None:
        try:
            override = float(x_solver_timeout_seconds)
        except (TypeError, ValueError):
            raise PlannerError(
                "X-Solver-Timeout-Seconds must be a positive number of seconds"
            )
        if not (override > 0.0) or override != override or override == float("inf"):
            raise PlannerError(
                "X-Solver-Timeout-Seconds must be a finite positive number of seconds"
            )

    result = plan(request, override_seconds=override)
    status_code = 503 if result.reason == UNDETERMINED else 200
    return JSONResponse(status_code=status_code, content=result.model_dump())
