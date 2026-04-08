# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the SupplyPilot Environment.

Exposes the supply-chain RL environment over HTTP and WebSocket endpoints,
compatible with EnvClient.

Standard OpenEnv endpoints (provided by create_app):
    POST /step    — Execute an action
    GET  /state   — Get current environment state
    GET  /schema  — Get action/observation schemas
    WS   /ws      — WebSocket endpoint for persistent sessions

Additional SupplyPilot endpoints:
    POST /reset   — Reset the environment (supports optional task_id body)
    GET  /health  — Health check
    GET  /tasks   — List available task configurations

Usage:
    # Development:
    uvicorn supply_pilot_env.server.app:app --reload --host 0.0.0.0 --port 7860

    # Production entry point (defined in pyproject.toml):
    uv run --project . server

    # Direct:
    python -m supply_pilot_env.server.app
"""

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from openenv.core.env_server import create_app

try:
    from ..models import SupplyAction, SupplyObservation
    from .supply_pilot_env_environment import SupplyPilotEnvironment
except ImportError:
    try:
        from supply_pilot_env.models import SupplyAction, SupplyObservation  # type: ignore
        from supply_pilot_env.server.supply_pilot_env_environment import SupplyPilotEnvironment  # type: ignore
    except ImportError:
        from models import SupplyAction, SupplyObservation  # type: ignore[assignment]
        from server.supply_pilot_env_environment import SupplyPilotEnvironment  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Shared environment instance
# ---------------------------------------------------------------------------
# We keep a single long-lived instance and pass a factory lambda so that
# create_app (which expects a callable) always receives the same object.
# This lets our custom /reset route share state with the OpenEnv /step route.

env_instance = SupplyPilotEnvironment()

app: FastAPI = create_app(
    lambda: env_instance,
    SupplyAction,
    SupplyObservation,
    env_name="supply_pilot",
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1"


# ---------------------------------------------------------------------------
# Custom routes
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset_env(request: ResetRequest = None):
    """
    Reset the environment, optionally selecting a task.

    Body (JSON, all fields optional):
        {"task_id": "task_1" | "task_2" | "task_3"}

    An empty body ``{}`` is also accepted and defaults to ``task_1``.
    """
    if request is None:
        request = ResetRequest()
    task_id = request.task_id or "task_1"
    obs = env_instance.reset(task_id=task_id)
    return {
        "observation": {
            "sku_id": obs.sku_id,
            "stock_level": obs.stock_level,
            "daily_demand": obs.daily_demand,
            "pending_order_units": obs.pending_order_units,
            "supplier_lead_time": obs.supplier_lead_time,
            "day": obs.day,
            "stockout_today": obs.stockout_today,
            "holding_cost_today": obs.holding_cost_today,
            "disruption_active": obs.disruption_active,
            "message": obs.message,
            "metadata": {},
        },
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/health")
async def health():
    """Liveness probe — returns HTTP 200 when the server is running."""
    return {"status": "ok", "env": "supply_pilot"}


@app.get("/tasks")
async def get_tasks():
    """Return metadata for all available tasks."""
    return {"tasks": SupplyPilotEnvironment.get_task_info()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Start the uvicorn server.

    Called by the ``server`` script defined in pyproject.toml:
        uv run --project . server
    """
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
