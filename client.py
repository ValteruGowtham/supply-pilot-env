# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SupplyPilot Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SupplyAction, SupplyObservation, SupplyState
except ImportError:
    try:
        from supply_pilot_env.models import SupplyAction, SupplyObservation, SupplyState  # type: ignore
    except ImportError:
        from models import SupplyAction, SupplyObservation, SupplyState  # type: ignore[assignment]


class SupplyPilotEnv(
    EnvClient[SupplyAction, SupplyObservation, SupplyState]
):
    """
    Client for the SupplyPilot supply-chain RL environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with low latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with SupplyPilotEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.stock_level)
        ...
        ...     action = SupplyAction(sku_id="SKU_A", order_quantity=50)
        ...     result = client.step(action)
        ...     print(result.observation.stock_level)

    Example with Docker:
        >>> client = SupplyPilotEnv.from_docker_image("supply_pilot_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SupplyAction(order_quantity=100))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SupplyAction) -> Dict:
        """
        Convert SupplyAction to JSON payload for the step message.

        Args:
            action: SupplyAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "sku_id": action.sku_id,
            "order_quantity": action.order_quantity,
            "supplier_id": action.supplier_id,
            "expedite": action.expedite,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SupplyObservation]:
        """
        Parse server response into StepResult[SupplyObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult wrapping a SupplyObservation
        """
        obs_data = payload.get("observation", {})
        observation = SupplyObservation(
            sku_id=obs_data.get("sku_id", "SKU_A"),
            stock_level=obs_data.get("stock_level", 0),
            daily_demand=obs_data.get("daily_demand", 0.0),
            pending_order_units=obs_data.get("pending_order_units", 0),
            supplier_lead_time=obs_data.get("supplier_lead_time", 3),
            day=obs_data.get("day", 0),
            stockout_today=obs_data.get("stockout_today", False),
            holding_cost_today=obs_data.get("holding_cost_today", 0.0),
            disruption_active=obs_data.get("disruption_active", False),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SupplyState:
        """
        Parse server response into SupplyState.

        Args:
            payload: JSON response from state request

        Returns:
            SupplyState with current episode and simulation statistics
        """
        return SupplyState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "task_1"),
            day=payload.get("day", 0),
            total_reward=payload.get("total_reward", 0.0),
            stockout_days=payload.get("stockout_days", 0),
            total_holding_cost=payload.get("total_holding_cost", 0.0),
            fill_rate=payload.get("fill_rate", 1.0),
            disruption_active=payload.get("disruption_active", False),
            units_demanded_total=payload.get("units_demanded_total", 0.0),
            units_fulfilled_total=payload.get("units_fulfilled_total", 0.0),
        )
