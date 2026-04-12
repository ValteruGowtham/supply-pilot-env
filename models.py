# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the SupplyPilot environment.

Defines the Action, Observation, and State types used by the supply-chain
reinforcement-learning environment.
"""

from openenv.core.env_server import Action, Observation, State
from pydantic import Field
from typing import Optional, Dict, Any


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SupplyAction(Action):
    """Action submitted by the agent each time-step.

    The agent chooses which SKU to manage, how many units to order, which
    supplier to use, and whether to pay the expedite premium for next-day
    delivery.
    """

    sku_id: str = Field(
        default="SKU_A",
        description="Identifier of the product (SKU) to act on.",
    )
    order_quantity: int = Field(
        default=0,
        description="Number of units to order this step. 0 means no order.",
    )
    supplier_id: str = Field(
        default="primary",
        description='Supplier to source from. One of "primary" or "backup".',
    )
    expedite: bool = Field(
        default=False,
        description="If True, pay a premium for 1-day delivery instead of "
                    "the standard lead time.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SupplyObservation(Observation):
    """Observation returned to the agent after each step.

    Inherits from Observation which already provides:
        done: bool
        reward: float | None
        metadata: Dict
    """

    sku_id: str = Field(
        default="SKU_A",
        description="SKU this observation refers to.",
    )
    stock_level: int = Field(
        default=0,
        description="Current on-hand inventory (units).",
    )
    daily_demand: float = Field(
        default=0.0,
        description="Realised demand for the current day (units).",
    )
    pending_order_units: int = Field(
        default=0,
        description="Total units currently in-transit (not yet received).",
    )
    supplier_lead_time: int = Field(
        default=3,
        description="Expected lead time from the active supplier (days).",
    )
    day: int = Field(
        default=0,
        description="Current simulation day (0-indexed).",
    )
    stockout_today: bool = Field(
        default=False,
        description="True if demand could not be fully met today.",
    )
    holding_cost_today: float = Field(
        default=0.0,
        description="Holding cost incurred today (currency units).",
    )
    disruption_active: bool = Field(
        default=False,
        description="True if a supply disruption is currently active.",
    )
    message: str = Field(
        default="",
        description="Optional human-readable status message from the environment.",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SupplyState(State):
    """Full internal state of the supply-chain environment.

    Inherits from State which already provides:
        episode_id: Optional[str]
        step_count: int
    """

    task_id: str = Field(
        default="task_1",
        description="Identifier of the task / scenario configuration.",
    )
    day: int = Field(
        default=0,
        description="Current simulation day.",
    )
    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward accumulated since episode start.",
    )
    stockout_days: float = Field(
        default=1.0,
        description="Number of days in this episode where a stockout occurred.",
    )
    total_holding_cost: float = Field(
        default=0.0,
        description="Sum of holding costs incurred across all days so far.",
    )
    fill_rate: float = Field(
        default=1.0,
        description="Fraction of demand fulfilled so far (units_fulfilled / units_demanded).",
    )
    disruption_active: bool = Field(
        default=False,
        description="Whether a supply disruption is currently in effect.",
    )
    units_demanded_total: float = Field(
        default=0.0,
        description="Cumulative units demanded across all days in this episode.",
    )
    units_fulfilled_total: float = Field(
        default=0.0,
        description="Cumulative units fulfilled across all days in this episode.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SupplyAction",
    "SupplyObservation",
    "SupplyState",
]
