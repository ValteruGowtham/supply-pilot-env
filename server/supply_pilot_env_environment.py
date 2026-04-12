# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SupplyPilot Environment Implementation.

Simulates inventory management over a 30-day episode across three tasks of
increasing difficulty:
  task_1 — single SKU, stable demand (easy)
  task_2 — multi-SKU, seasonal demand with a daily order budget (medium)
  task_3 — single SKU, supplier disruption on day 10 (hard)
"""

import uuid
import random
from typing import Dict, List, Any

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import SupplyAction, SupplyObservation, SupplyState
except ImportError:
    from models import SupplyAction, SupplyObservation, SupplyState  # type: ignore


# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

_TASK_INFO = [
    {
        "id": "task_1",
        "name": "Stable Demand",
        "difficulty": "easy",
        "description": (
            "Single SKU (SKU_A) with fixed daily demand of 50 units. "
            "Keep stock above zero over 30 days using the primary supplier "
            "(lead time 3 days) or the backup (lead time 1 day, 2× cost)."
        ),
    },
    {
        "id": "task_2",
        "name": "Seasonal Multi-SKU",
        "difficulty": "medium",
        "description": (
            "Three SKUs with seasonal demand patterns and a total daily order "
            "budget of 500 units. SKU_A spikes on Mondays/Fridays, SKU_B has "
            "a mid-month surge, SKU_C has random demand. Primary lead time: 2 days."
        ),
    },
    {
        "id": "task_3",
        "name": "Supplier Disruption",
        "difficulty": "hard",
        "description": (
            "Single SKU (SKU_A) with fixed demand of 60 units/day. On day 10 "
            "the primary supplier is disrupted and its lead time jumps to 14 days. "
            "Score rewards both service level and how quickly the agent switches "
            "to the backup supplier."
        ),
    },
]

# Per-task static configuration
_TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "task_1": {
        "skus": ["SKU_A"],
        "starting_stock": {"SKU_A": 200},
        "fixed_demand": {"SKU_A": 50},
        "primary_lead_time": 3,
        "backup_lead_time": 1,
        "order_budget": None,
    },
    "task_2": {
        "skus": ["SKU_A", "SKU_B", "SKU_C"],
        "starting_stock": {"SKU_A": 150, "SKU_B": 100, "SKU_C": 120},
        "fixed_demand": None,           # computed per-day
        "primary_lead_time": 2,
        "backup_lead_time": 1,
        "order_budget": 500,
    },
    "task_3": {
        "skus": ["SKU_A"],
        "starting_stock": {"SKU_A": 250},
        "fixed_demand": {"SKU_A": 60},
        "primary_lead_time": 3,         # becomes 14 after disruption
        "backup_lead_time": 1,
        "order_budget": None,
    },
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupplyPilotEnvironment(Environment):
    """
    Supply-chain RL environment for the SupplyPilot project.

    Each episode spans ``self._episode_length`` days (default 30).

    Typical usage::

        env = SupplyPilotEnvironment()
        obs = env.reset(task_id="task_3")
        while not obs.done:
            action = SupplyAction(order_quantity=80, supplier_id="backup")
            obs = env.step(action)
        print(env.get_score())
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise with a default task_1 episode."""
        self._episode_length: int = 30
        self._disruption_day: int = 10          # task_3 only

        # These are overwritten by reset(); set sensible defaults so that
        # the object is never in a completely uninitialised state.
        self._task_id: str = "task_1"
        self._day: int = 0
        self._stock: Dict[str, int] = {"SKU_A": 200}
        self._pending_orders: List[Dict[str, Any]] = []
        self._primary_down: bool = False
        self._days_after_disruption: int = 0
        self._days_primary_used_after_disruption: int = 0

        # Rolling totals for fill-rate tracking
        self._units_demanded_total: float = 0.0
        self._units_fulfilled_total: float = 0.0

        self._state = SupplyState(
            task_id="task_1",
            episode_id=str(uuid.uuid4()),
            step_count=0,
            day=0,
        )

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task_1", **kwargs) -> SupplyObservation:
        """
        Start a new episode.

        Args:
            task_id: One of ``"task_1"``, ``"task_2"``, ``"task_3"``.
                     Defaults to ``"task_1"`` if an unknown value is supplied.
            **kwargs: Absorbed for compatibility with the OpenEnv framework.

        Returns:
            Initial :class:`SupplyObservation` with ``done=False``.
        """
        if task_id not in _TASK_CONFIG:
            task_id = "task_1"

        cfg = _TASK_CONFIG[task_id]
        episode_id = str(uuid.uuid4())

        # Reset bookkeeping
        self._task_id = task_id
        self._day = 0
        self._stock = dict(cfg["starting_stock"])
        self._pending_orders = []
        self._primary_down = False
        self._days_after_disruption = 0
        self._days_primary_used_after_disruption = 0
        self._units_demanded_total = 0.0
        self._units_fulfilled_total = 0.0

        self._state = SupplyState(
            task_id=task_id,
            episode_id=episode_id,
            step_count=0,
            day=0,
            total_reward=0.0,
            stockout_days=0,
            total_holding_cost=0.0,
            fill_rate=1.0,
            disruption_active=False,
            units_demanded_total=0.0,
            units_fulfilled_total=0.0,
        )

        return SupplyObservation(
            sku_id="SKU_A",
            stock_level=self._stock["SKU_A"],
            daily_demand=0.0,
            pending_order_units=0,
            supplier_lead_time=cfg["primary_lead_time"],
            day=0,
            stockout_today=False,
            holding_cost_today=0.0,
            disruption_active=False,
            done=False,
            reward=0.0,
            message="Episode started",
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: SupplyAction) -> SupplyObservation:  # type: ignore[override]
        """
        Advance the simulation by one day.

        Args:
            action: Agent's :class:`SupplyAction` for the current day.

        Returns:
            :class:`SupplyObservation` describing the outcome of this step.
        """
        cfg = _TASK_CONFIG[self._task_id]

        # ------------------------------------------------------------------ #
        # 1. DELIVER pending orders that arrive on or before today             #
        # ------------------------------------------------------------------ #
        still_pending: List[Dict[str, Any]] = []
        for order in self._pending_orders:
            if order["arrive_on_day"] <= self._day:
                sku = order["sku_id"]
                if sku in self._stock:
                    self._stock[sku] += order["quantity"]
            else:
                still_pending.append(order)
        self._pending_orders = still_pending

        # ------------------------------------------------------------------ #
        # 2. COMPUTE DEMAND for today                                          #
        # ------------------------------------------------------------------ #
        demand: Dict[str, float] = self._compute_demand(self._day, cfg)

        # ------------------------------------------------------------------ #
        # 3. FULFILL DEMAND                                                    #
        # ------------------------------------------------------------------ #
        units_fulfilled: Dict[str, float] = {}
        stockout_per_sku: Dict[str, bool] = {}
        holding_cost_per_sku: Dict[str, float] = {}

        for sku in cfg["skus"]:
            d = demand[sku]
            available = self._stock.get(sku, 0)
            fulfilled = min(available, d)
            self._stock[sku] = available - int(fulfilled)
            units_fulfilled[sku] = fulfilled
            stockout_per_sku[sku] = fulfilled < d
            holding_cost_per_sku[sku] = self._stock[sku] * 0.001

        # ------------------------------------------------------------------ #
        # 4. HANDLE DISRUPTION (task_3 only)                                   #
        # ------------------------------------------------------------------ #
        if self._task_id == "task_3":
            if self._day == self._disruption_day:
                self._primary_down = True
            if self._primary_down:
                self._days_after_disruption += 1
                if action.supplier_id == "primary":
                    self._days_primary_used_after_disruption += 1

        # ------------------------------------------------------------------ #
        # 5. PROCESS ORDER ACTION                                              #
        # ------------------------------------------------------------------ #
        if action.order_quantity > 0:
            self._place_order(action, cfg)

        # ------------------------------------------------------------------ #
        # 6. COMPUTE REWARD                                                    #
        # ------------------------------------------------------------------ #
        step_reward, any_stockout = self._compute_reward(
            demand, units_fulfilled, stockout_per_sku, holding_cost_per_sku, cfg
        )

        # ------------------------------------------------------------------ #
        # 7. UPDATE STATE                                                      #
        # ------------------------------------------------------------------ #
        self._day += 1
        self._state.step_count += 1
        self._state.day = self._day

        # Rolling demand / fulfilled totals
        for sku in cfg["skus"]:
            self._units_demanded_total += demand[sku]
            self._units_fulfilled_total += units_fulfilled[sku]

        self._state.units_demanded_total = self._units_demanded_total
        self._state.units_fulfilled_total = self._units_fulfilled_total

        if self._units_demanded_total > 0:
            self._state.fill_rate = (
                self._units_fulfilled_total / self._units_demanded_total
            )

        if any_stockout:
            self._state.stockout_days += 1

        # Accumulate holding cost for SKU_A
        self._state.total_holding_cost += holding_cost_per_sku.get("SKU_A", 0.0)
        self._state.disruption_active = self._primary_down

        # ------------------------------------------------------------------ #
        # 8. CHECK DONE + COMPLETION BONUS                                     #
        # ------------------------------------------------------------------ #
        done = self._day >= self._episode_length

        if done:
            completion_bonus = 1.0 if self._state.fill_rate > 0.95 else 0.0
            step_reward = max(-1.0, min(1.0, step_reward + completion_bonus))

        self._state.total_reward += step_reward

        # ------------------------------------------------------------------ #
        # 9. BUILD OBSERVATION                                                 #
        # ------------------------------------------------------------------ #
        primary_sku_lead = self._effective_primary_lead_time(cfg)
        obs_lead_time = (
            14 if (self._task_id == "task_3" and self._primary_down)
            else primary_sku_lead
        )

        pending_units_a = sum(
            o["quantity"] for o in self._pending_orders if o["sku_id"] == "SKU_A"
        )

        message = (
            f"Day {self._day}: stock={self._stock['SKU_A']}, "
            f"demand={demand['SKU_A']:.0f}"
        )

        return SupplyObservation(
            sku_id="SKU_A",
            stock_level=self._stock["SKU_A"],
            daily_demand=float(demand["SKU_A"]),
            pending_order_units=pending_units_a,
            supplier_lead_time=obs_lead_time,
            day=self._day,
            stockout_today=stockout_per_sku["SKU_A"],
            holding_cost_today=holding_cost_per_sku["SKU_A"],
            disruption_active=self._primary_down,
            done=done,
            reward=step_reward,
            message=message,
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> SupplyState:
        """Return the current :class:`SupplyState`."""
        return self._state

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_score(self) -> float:
        """
        Compute end-of-episode score in [0.0, 1.0].

        Returns:
            Scalar score appropriate for the current task.
        """
        if self._task_id == "task_1":
            score = 1.0 - (self._state.stockout_days / self._episode_length)

        elif self._task_id == "task_2":
            score = self._state.fill_rate

        else:  # task_3
            service_score = self._state.fill_rate
            if self._days_after_disruption > 0:
                switch_score = 1.0 - (
                    self._days_primary_used_after_disruption
                    / self._days_after_disruption
                )
            else:
                switch_score = 1.0
            score = (service_score * 0.6) + (switch_score * 0.4)

        # Deep validation expects strict bounds: 0 < score < 1.
        # Use a practical margin to avoid downstream rounding to 0.0/1.0.
        eps = 1e-2
        return max(eps, min(1.0 - eps, score))

    # ------------------------------------------------------------------
    # Task metadata
    # ------------------------------------------------------------------

    @classmethod
    def get_task_info(cls) -> List[Dict[str, Any]]:
        """
        Return metadata for all available tasks.

        Returns:
            List of dicts with keys: ``id``, ``name``, ``difficulty``,
            ``description``.
        """
        return _TASK_INFO

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_demand(
        self, day: int, cfg: Dict[str, Any]
    ) -> Dict[str, float]:
        """Return today's demand for each SKU."""
        if cfg["fixed_demand"] is not None:
            return {sku: float(cfg["fixed_demand"][sku]) for sku in cfg["skus"]}

        # task_2 seasonal demand
        demand: Dict[str, float] = {}
        demand["SKU_A"] = 60.0 if (day % 7 == 0 or day % 7 == 4) else 40.0
        demand["SKU_B"] = 90.0 if (10 <= day <= 15) else 30.0
        rng = random.Random(42 + day)
        demand["SKU_C"] = float(rng.randint(20, 70))
        return demand

    def _effective_primary_lead_time(self, cfg: Dict[str, Any]) -> int:
        """Lead time for the primary supplier given current disruption state."""
        if self._task_id == "task_3" and self._primary_down:
            return 14
        return int(cfg["primary_lead_time"])

    def _place_order(self, action: SupplyAction, cfg: Dict[str, Any]) -> None:
        """Validate action, apply budget cap (task_2), compute lead time, enqueue order."""
        sku = action.sku_id if action.sku_id in self._stock else "SKU_A"
        quantity = max(0, action.order_quantity)

        # task_2 order-budget cap (proportional scaling against single-SKU order)
        if cfg["order_budget"] is not None and quantity > cfg["order_budget"]:
            quantity = cfg["order_budget"]

        if quantity == 0:
            return

        # Lead time
        if action.supplier_id == "backup":
            lead_time = int(cfg["backup_lead_time"])
        else:
            lead_time = self._effective_primary_lead_time(cfg)

        if action.expedite:
            lead_time = max(1, lead_time - 1)

        self._pending_orders.append(
            {
                "sku_id": sku,
                "quantity": quantity,
                "arrive_on_day": self._day + lead_time,
                "supplier": action.supplier_id,
            }
        )

    def _compute_reward(
        self,
        demand: Dict[str, float],
        units_fulfilled: Dict[str, float],
        stockout_per_sku: Dict[str, bool],
        holding_cost_per_sku: Dict[str, float],
        cfg: Dict[str, Any],
    ):
        """
        Compute the per-step reward averaged across all SKUs.

        Returns:
            Tuple of (step_reward clamped to [-1, 1], any_stockout bool).
        """
        skus = cfg["skus"]
        total_service = 0.0
        total_stockout_penalty = 0.0
        total_holding = 0.0
        any_stockout = False

        for sku in skus:
            d = demand[sku]
            service = units_fulfilled[sku] / d if d > 0 else 1.0
            stockout_penalty = -0.3 if stockout_per_sku[sku] else 0.0
            holding = holding_cost_per_sku[sku]

            total_service += service
            total_stockout_penalty += stockout_penalty
            total_holding += holding

            if stockout_per_sku[sku]:
                any_stockout = True

        n = len(skus)
        avg_service = total_service / n
        avg_stockout = total_stockout_penalty / n
        avg_holding = total_holding / n

        step_reward = (avg_service * 0.5) + avg_stockout - avg_holding
        return max(-1.0, min(1.0, step_reward)), any_stockout
