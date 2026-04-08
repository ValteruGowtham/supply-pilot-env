"""
inference.py — SupplyPilot hackathon inference script.

Runs an LLM agent against all three SupplyPilot tasks and emits the
exact log format required by the Scaler x Meta OpenEnv hackathon scorer.
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
IMAGE_NAME: Optional[str] = os.getenv("IMAGE_NAME")

BENCHMARK: str = "supply_pilot"
MAX_STEPS: int = 30
TEMPERATURE: float = 0.3
MAX_TOKENS: int = 150
SUCCESS_SCORE_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = textwrap.dedent(
    """\
    You are a supply chain manager AI. You receive daily inventory data and must decide how to reorder stock.
    You MUST respond with ONLY a valid JSON object. No explanation, no markdown, no code blocks.
    Required JSON keys:
      sku_id: string (SKU_A, SKU_B, or SKU_C)
      order_quantity: integer between 0 and 500
      supplier_id: string, either 'primary' or 'backup'
      expedite: boolean true or false
    Example: {"sku_id": "SKU_A", "order_quantity": 150, "supplier_id": "primary", "expedite": false}"""
)


# ---------------------------------------------------------------------------
# Logging helpers  (format is exact — any deviation = disqualification)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score_or_rewards=None, rewards: List[float] = None) -> None:
    # Accepts either (success, steps, rewards) or legacy (success, steps, score, rewards)
    # score is accepted but intentionally NOT emitted in the [END] line
    if rewards is None:
        rewards = score_or_rewards if isinstance(score_or_rewards, list) else []
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(obs, step: int) -> str:
    return (
        f"Day: {obs.day}\n"
        f"SKU: {obs.sku_id}\n"
        f"Current stock: {obs.stock_level} units\n"
        f"Today demand: {obs.daily_demand:.1f} units\n"
        f"Pending orders: {obs.pending_order_units} units arriving in {obs.supplier_lead_time} days\n"
        f"Stockout today: {obs.stockout_today}\n"
        f"Supplier disruption active: {obs.disruption_active}\n"
        f"Step: {step} of {MAX_STEPS}\n"
        f"\n"
        f"What is your reorder decision? Respond with JSON only."
    )


# ---------------------------------------------------------------------------
# LLM action
# ---------------------------------------------------------------------------

def get_action(client: OpenAI, obs, step: int) -> dict:
    """
    Query the LLM for a reorder decision.

    Returns a dict with keys: sku_id, order_quantity, supplier_id, expedite.
    Falls back to a safe default on any error so the episode can continue.
    """
    _FALLBACK = {
        "sku_id": "SKU_A",
        "order_quantity": 100,
        "supplier_id": "primary",
        "expedite": False,
    }

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs, step)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = completion.choices[0].message.content.strip()

        # Strip markdown code fences if the model wrapped the JSON
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        return json.loads(text)

    except Exception:
        return _FALLBACK


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, env, task_id: str) -> float:
    """
    Run one full episode for the given task_id.

    Args:
        client:  Initialised OpenAI client.
        env:     SupplyPilotEnv async WebSocket client.
        task_id: One of "task_1", "task_2", "task_3".

    Returns:
        Normalised episode score in [0.0, 1.0].
    """
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset — task_id is forwarded to the server via **kwargs
        result = await env.reset(task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            action_dict = get_action(client, obs, step)

            # Short action string for [STEP] log
            action_str = (
                f"order={action_dict.get('order_quantity', 0)},"
                f"sku={action_dict.get('sku_id', 'SKU_A')},"
                f"supplier={action_dict.get('supplier_id', 'primary')}"
            )

            # Import SupplyAction — dual-import so file works both installed
            # and run directly from supply_pilot_env/ directory
            try:
                from supply_pilot_env.models import SupplyAction
            except ImportError:
                from models import SupplyAction  # type: ignore

            action = SupplyAction(
                sku_id=str(action_dict.get("sku_id", "SKU_A")),
                order_quantity=int(action_dict.get("order_quantity", 100)),
                supplier_id=str(action_dict.get("supplier_id", "primary")),
                expedite=bool(action_dict.get("expedite", False)),
            )

            result = await env.step(action)

            reward = float(result.reward) if result.reward is not None else 0.0
            done = bool(result.done)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        # Normalise score: max possible ≈ 0.5 reward/step × 30 steps = 15.0
        total_reward = sum(rewards)
        max_possible = MAX_STEPS * 0.5
        score = max(0.0, min(1.0, total_reward / max_possible if max_possible > 0 else 0.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Import env client — dual-import for installed vs direct-run contexts
    try:
        from supply_pilot_env.client import SupplyPilotEnv
    except ImportError:
        from client import SupplyPilotEnv  # type: ignore

    env = await SupplyPilotEnv.from_docker_image(IMAGE_NAME)

    try:
        for task_id in ["task_1", "task_2", "task_3"]:
            await run_task(client, env, task_id)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
