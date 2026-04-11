"""
inference.py — SupplyPilot hackathon inference script.
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables
# CRITICAL: Validator injects API_KEY, guidelines say HF_TOKEN, support both
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
# Read API_KEY first (what validator injects), fallback to HF_TOKEN (for local testing)
API_KEY: str = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "dummy-key-for-proxy"
IMAGE_NAME: Optional[str] = os.getenv("IMAGE_NAME")

BENCHMARK: str = "supply_pilot"
MAX_STEPS: int = 30
TEMPERATURE: float = 0.3
MAX_TOKENS: int = 200
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
# Logging helpers
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

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
        f"Step: {step} of {MAX_STEPS}\n\n"
        f"What is your reorder decision? Respond with JSON only."
    )

# ---------------------------------------------------------------------------
# LLM action
# ---------------------------------------------------------------------------
def get_action(client: OpenAI, obs, step: int) -> dict:
    """Query LLM for action. This call is tracked by the validator."""
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
        
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(line for line in lines if not line.startswith("```")).strip()
        
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        print(f"[DEBUG] JSON parse failed at step {step}", flush=True)
        return _FALLBACK

# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------
async def run_task(client: OpenAI, env, task_id: str) -> float:
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            action_dict = get_action(client, obs, step)  # LLM call happens here

            action_str = (
                f"order={action_dict.get('order_quantity', 0)},"
                f"sku={action_dict.get('sku_id', 'SKU_A')},"
                f"supplier={action_dict.get('supplier_id', 'primary')}"
            )

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

        total_reward = sum(rewards)
        max_possible = MAX_STEPS * 0.5
        score = max(0.0, min(1.0, total_reward / max_possible if max_possible > 0 else 0.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

    finally:
        log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] API_KEY={'set' if API_KEY else 'NOT SET'}", flush=True)

    # Initialize OpenAI client - uses validator's proxy
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        from supply_pilot_env.client import SupplyPilotEnv
    except ImportError:
        from client import SupplyPilotEnv  # type: ignore

    env = None
    try:
        env = await SupplyPilotEnv.from_docker_image(IMAGE_NAME)
        
        for task_id in ["task_1", "task_2", "task_3"]:
            await run_task(client, env, task_id)
                
    except Exception as e:
        print(f"[DEBUG] Environment error: {e}", flush=True)
        for task_id in ["task_1", "task_2", "task_3"]:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(task=task_id, success=False, steps=0, score=0.0, rewards=[])
            
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[DEBUG] Fatal: {type(e).__name__}: {e}", flush=True)
        import sys
        sys.exit(0)
