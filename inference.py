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
# Environment variables — exactly matching official guidelines PDF
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
IMAGE_NAME = os.getenv("IMAGE_NAME")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client exactly as shown in official guidelines
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

BENCHMARK = "supply_pilot"
MAX_STEPS = 30
TEMPERATURE = 0.3
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Import SupplyAction once at module level — not inside loop
# ---------------------------------------------------------------------------
try:
    from supply_pilot_env.models import SupplyAction
except ImportError:
    from models import SupplyAction  # type: ignore

try:
    from supply_pilot_env.client import SupplyPilotEnv
except ImportError:
    from client import SupplyPilotEnv  # type: ignore

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent(
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
# Logging helpers — official format from guidelines PDF
# [END] has NO score= field per the official PDF example
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

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
# LLM action — uses module-level client, no broad exception hiding
# ---------------------------------------------------------------------------
def get_action(obs, step: int) -> dict:
    _FALLBACK = {"sku_id": "SKU_A", "order_quantity": 100, "supplier_id": "primary", "expedite": False}
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
    except json.JSONDecodeError:
        print(f"[DEBUG] JSON parse error at step {step}", flush=True)
        return _FALLBACK
    except Exception as e:
        print(f"[DEBUG] LLM error at step {step}: {type(e).__name__}: {e}", flush=True)
        return _FALLBACK

# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------
async def run_task(env, task_id: str) -> float:
    rewards: List[float] = []
    steps_taken: int = 0
    success: bool = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            action_dict = get_action(obs, step)

            action_str = (
                f"order={action_dict.get('order_quantity', 0)},"
                f"sku={action_dict.get('sku_id', 'SKU_A')},"
                f"supplier={action_dict.get('supplier_id', 'primary')}"
            )

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

        score = sum(rewards) / (MAX_STEPS * 0.5)
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {type(e).__name__}: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return sum(rewards)

# ---------------------------------------------------------------------------
# Entry point — mirrors official sample structure exactly
# ---------------------------------------------------------------------------
async def main() -> None:
    env = await SupplyPilotEnv.from_docker_image(IMAGE_NAME)

    try:
        for task_id in ["task_1", "task_2", "task_3"]:
            await run_task(env, task_id)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
