---
title: SupplyPilot
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - rl
  - supply-chain
  - inventory
---

# SupplyPilot — Supply Chain RL Environment

A real-world reinforcement learning environment built on Meta's OpenEnv framework.
An AI agent manages inventory levels, reorder decisions, and supplier selection
across a simulated supply chain over a 30-day episode.

## Environment Description

SupplyPilot simulates the daily decision-making of a supply chain manager:
observe current stock levels and demand signals, then decide when and how much
to reorder from which supplier to avoid stockouts while minimising holding costs.

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| sku_id | string | Product to act on (SKU_A, SKU_B, SKU_C) |
| order_quantity | integer 0-500 | Units to order, 0 = no order |
| supplier_id | string | "primary" (cheaper, slower) or "backup" (expensive, fast) |
| expedite | boolean | Pay premium to reduce lead time by 1 day |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| sku_id | string | Current SKU being observed |
| stock_level | integer | Current units on hand |
| daily_demand | float | Today's demand in units |
| pending_order_units | integer | Units already ordered, not yet arrived |
| supplier_lead_time | integer | Days until pending order arrives |
| day | integer | Current day (0–30) |
| stockout_today | boolean | True if demand exceeded stock |
| disruption_active | boolean | True if primary supplier is disrupted |
| reward | float | Step reward |
| done | boolean | True if episode complete |

## Tasks

### Task 1 — Easy: Single SKU stable demand
Single product (SKU_A), fixed demand of 50 units/day, 200 units starting stock.
Agent must reorder at the right time to avoid stockouts over 30 days.
**Grader:** score = 1.0 − (stockout_days / 30)

### Task 2 — Medium: Multi-SKU seasonal demand
Three products with variable demand patterns under a 500-unit daily order budget.
SKU_A spikes on Monday/Friday, SKU_B spikes mid-month, SKU_C is random.
**Grader:** score = weighted fill rate across all SKUs

### Task 3 — Hard: Supplier disruption
Primary supplier goes down on day 10, extending lead time from 3 to 14 days.
Agent must detect the disruption signal and switch to the backup supplier.
**Grader:** score = 0.6 × service_level + 0.4 × supplier_switch_score

## Setup & Usage

### Docker
```bash
docker build -t supply-pilot-env -f server/Dockerfile .
docker run -p 7860:7860 supply-pilot-env
```

### Local
```bash
pip install openenv-core
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Validate
```bash
openenv validate
```

## Baseline Scores

| Task | Difficulty | Baseline Score |
|------|-----------|---------------|
| task_1 | Easy | TBD |
| task_2 | Medium | TBD |
| task_3 | Hard | TBD |

## Author
Gowtham1111
