# Data Labelling Spec

This module extracts reward/punishment labels from raw capture frames produced by the `input_capture` module. Each per-frame JSON (`frames/raw/frame_NNNNNN.json`) contains a full-screen JPEG, audio, and input state. The data labelling pipeline reads these frames and produces numeric reward signals for RL training.

All pixel coordinates below assume a **1920x1080** screen resolution.

---

## Reward Signals

### Primary Rewards

#### 1. Time Burn

Measures seconds burned from the survivor clock per minute. This is the single most important reward signal — it directly tracks the Mastermind's core objective.

- **Extraction:** OCR the time-burn region of the screen.
- **Region:** Top-left corner at **(1146, 68)**, box size **179x81** pixels.
- **Value type:** Positive integer (seconds burned). A negative value (survivors gaining time) should be treated as a punishment.
- **Deduplication:** The on-screen decrement animation persists for several frames. Deduplicate so each time-burn event is counted once.

#### 2. Bio Efficiency

Measures how much value (damage / time burn) is generated per unit of bio energy spent. Tracks resource efficiency.

- **Extraction:** OCR the bio energy counter over time. Compute a ratio of value gained to bio spent in a rolling window.
- **Region:** Top-left corner at **(122, 938)**, box size **64x76** pixels.
- **Value type:** Positive integer (current bio energy). This is a discontinuous function — it changes in discrete steps as bio is spent or regenerated.

#### 3. Survivor Debuffs

Reward for negative status effects on survivors (eliminations, damage, infections, downs).

- **Extraction:** Sample pixels from survivor status icon regions and the side health bar. Classify the dominant colour:
  - **Red** = damaged/downed (high reward)
  - **Purple** = infected (high reward)
  - **Yellow** = debuffed (moderate reward)
  - **Green** = healthy (no reward)
- **Regions:** All icon boxes are **62x53** pixels.

| Survivor | Icon Top-Left |
|----------|---------------|
| 1        | (114, 227)    |
| 2        | (114, 293)    |
| 3        | (114, 361)    |
| 4        | (114, 427)    |

### Tertiary Rewards

#### 4. Camera Uptime

Reward for keeping the camera active. A destroyed camera means lost map vision.

- **Extraction:** Check the camera icon colour.
  - **Red** = camera disabled (punishment).
  - **Any other colour** = camera active (reward).
- **Region:** Top-left corner at **(1745, 81)**, box size **61x41** pixels.

---

## Punishment Signals

| Signal | Description |
|--------|-------------|
| **Survivor time gain** | Detected via the Time Burn metric going negative. Compounds with the primary reward calculation. |
| **Bio depletion without results** | Bio energy drops with no corresponding time burn or survivor debuff within a short window. Derived from Bio Efficiency + Time Burn. |
| **Repeated input sequences** | Identical or near-identical input sequences detected within a configurable time window. Discourages the agent from "spamming". |
| **Camera broken while watching** | Camera icon transitions to red. Indicates the agent failed to defend the camera it was using. |

---

## Delayed Reward Tracking

Some actions (e.g. spawning creatures) have outcomes that only manifest 1-3 seconds later. The labelling pipeline must support a delayed evaluation window:

1. When a spawn/placement action is detected, record the timestamp.
2. After 1-3 seconds, evaluate:
   - Did time burn increase? (reward)
   - Did survivors gain time? (punishment)
   - Was the spawn near an objective? (context signal)
   - Did the spawn produce any measurable effect? (no effect = mild punishment)

This delay is a property of the learning algorithm (e.g. temporal-difference or n-step returns) but the labelling pipeline must provide the raw outcome data at the correct timestamps.

---

## Extraction Method Summary

| Metric | Method | Region (x, y, w, h) | Output |
|--------|--------|----------------------|--------|
| Time Burn | OCR | (1146, 68, 179, 81) | int (seconds burned, can be negative) |
| Bio Energy | OCR | (122, 938, 64, 76) | int (current bio energy) |
| Survivor 1 Status | Pixel colour sampling | (114, 227, 62, 53) | enum {red, purple, yellow, green} |
| Survivor 2 Status | Pixel colour sampling | (114, 293, 62, 53) | enum {red, purple, yellow, green} |
| Survivor 3 Status | Pixel colour sampling | (114, 361, 62, 53) | enum {red, purple, yellow, green} |
| Survivor 4 Status | Pixel colour sampling | (114, 427, 62, 53) | enum {red, purple, yellow, green} |
| Camera Status | Pixel colour check | (1745, 81, 61, 41) | bool (active / disabled) |
| Input Repetition | Input event analysis | N/A (from `input_raw`) | float (repetition score) |
