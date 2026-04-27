"""train_online.py — Online DQN fine-tuning against the live game.

Captures the screen at ~10 Hz, feeds frames through the pretrained network,
executes actions via pynput, collects rewards from the live game, and
fine-tunes the model via experience replay (DQN with hard target network).

Toggle key: F8 — pause/resume training mid-session.
  Paused  → all held keys released, no actions sent, no gradient updates.
  Resumed → training resumes from where it left off.

Prerequisites:
    pip install mss pynput
    Run as Administrator on Windows so SendInput reaches the game.

Usage:
    python online_training/train_online.py \\
        --checkpoint modeling/checkpoints/2026-03-21-deep_q_v1.1.pt
    python online_training/train_online.py \\
        --checkpoint <path> --max-steps 100000 --dry-run
"""

from __future__ import annotations

import argparse
import copy
import datetime
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pynput import keyboard as _pynput_kb

# ── sys.path setup ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent   # RL_Resistance_MM/
sys.path.insert(0, str(_ROOT / "modeling"))
# Python automatically adds this script's directory (online_training/) to
# sys.path[0], so sibling imports (config, frame_stack, etc.) work as-is.

from experiment import ExperimentConfig, build_model
from config import OnlineConfig
from frame_stack import FrameStack
from live_reward import LiveRewardExtractor
from action_executor import ActionExecutor
from replay_buffer import ReplayBuffer

try:
    import mss as _mss
    from PIL import Image as _PIL_Image
    _CAPTURE_AVAILABLE = True
except ImportError:
    _CAPTURE_AVAILABLE = False

logger = logging.getLogger(__name__)

_TOGGLE_KEY = _pynput_kb.Key.f8


# ── Online Q-loss ─────────────────────────────────────────────────────────────

def _online_q_loss(
    q_pred: torch.Tensor,      # (B, action_dim)
    actions: torch.Tensor,     # (B, action_dim) — 1.0 active, 0.0 inactive
    target_vals: torch.Tensor, # (B,) — Bellman bootstrapped targets
    l1_weight: float,
    space_idx: int,
) -> torch.Tensor:
    """Bellman-bootstrapped masked Q-loss (mirrors offline masked_q_loss, no confidence bonus)."""
    active   = (actions != 0).float()
    inactive = 1.0 - active

    target_exp = target_vals.unsqueeze(1).expand_as(q_pred)
    n_active = active.sum().clamp(min=1)
    mse_loss = ((q_pred - target_exp) ** 2 * active).sum() / n_active

    # key_space exempt from inactive L1 — same convention as offline training
    inactive_masked = inactive.clone()
    inactive_masked[:, space_idx] = 0.0
    n_inactive = inactive_masked.sum().clamp(min=1)
    l1_loss = (q_pred.abs() * inactive_masked).sum() / n_inactive

    return mse_loss + l1_weight * l1_loss


# ── Screen capture ────────────────────────────────────────────────────────────

def _make_capture_fn(cfg: OnlineConfig):
    """Return a zero-arg callable that grabs the primary monitor as a PIL RGB Image."""
    if not _CAPTURE_AVAILABLE:
        raise RuntimeError("mss not installed.  Run: pip install mss pillow")
    sct = _mss.mss()
    w, h = cfg.resolution
    monitor = {"top": 0, "left": 0, "width": w, "height": h}

    def _capture() -> "_PIL_Image.Image":
        raw = sct.grab(monitor)
        arr = np.array(raw)              # (H, W, 4) BGRA uint8
        rgb = arr[:, :, :3][:, :, ::-1] # BGRA → RGB
        return _PIL_Image.fromarray(rgb.astype(np.uint8))

    return _capture


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_pretrained(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[nn.Module, ExperimentConfig]:
    ckpt = torch.load(ckpt_path, map_location=device)
    exp_cfg = ExperimentConfig.from_checkpoint(ckpt)
    model = build_model(exp_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, exp_cfg


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    exp_cfg: ExperimentConfig,
    step: int,
    losses: list[float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    path = output_dir / f"{today}-online-{exp_cfg.name}-step{step}.pt"
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "experiment_config":    exp_cfg.to_dict(),
        "online_step":          step,
        "online_losses":        losses,
    }, path)
    logger.info(f"Checkpoint saved: {path}")


def _save_final_model(
    model: nn.Module,
    exp_cfg: ExperimentConfig,
    output_dir: Path,
) -> None:
    """Final model in pretrained format — loadable by run_agent.py and _load_pretrained."""
    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    path = output_dir / f"{today}-online-{exp_cfg.name}-final.pt"
    torch.save({
        "model_state_dict":  model.state_dict(),
        "experiment_config": exp_cfg.to_dict(),
    }, path)
    logger.info(f"Final model saved: {path}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    online_cfg = OnlineConfig()

    # ── Load pretrained model ─────────────────────────────────────────────────
    online_model, exp_cfg = _load_pretrained(Path(args.checkpoint), device)
    target_model = copy.deepcopy(online_model)
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)
    # Keep online_model in eval by default; BatchNorm is undefined over batch_size=1
    # in train mode — switch to train only during multi-sample gradient updates.
    online_model.eval()

    output_columns = list(exp_cfg.output_columns)
    space_idx      = output_columns.index("key_space")
    action_dim     = exp_cfg.num_outputs
    img_size       = exp_cfg.img_size[0]   # assume square (H == W)
    stack_size     = exp_cfg.stack_size

    logger.info(
        f"Loaded {exp_cfg.name} | img={img_size}px  stack={stack_size}  "
        f"actions={action_dim}  buffer_cap={online_cfg.buffer_capacity}"
    )

    # ── Components ────────────────────────────────────────────────────────────
    buffer    = ReplayBuffer(online_cfg.buffer_capacity, (stack_size, img_size, img_size), action_dim)
    stack     = FrameStack(stack_size, img_size)
    extractor = LiveRewardExtractor(weights=online_cfg.reward_weights, max_reward=online_cfg.max_reward)
    executor  = ActionExecutor(
        output_columns=output_columns,
        action_threshold=online_cfg.action_threshold,
        suppress_mouse_movement=online_cfg.suppress_mouse_movement,
        blacklisted_keys=online_cfg.blacklisted_keys,
    )
    capture   = _make_capture_fn(online_cfg)
    optimizer = torch.optim.Adam(online_model.parameters(), lr=exp_cfg.learning_rate)
    output_dir = Path(online_cfg.checkpoint_dir)

    if args.dry_run:
        logger.info("Dry-run mode: keypresses suppressed.")

    # ── F8 toggle listener ────────────────────────────────────────────────────
    training_active = [True]

    def _on_press(key):
        if key == _TOGGLE_KEY:
            training_active[0] = not training_active[0]
            if training_active[0]:
                logger.info("RESUMED")
            else:
                executor.release_all()
                stack.reset()
                extractor.reset()
                logger.info("PAUSED — all keys released")

    _listener = _pynput_kb.Listener(on_press=_on_press)
    _listener.start()

    # ── Epsilon schedule ──────────────────────────────────────────────────────
    def _epsilon(step: int) -> float:
        frac = min(step / max(online_cfg.epsilon_decay_steps, 1), 1.0)
        return online_cfg.epsilon_start + frac * (online_cfg.epsilon_end - online_cfg.epsilon_start)

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    _stop = [False]
    def _handle_signal(sig, frame):
        logger.info("Interrupt received — will stop after this step.")
        _stop[0] = True
    signal.signal(signal.SIGINT, _handle_signal)

    # ── Training loop ─────────────────────────────────────────────────────────
    step = 0
    n_updates = 0
    prev_state:  np.ndarray | None = None
    prev_action: np.ndarray | None = None
    losses: list[float] = []
    frame_interval = 1.0 / online_cfg.capture_fps

    logger.info("Online training started. F8 to pause/resume. Ctrl+C to stop.")

    while not _stop[0]:
        if args.max_steps is not None and step >= args.max_steps:
            logger.info(f"Reached --max-steps {args.max_steps}.")
            break

        if not training_active[0]:
            # Reset transition state so we don't bridge across a pause
            prev_state  = None
            prev_action = None
            time.sleep(frame_interval)
            continue

        t0 = time.monotonic()

        # ── Capture & reward ──────────────────────────────────────────────
        pil_frame = capture()
        stack.push(pil_frame)
        reward, _labels = extractor.extract(pil_frame)

        if not stack.is_ready():
            time.sleep(max(0.0, frame_interval - (time.monotonic() - t0)))
            continue

        current_state = stack.get_stack()  # (S, H, W) float32

        # ── Store previous transition ─────────────────────────────────────
        # Reward observed entering current_state is credited to prev transition.
        if prev_state is not None:
            buffer.push(prev_state, prev_action, reward, current_state, done=False)

        # ── Action selection (epsilon-greedy) ─────────────────────────────
        eps = _epsilon(step)
        if np.random.random() < eps:
            q_values = np.random.randn(action_dim).astype(np.float32)
        else:
            with torch.no_grad():
                state_t = stack.get_stack_tensor(device)  # (1, S, H, W)
                q_values = online_model(state_t).squeeze(0).cpu().numpy()

        if not args.dry_run:
            action_vec = executor.execute(q_values)
        else:
            action_vec = (q_values > online_cfg.action_threshold).astype(np.float32)

        prev_state  = current_state
        prev_action = action_vec

        # ── Gradient update ───────────────────────────────────────────────
        if (len(buffer) >= online_cfg.min_buffer_size and
                step % online_cfg.update_every_n_steps == 0):

            states, actions, rewards_b, next_states, dones = buffer.sample(
                exp_cfg.batch_size, device
            )

            with torch.no_grad():
                q_next      = target_model(next_states)       # (B, action_dim)
                max_q_next  = q_next.max(dim=1).values        # (B,)
                target_vals = rewards_b + exp_cfg.gamma * max_q_next * (1.0 - dones)

            online_model.train()
            q_pred = online_model(states)
            loss = _online_q_loss(q_pred, actions, target_vals,
                                  exp_cfg.l1_inactive_weight, space_idx)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            online_model.eval()

            losses.append(loss.item())
            n_updates += 1

        # ── Logging ───────────────────────────────────────────────────────
        if step % args.log_every == 0:
            recent = float(np.mean(losses[-args.log_every:])) if losses else float("nan")
            logger.info(
                f"step={step:6d}  eps={eps:.3f}  buffer={len(buffer):5d}"
                f"  loss={recent:.4f}  reward={reward:.3f}  updates={n_updates}"
            )

        # ── Hard target network copy ──────────────────────────────────────
        if step > 0 and step % online_cfg.target_update_every_n_steps == 0:
            target_model.load_state_dict(online_model.state_dict())
            logger.info(f"step={step}: target network synced")

        # ── Periodic checkpoint ───────────────────────────────────────────
        if step > 0 and step % online_cfg.checkpoint_every_n_steps == 0:
            _save_checkpoint(online_model, optimizer, exp_cfg, step, losses, output_dir)

        step += 1
        time.sleep(max(0.0, frame_interval - (time.monotonic() - t0)))

    # ── Cleanup ───────────────────────────────────────────────────────────────
    logger.info("Shutting down...")
    _listener.stop()
    executor.release_all()
    extractor.shutdown()
    _save_checkpoint(online_model, optimizer, exp_cfg, step, losses, output_dir)
    _save_final_model(online_model, exp_cfg, output_dir)
    logger.info(f"Done. steps={step}  gradient_updates={n_updates}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online DQN fine-tuning for RL Resistance MM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to pretrained .pt checkpoint.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after N steps (default: run until Ctrl+C).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Capture + reward + buffer + gradients run, but no keypresses sent.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    parser.add_argument(
        "--log-every",
        type=int
        default=100,
        help="Log training loss every X epochs.",
    )
    main_args = parser.parse_args()
    run(main_args)


if __name__ == "__main__":
    main()
