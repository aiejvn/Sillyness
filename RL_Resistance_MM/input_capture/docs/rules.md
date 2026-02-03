# Input Capture Module — Data Schema Reference

## Purpose

This module records raw gameplay data from a Windows desktop for an RL (Reinforcement Learning) project codenamed **RL_Resistance_MM**. It synchronizes three data streams — screen, audio, and player input — into per-frame JSON snapshots saved to disk.

---

## Session Directory Layout

A capture session produces the following on-disk structure:

```
<output_dir>/<session_id>/
├── config.json                # CaptureConfig used for this session
├── capture.log                # Runtime log
├── frames/
│   └── raw/
│       ├── frame_000000.json  # Per-frame JSON (see schema below)
│       ├── frame_000001.json
│       └── ...
└── screens/
    ├── frame_000000.jpg       # Standalone JPEG copy of each frame
    ├── frame_000001.jpg
    └── ...
```

The default `output_dir` is `./re_resistance_captures`.

---

## Capture Configuration (`config.json`)

Saved once per session. Fields match `CaptureConfig`:

| Field                  | Type   | Default        | Notes                                   |
|------------------------|--------|----------------|-----------------------------------------|
| `screen_resolution`    | [int, int] | [1920, 1080] | Target resolution; frames are resized to this |
| `capture_fps`          | int    | 60             | Target screen-capture framerate         |
| `audio_sample_rate`    | int    | 48000          | Audio Hz                                |
| `audio_channels`       | int    | 2              | Stereo (may be downgraded to 1 if device only supports mono) |
| `audio_buffer_seconds` | float  | 1.0            | Rolling audio window attached to each frame |
| `input_poll_rate`      | int    | 1000           | Mouse/keyboard polling Hz               |
| `output_dir`           | str    | `./capture_data` |                                        |
| `session_id`           | str    | auto-generated | Format: `session_YYYYMMDD_HHMMSS`       |
| `compression_quality`  | int    | 85             | JPEG quality 0-100                      |

---

## Per-Frame JSON Schema (`frames/raw/frame_NNNNNN.json`)

Each file is a self-contained snapshot tying together visual, audio, and input data at one screen-capture tick.

```jsonc
{
  // === Identity ===
  "frame_id":              "session_003_frame_000042",   // "<session_id>_frame_<6-digit zero-padded>"
  "session_id":            "session_003",
  "monotonic_timestamp":   0.7123,                       // Seconds since capture started (perf_counter based)
  "frame_number":          42,                           // 0-indexed sequential integer

  // === Visual Data ===
  "visual_raw": {
    "full_screen": {
      "format":        "jpeg_85",       // "jpeg_<quality>"
      "data_base64":   "<base64>",      // Full-screen JPEG encoded as base64 string
      "width":         1920,            // Pixels (matches config resolution)
      "height":        1080
    },
    "regions_of_interest": {}           // Reserved; cropped sub-images would go here (see Region Definitions below)
  },

  // === Audio Data ===
  "audio_raw": {
    "format":            "wav",                // WAV-encoded PCM (int16). Production may switch to "opus"
    "duration_ms":       1000.0,               // Length of audio window in ms (matches audio_buffer_seconds * 1000)
    "data_base64":       "<base64>",           // base64-encoded WAV file bytes
    "timestamp_offset":  -0.002                // Audio timestamp minus frame timestamp (seconds); negative means audio is slightly earlier
  },

  // === Input Data ===
  "input_raw": {
    "keyboard": {
      "keys_pressed":      ["w", "shift"],     // Keys held at time of this frame (InputState PRESS or HOLD)
      "key_state_changes": []                  // Currently always empty in code (keyboard events go through hook but are not serialized into per-frame mouse-style dicts)
    },
    "mouse": {
      "position":          [960, 540],         // [x, y] pixel coords at frame time
      "movement_events": [                     // Mouse moves/clicks between this frame and the previous
        {
          "type":             "mouse_move",    // One of: mouse_move, mouse_press_left, mouse_press_middle, mouse_press_right, mouse_release_left, mouse_release_middle, mouse_release_right
          "x":                961,
          "y":                541,
          "dx":               1,               // Delta from previous poll
          "dy":               1,
          "buttons":          [true, false, false],  // [left, middle, right] pressed state at event time
          "timestamp_offset": -0.005           // Event timestamp minus frame timestamp (seconds)
        }
      ],
      "buttons_current":   [false, false, false]  // [left, middle, right] at frame time
    }
  },

  // === Sync Marks ===
  "synchronization_marks": {
    "frame_number":          42,
    "system_time":           1706900000.123,    // Unix epoch (time.time())
    "audio_peak_frequency":  0                  // Placeholder (always 0 for now)
  }
}
```

---

## Key Data Types (from `schemas.py`)

### InputState (Enum)
- `"press"` — key/button just pressed
- `"release"` — key/button released
- `"hold"` — key/button sustained (tracked internally but collapsed to `press` in output)

### MouseEvent
| Field     | Type       | Notes |
|-----------|------------|-------|
| `x`, `y`  | int       | Absolute screen position |
| `dx`, `dy` | int      | Delta since last poll |
| `buttons` | (bool, bool, bool) | Left, Middle, Right |
| `wheel`   | (int, int) | Horizontal, Vertical scroll (defined but not captured in current code) |
| `timestamp` | float   | `time.perf_counter()` value |

### ScreenFrame
| Field         | Type          | Notes |
|---------------|---------------|-------|
| `image`       | np.ndarray    | RGB uint8, shape (H, W, 3) |
| `timestamp`   | float         | perf_counter |
| `frame_number`| int           | 0-indexed |
| `regions`     | dict or None  | Named sub-crops (see below) |

### AudioBuffer
| Field        | Type       | Notes |
|--------------|------------|-------|
| `data`       | str        | base64-encoded WAV bytes |
| `sample_rate`| int        | Hz |
| `channels`   | int        | 1 or 2 |
| `timestamp`  | float      | Aligned to frame timestamp |
| `duration`   | float      | Seconds |

---

## Screen Region Definitions

Hardcoded in `ScreenCapture` (TODO: these need verification):

| Region Name    | Coords (x1, y1, x2, y2) | Purpose |
|----------------|--------------------------|---------|
| `ui_top_bar`   | (0, 0, 1920, 60)        | Top HUD bar |
| `camera_view`  | (300, 100, 1620, 900)   | Main game viewport |
| `ability_bar`  | (0, 980, 1920, 1080)    | Bottom ability/action bar |

These are cropped from each frame into `ScreenFrame.regions` but are **not** currently serialized into the per-frame JSON (the `regions_of_interest` field is always `{}`).

---

## Timing Model

- All timestamps are `time.perf_counter()` values (monotonic, high-resolution).
- `monotonic_timestamp` in JSON = `frame.timestamp - capture_start_time` (seconds since session start).
- `timestamp_offset` fields express the difference between a sub-event and its parent frame (negative = happened before the frame tick).
- `system_time` in `synchronization_marks` is wall-clock (`time.time()`) for correlating with external logs.

---

## Audio Details

- Captured via **PyAudio** using WASAPI shared mode on Windows.
- Rolling circular buffer of `buffer_seconds` (default 1s) at `sample_rate` Hz.
- Each frame gets the **entire rolling buffer** as a WAV-encoded base64 blob — meaning consecutive frames have heavily overlapping audio.
- Internal float32 samples are converted to int16 for WAV encoding.
- Debug mode (`debug=True` by default) accumulates all raw audio frames and saves a single amplified WAV (`debug_audio_capture.wav`) on stop. The amplification factor is 150x, clipped to [-1.0, 1.0].

---

## Input Capture Details

- **Keyboard**: Global hook via `keyboard` module. Events are `key_press` / `key_release` with key name strings (e.g. `"w"`, `"shift"`, `"space"`). The `key_state_changes` array in per-frame JSON is currently always empty — only `keys_pressed` (snapshot) is populated.
- **Mouse**: Polled at `input_poll_rate` Hz via `pyautogui.position()` and `mouse.is_pressed()`. Movement deltas and button state changes are captured as discrete events between frames.
- Wheel/scroll events are defined in the schema but **not captured** in the current implementation.

---

## Known Gaps / TODOs for Downstream Consumers

1. `key_state_changes` is always `[]` — keyboard event history between frames is lost; only the snapshot (`keys_pressed`) is available.
2. `regions_of_interest` is always `{}` — region crops exist in memory but aren't serialized.
3. `audio_peak_frequency` is always `0` — placeholder for future audio fingerprinting.
4. Mouse wheel/scroll is not captured despite being in the schema.
5. Screen region coordinates are hardcoded and flagged as unverified.
6. Audio buffers overlap heavily between consecutive frames (each contains the full rolling window).
