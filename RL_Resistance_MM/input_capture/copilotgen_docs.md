# RL_Resistance_MM — `input_capture` module

This document explains each file in the `input_capture` package, how to run the capture system, and how to troubleshoot common errors.

## Files

- `audio_capture.py`:

  - Purpose: Continuously captures system audio into a rolling buffer and exposes the most recent buffer as a base64-encoded WAV payload.
  - Key class: `AudioCapture` — constructor args: `sample_rate`, `channels`, `buffer_seconds`.
  - Main methods:
    - `start()` — opens a `sounddevice.InputStream` and fills a circular NumPy buffer.
    - `stop()` — stops and closes the stream.
    - `get_audio_buffer(timestamp)` — returns an `AudioBuffer` (from `schemas`) containing WAV data encoded as base64. (Note: the code converts to int16 WAV; Opus encoding is mentioned but not implemented.)
  - Notes/Troubleshooting:
    - Requires `sounddevice` and `soundfile` and a working PortAudio backend. On Windows, the PortAudio dependency is provided by `sounddevice` but you may need audio drivers or exclusive access settings.
    - Common error: `PortAudioError` or `OSError` when opening the input stream — try a different `sample_rate`, ensure your microphone/loopback device is available, and close other apps using the audio device.

- `input_capture.py`:

  - Purpose: Polls mouse (and nominally keyboard) state and queues input events for later synchronization with frames.
  - Key class: `InputCapture` — constructor arg: `poll_rate` (Hz).
  - Main methods:
    - `start()` / `stop()` — manage a capture thread.
    - `_capture_loop()` — polls at `poll_rate` using `_poll_inputs()`.
    - `_poll_inputs(timestamp)` — collects mouse position and button states using `pyautogui` and `mouse`; puts `MouseEvent` objects into an internal queue when movement or button changes occur.
    - `get_events_since(since_timestamp)` — drains queued events newer than a timestamp.
    - `get_current_state()` — snapshot of tracked states.
  - Notes/Troubleshooting:
    - The code imports `keyboard` but keyboard handling appears not implemented (keyboard state tracking/stubs are present in `schemas` but not captured here). If you need keyboard events, this module will need additional implementation using the `keyboard` library's hooks.
    - On Windows, the `keyboard` and `mouse` packages may require running the Python process as Administrator for global hooks.
    - `pyautogui.position()` and `mouse.is_pressed()` are used; ensure the `mouse` package is installed (pip package `mouse`).

- `main.py`:

  - Purpose: `GameCaptureController` coordinates `InputCapture`, `AudioCapture`, and `ScreenCapture`, collects synchronized frame/audio/input data, and writes JSON + image/audio files to disk.
  - Key class: `GameCaptureController` — accepts a `CaptureConfig` (from `schemas`).
  - Main workflow:
    - Initializes capture modules and output directories.
    - `start()` — saves config, starts each capture module, and starts a background save worker thread.
    - `capture_loop()` — main loop that fetches latest screen frames, gets audio buffer and input events synchronized to the frame timestamp, assembles a `frame_data` dict, and enqueues it for the save thread.
    - `_save_worker()` — dequeues frame data and writes JSON (`frames/raw/frame_XXXXXX.json`) and image JPG (`screens/frame_XXXXXX.jpg`).
    - `stop()` — stops modules, signals save thread, and saves remaining buffered data.
  - Example usage: the `main()` function at the bottom shows how to create a `CaptureConfig` and run the capture; pressing Ctrl+C will stop capture gracefully.
  - Notes/Troubleshooting:
    - Output/logs: capture creates an `output_dir` per session and a `capture.log` file — check `capture.log` for detailed errors.
    - File I/O errors: make sure the process has write permissions to the configured `output_dir`.
    - Performance: high FPS (e.g., 60) plus saving large images and audio can be I/O and CPU intensive — ensure enough CPU and disk throughput.

- `schemas.py`:

  - Purpose: Shared dataclasses, enums, and helpers for frame/audio/input representations and the `CaptureConfig`.
  - Key dataclasses: `CaptureConfig`, `KeyEvent`, `MouseEvent`, `AudioBuffer`, `ScreenFrame`.
  - Utilities: `ScreenFrame.to_jpeg_base64()` and `crop_regions()` — `to_jpeg_base64()` uses PIL to encode images.
  - Notes:
    - `CaptureConfig` has defaults; if `session_id` is not provided, it will be auto-generated.

- `screen_capture.py`:

  - Purpose: Captures screen screenshots at a target FPS using `pyautogui` and packages them as `ScreenFrame` objects.
  - Key class: `ScreenCapture` — constructor args: `target_fps`, `resolution`.
  - Main methods:
    - `start()` / `stop()` — manage a capture thread.
    - `_capture_loop()` — attempts to maintain the configured FPS and puts `ScreenFrame` objects into an internal queue.
    - `_capture_single_frame(timestamp)` — grabs a screenshot (`pyautogui.screenshot()`), converts/resizes via OpenCV, wraps into a `ScreenFrame`, and crops predefined regions.
    - `get_latest_frame()` — fetch the next frame from the queue.
  - Notes/Troubleshooting:
    - `pyautogui.screenshot()` can be slow; headless environments or remote sessions may behave differently.
    - OpenCV is used for resizing and color conversion — ensure `opencv-python` is installed.
    - Multi-monitor setups: `pyautogui.screenshot()` behavior depends on display setup; check captured image size and adjust `target_resolution` as needed.


## How to run

1. Install dependencies. From the workspace root, install the packages used by the module (some may already be in your global env or in the project `requirements.txt`). Example:

```bash
pip install numpy sounddevice soundfile pyautogui opencv-python pillow keyboard mouse psutil
```

If you prefer, add these to your `requirements.txt`/virtual environment.

2. Run the example `main.py` from the `input_capture` folder:

```bash
cd RL_Resistance_MM/input_capture
python main.py
```

This runs the `main()` example which creates a `CaptureConfig`, starts capture, prints session info, and runs `capture_loop()` until you press Ctrl+C.

Notes:

- On Windows you may need to run the terminal as Administrator for `keyboard`/`mouse` to capture global input.
- If you get an error opening audio input, verify your audio devices and try different `sample_rate` or set the default input device in Windows Sound settings.

## Configuring the capture

- Edit the `CaptureConfig` creation in `main.py` (or construct a `CaptureConfig` yourself) to change:

  - `screen_resolution` — desired capture resolution
  - `capture_fps` — how many frames per second to attempt
  - `audio_sample_rate`, `audio_channels`, `audio_buffer_seconds`
  - `output_dir` and `session_id`

- The controller saves a `config.json` to the session folder for later reference.

## Troubleshooting tips (common issues)

- Permission issues capturing keyboard/mouse:

  - Windows: run Python as Administrator.
  - The `keyboard` and `mouse` packages often require elevated permissions to install global hooks.

- `sounddevice` / audio errors:

  - `PortAudioError` or `OSError` when starting the input stream: check that no other program exclusively holds the device; try different `sample_rate` values (e.g., 44100), and ensure correct input device is set. For advanced diagnosis, print `sd.query_devices()`.

- `pyautogui.screenshot()` errors or blank images:

  - On headless or remote desktop environments, screenshots may produce unexpected results. Test `pyautogui.screenshot().size` in a REPL.
  - Slow screenshots reduce achievable FPS; lower `capture_fps` if the system can't keep up.

- OpenCV / PIL issues:

  - Install `opencv-python` and `Pillow` (PIL) via pip.

- Disk I/O bottleneck:

  - The save worker writes JSON and JPG files; slow disks will make the queue fill. Reduce capture FPS, lower resolution, or increase disk throughput.

- Missing keyboard capture implementation:
  - `input_capture.py` currently captures mouse events but does not implement keyboard polling/handlers. If you require key events, add keyboard hooks (e.g., `keyboard.hook()` or `keyboard.is_pressed()` polling) and enqueue `KeyEvent` objects similar to `MouseEvent`.

## Where to look for logs and output

- Each capture session creates an output folder: `output_dir/<session_id>`.
- `capture.log` inside the session folder contains logged info and errors.
- Frames JSON: `frames/raw/frame_000000.json` and corresponding images in `screens/`.
- Audio files: `audio/` (the current code stores audio as base64 WAV inside the frame JSON; adjust `AudioCapture` if you want separate `.wav` files).


