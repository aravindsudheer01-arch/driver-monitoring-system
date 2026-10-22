# Driver Monitoring System — Phase 3

An **educational** real-time computer vision prototype that combines an existing fatigue pipeline with coarse 3D head-pose estimation.

**Not validated, certified, or intended for driving safety decisions. Never test while driving.** Head tilt is not evidence that a person is asleep; this prototype explicitly labels the fusion `POSSIBLE HEAD SLUMP` rather than confirming sleep.

## What works in the current code

- MediaPipe Face Landmarker estimates facial landmarks from webcam frames.
- OpenCV/NumPy calculate eye aspect ratio (EAR) and mouth opening ratio.
- Independent short blink detection and sustained eye-closure detection, yawning counter, PERCLOS-like time-weighted rolling statistic.
- macOS repeating two-tone warning for prolonged eye closure (Windows supported by `winsound`).
- **New Phase 3:** approximate pitch/yaw/roll using six landmarks and OpenCV `solvePnP`.
- **New Phase 3:** press `C` to set a neutral head orientation; compute **relative** angles.
- **New Phase 3:** turn/tilt timers; brief head movements are not classified as sustained.
- **New Phase 3:** possible slump label when a sustained tilt/nod **coincides with an existing eye-closure warning**. Neither measurement is proof of sleep.
- Independent camera-free unit tests for blink/fatigue/pose logic and synthetic-image head-pose tests.

## Installation (Python 3.12; existing Intel Mac setup)

In VS Code terminal, from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_model.py
python -m unittest discover -s tests -v
python main.py --camera 1 --debug-pose
```

**If you already have this project from Phase 2.1, don't recreate `.venv` or re-download the model.** Copy in the update files following `README_PHASE3.md`.

Camera `1` was the index that worked on the test Intel Mac; use `--camera 0` or `2` on another system. If camera is denied: macOS Settings → Privacy & Security → Camera, allow VS Code/Terminal, then restart the app.

### Keys

| Key | Action |
| --- | --- |
| `C` | **Calibrate** neutral orientation while looking straight ahead; hold head upright, remain still |
| `M` | Mute/unmute repeating eye-closure alarm |
| `Q` / `Esc` | Quit |

Calibrate again if the webcam position changes or the reference is inappropriate. Allow 1–2 seconds of stable tracking before pressing `C`. You can adjust thresholds when launching:

```bash
python main.py --camera 1 --width 640 --height 480 --debug-pose
python main.py --camera 1 --yaw-threshold 35 --roll-threshold 30 --pitch-threshold 30
python main.py --camera 1 --max-reprojection-error 25 --debug-pose
```

**The `--max-reprojection-error` setting is a fitting-quality gate, not a promise of angle accuracy.** If pose never appears, test this setting and check camera angle, lighting and face visibility. Don't simply raise it to suppress all failure detection.

## Expected behavior (prototype heuristics)

| Observation | Visual indication | Eye-closure alarm |
| --- | --- | --- |
| Face neutral and open-eyed | `HEAD NEUTRAL` / `NORMAL` | Off |
| Brief look toward mirror / quick turn | `HEAD MOVING` | Off |
| Head yaw deviation >30° for >1.5s | `SUSTAINED HEAD TURN` | Off |
| Head roll or pitch deviation >25° for >1.0s | `SUSTAINED HEAD TILT` | Off if eyes open |
| Tilt held while eye-closure warning is active | `POSSIBLE HEAD SLUMP` | On (from eye closure) |
| Eyes closed for >1.5s | `DROWSINESS WARNING` | On |
| Eyes closed for >3s | `POSSIBLE MICROSLEEP` | On |
| Face or PnP pose lost | `NO FACE` / `POSE UNAVAILABLE` | Off unless valid eye-closure measurements still independently produce a warning |

Threshold values are starting **engineering heuristics**, not measured clinical cutoffs. The label `POSSIBLE MICROSLEEP` is similarly a prototype hypothesis based on eye-closure duration, not a confirmed diagnosis.

## How the code is organized

- `main.py`: webcam, model invocation, HUD, input handling and alarm orchestration.
- `metrics.py`: eye/mouth geometric measurements (Phase 1).
- `monitor.py`: frame-independent blink, yawn, eye-closure state and rolling PERCLOS-like logic (Phase 2/2.1).
- `alerts.py`: nonblocking audible alarm (Phase 2.1 audio update).
- `head_pose.py`: 2D facial landmarks → approximate 3D orientation using PnP, with reprojection filtering; draws 3D axes (Phase 3).
- `pose_monitor.py`: neutral calibration, angular differences, time thresholds and conservative fatigue fusion (Phase 3).
- `tests/`: automated behavior and synthetic geometry tests.
- `INTERVIEW_GUIDE.md`: explanation, math, assumptions and possible questions.

## Validation & limitations

The new PnP head-pose code is verified on **synthetic projections** and the state machine is tested with **simulated timestamps**. This verifies implementation behavior under those conditions; it does **not** establish real-world performance.

The webcam's intrinsics are only approximated: fx = fy = image width, principal point = image center, lens distortion ignored. The assumed six-point 3D face template is generic. The extracted angles are **rough estimates** that can drift with face shape, camera location, lighting and spectacles. The `C` key subtracts one neutral orientation to reduce constant camera/head offsets, but cannot correct all errors. MediaPipe may fail when face is heavily turned or in darkness. Since inference is ~12 FPS on the tested Mac, fast blinks can also be missed.

For a stronger portfolio project, run a labeled protocol: multiple participants/lighting conditions, 20–30 head turns/tilts per scenario, short mirror-like turns, sustained tilts with eyes open, eyes-closed events, false alarms over 10 minutes, tracking-loss episodes, FPS and inference latency. Log **precision, recall, false alarms per hour, timing delay**, and known test conditions. Avoid collecting identifiable face recordings without consent.

See [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md) for a detailed technical explanation and example interview answers.
