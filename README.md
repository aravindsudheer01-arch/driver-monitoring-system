# Driver Monitoring System

A real-time, webcam-based **driver fatigue and head-pose monitoring prototype** built with Python, OpenCV, MediaPipe, and NumPy. It analyzes facial landmarks to estimate eye closure, blinking, mouth opening, and head orientation, then displays alerts for sustained conditions.

> **Educational prototype only.** This project is not a certified driver-monitoring or medical system and must not be relied on for driving decisions. Test it while stationary, not while driving. A head tilt, head turn, or long eye closure cannot by itself establish that someone is asleep.

## Features

- **Facial landmark tracking:** Uses the pretrained MediaPipe Face Landmarker on live webcam frames.
- **Eye monitoring:** Calculates eye aspect ratio (EAR), counts detected blinks, and monitors prolonged eye closure.
- **Fatigue-related warnings:** Displays drowsiness and possible microsleep labels when eye closure persists beyond configured durations.
- **Mouth monitoring:** Calculates a mouth-opening ratio and counts sustained wide-opening events as *possible yawns*.
- **PERCLOS-like statistic:** Tracks the proportion of valid observed time classified as eye-closed in a rolling window. This is a simplified proxy, not a validated clinical or automotive PERCLOS measurement.
- **Audible alerts:** Plays a repeating warning for sustained eye closure, with an option to mute.
- **Head-pose estimation:** Uses OpenCV `solvePnP` and selected facial landmarks to estimate approximate yaw (left/right rotation), pitch (up/down rotation), and roll (sideways tilt).
- **Neutral calibration and temporal filtering:** The `C` key sets a neutral head reference; brief movements are distinguished from sustained deviations.
- **Combined observations:** A sustained tilt or nod alongside an eye-closure warning is labeled **POSSIBLE HEAD SLUMP**, *not* confirmed sleep.
- **Automated tests:** Tests cover geometric calculations, monitoring state changes, alerts, and head-pose behavior on synthetic inputs.

## Technology stack

- Python 3.12 (tested on an Intel MacBook)
- OpenCV and NumPy for image processing and geometric calculations
- MediaPipe Face Landmarker (pretrained model) for facial landmark estimation
- Python `unittest` for automated tests

The project **uses a pretrained model**; it does not train its own facial landmark detector.

## Getting started

### Requirements

- Python 3.12 and a working webcam
- Camera access permission for the terminal or VS Code
- Internet access for the initial dependency and model downloads

### Setup on macOS / Linux

Open this repository in VS Code, then run from its root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_model.py
python -m unittest discover -s tests -v
python main.py --camera 0
```

If camera `0` does not work, try `--camera 1`. On the tested MacBook, camera `1` worked. You may need to grant camera access in **System Settings > Privacy & Security > Camera** and restart VS Code.

### Setup on Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_model.py
python -m unittest discover -s tests -v
python main.py --camera 0
```

The webcam index depends on the computer. Run `python main.py --help` for supported options and adjustable thresholds.

## Controls

| Key | Action |
| --- | --- |
| `C` | Calibrate a neutral head pose while facing the camera upright |
| `M` | Mute or unmute the audible warning |
| `Q` or `Esc` | Exit the application |

After startup, sit upright facing the webcam, allow tracking to stabilize, and press **C**. Recalibrate if the camera position changes.

## Monitoring behavior

The conditions below are **configurable prototype heuristics**, not validated driver-safety thresholds.

| Observed condition | Displayed indication | Audible eye-closure alarm |
| --- | --- | --- |
| Neutral head, eyes open | `NORMAL` / `HEAD NEUTRAL` | Off |
| Brief head movement | `HEAD MOVING` | Off |
| Sustained sideways head turn | `SUSTAINED HEAD TURN` | Off unless prolonged eye closure is also detected |
| Sustained head tilt or nod | `SUSTAINED HEAD TILT` | Off unless prolonged eye closure is also detected |
| Sustained head tilt/nod with eye-closure warning | `POSSIBLE HEAD SLUMP` | On because of eye closure |
| Eyes classified closed for more than ~1.5 seconds | `DROWSINESS WARNING` | On |
| Eyes classified closed for more than ~3 seconds | `POSSIBLE MICROSLEEP` | On |
| Face or pose lost | `NO FACE` / `POSE UNAVAILABLE` | Not automatically treated as drowsiness |

A driver may turn their head to check a mirror or tilt it while awake. The system therefore does **not** use sideways movement alone to confirm sleep.

## How it works

1. **Capture:** OpenCV captures frames from the selected camera.
2. **Landmarks:** MediaPipe estimates facial landmark locations in each detected face.
3. **Facial measurements:** Landmark coordinates are used to compute EAR and a mouth-opening ratio.
4. **Temporal logic:** A monitoring module tracks blink events, sustained eye closure, possible yawns, and a rolling eye-closure statistic over time.
5. **Head orientation:** Six 2D facial landmarks are matched to a generic 3D face template. OpenCV `solvePnP` estimates approximate orientation. Pressing `C` establishes a neutral reference for relative angles.
6. **Warnings:** Separate state logic combines duration and pose information to display labels; an independent alarm module handles sound during sustained eye-closure warnings.

### Repository structure

```text
driver-monitoring-system/
├── main.py             # Webcam loop and display
├── metrics.py          # Eye and mouth measurements
├── monitor.py          # Blink, fatigue, and mouth-event logic
├── head_pose.py        # Approximate 3D orientation estimation
├── pose_monitor.py     # Head-pose calibration and timed labels
├── alerts.py           # Audible warning
├── download_model.py   # Pretrained model downloader
├── requirements.txt
├── tests/              # Automated tests
└── README.md
```

The downloaded MediaPipe model and local Python environment are not stored in the repository.

## Testing and limitations

Automated tests check the implemented calculations and state transitions with synthetic geometry and simulated timestamps. Live webcam checks on an Intel MacBook showed functioning eye/mouth tracking, head-pose labels, yawning indications, and eye-closure audio warnings. These checks **do not establish accuracy, sensitivity, specificity, or reliability in driving conditions**.

Known limitations include:

- The camera intrinsic parameters are approximated (focal lengths based on image width, principal point at image center, lens distortion ignored), and the 3D face template is generic. Reported head angles are rough estimates rather than calibrated ground truth.
- Lighting, glasses, head orientation, occlusion, camera placement, and differences in facial geometry can affect landmark estimates.
- The observed application frame rate on the test machine was around 12 FPS; fast blinks may be missed.
- Visible-light webcams can perform poorly in very low light or darkness.
- `POSSIBLE MICROSLEEP` and `POSSIBLE HEAD SLUMP` are heuristic labels; they cannot confirm sleep or medical fatigue.

A future evaluation should use labeled test sessions across participants, camera positions, and lighting conditions, reporting blink-event precision/recall, missed detections, false alarms per hour, detection delay, FPS, and inference latency. Do not record or publish identifiable face videos without consent.

## Future improvements

- Evaluate blink thresholds and head-pose alerts on labeled scenarios.
- Improve low-light robustness and handle tracking loss more explicitly.
- Measure frame rate, latency, and false-alarm rates on multiple devices.
- Add a short demonstration video and a reproducible evaluation summary.

## Model attribution

Facial landmarks are estimated with Google's pretrained [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker). See `download_model.py` for the model download used by this project.
