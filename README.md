# Classroom Door Attendance System (MTCNN + FaceNet)

A door-mounted face recognition attendance system for classroom use. A student
walks up to the door camera, the system confirms their identity over ~2.5 s
of consistent recognition, logs attendance to a daily CSV, and shows a
"Welcome" card overlay. Back-to-back entries are supported, and each student
is on a 60-minute cooldown after being marked.

---

## 1. Project Overview

| Component | Role |
|---|---|
| **MTCNN** | Detects faces in each camera frame, outputs a 160x160 aligned crop |
| **InceptionResnetV1 (FaceNet, VGGFace2)** | Converts each crop to a 512-D embedding |
| **Cosine similarity** | Matches the live embedding against enrolled students |
| **Per-student confirmation timer** | Prevents false positives from single flickering frames |
| **Per-student 60-min cooldown** | Prevents double logging when a student walks in/out |
| **Daily CSV log** | `attendance_logs/attendance_YYYY-MM-DD.csv` |

The camera is expected to face the **door entry point**, seeing one student
at a time. If two faces ever appear, only the **largest** (most prominent,
i.e. closest to the door) is processed.

```
attendance_system/
├── main.py
├── enroll.py
├── attendance_logger.py
├── face_utils.py
├── ui_overlay.py
├── recognition_state.py
├── database/
│   └── students.json
├── embeddings/
├── attendance_logs/
├── requirements.txt
└── README.md
```

---

## 2. Installation

Python 3.10+ is required.

### Windows / macOS (developer machines)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate.bat
# macOS:   source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Raspberry Pi 5 (ARM64, CPU-only)

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libatlas-base-dev libopenblas-dev libjpeg-dev

python3 -m venv .venv
source .venv/bin/activate.bat
pip install --upgrade pip

# Install CPU-only PyTorch built for ARM64 linux
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Then the rest
pip install -r requirements.txt
```

If `opencv-python` fails to build, try `pip install opencv-python-headless`
instead (useful for headless Pi setups; you still get cv2.imshow via an
attached HDMI display on Pi OS Desktop, but the headless build is smaller).

---

## 3. Enrolling Students

Run from inside the project folder:

```bash
python enroll.py
```

You will be prompted for:

- **Name** (e.g. "Ahmad Faris")
- **Matrix Number** (e.g. "A12345")

The webcam opens, a green box tracks the student's face, and 15 embeddings
are captured. They are averaged, L2-normalised, and saved as:

- `embeddings/<matrix_number>.npy`
- An entry is appended to `database/students.json`.

If the matrix number already exists, you'll be asked whether to overwrite.

---

## 4. Running the System

```bash
python main.py
```

Per-frame behaviour:

1. Grab frame, mirror horizontally.
2. MTCNN detects faces; the **largest** face is selected.
3. FaceNet produces an embedding; `identify_face()` finds the best match.
4. If **unknown**: red box.
5. If **known + on cooldown**: blue box + "Already Recorded".
6. If **known + not on cooldown**: yellow scanning box with progress bar.
   After ~2.5 s of consistent match -> attendance logged, green box shown,
   and a centred "ATTENDANCE RECORDED" card appears for 3 s.

Press **`q`** in the window to quit.

---

## 5. Attendance Logs

- Directory: `attendance_logs/`
- File pattern: `attendance_YYYY-MM-DD.csv`
- Columns: `Name, Matrix Number, Date, Time, Status`
- `Status` is always `Present`.
- Duplicate guard: a student already in today's CSV will not be added again.

---

## 6. Tuning Parameters

| Parameter | File | Default | Notes |
|---|---|---|---|
| `MATCH_THRESHOLD` | `main.py` | `0.82` | Cosine similarity floor. Raise to be stricter. |
| `CONFIRM_DURATION` | `main.py` | `2.5 s` | Seconds of consistent match before logging. |
| `COOLDOWN_MINUTES` | `main.py` | `60` | Minutes before the same student can be re-logged. |
| `MIN_DETECTION_PROB` | `face_utils.py` | `0.92` | MTCNN confidence floor. |
| `WELCOME_CARD_SECONDS` | `main.py` | `3.0 s` | How long the pop-up card stays on screen. |

For noisier lighting you may want `MATCH_THRESHOLD = 0.78` and
`CONFIRM_DURATION = 3.0`.

---

## 7. Raspberry Pi 5 Deployment Notes

- **Drop the resolution** to 320x240 for a noticeable FPS boost. In
  `main.py`, change:

  ```python
  FRAME_W = 320
  FRAME_H = 240
  ```

- **Camera source**: `cv2.VideoCapture(0)` works for USB webcams. For the
  official Pi Camera Module, use the `picamera2` library and feed frames
  into the loop instead:

  ```python
  from picamera2 import Picamera2
  picam = Picamera2()
  picam.configure(picam.create_preview_configuration(main={"size": (320, 240)}))
  picam.start()
  # replace `cap.read()` with:
  frame = picam.capture_array()
  ```

- **Performance expectation**: ~3–5 FPS with FaceNet on the Pi 5 CPU. That
  is perfectly adequate for a door-entry scenario, where a student lingers
  in frame for 2–3 seconds.

- Set the Pi to performance CPU governor for steadier latency:
  ```bash
  sudo cpufreq-set -g performance
  ```

- Keep the Pi well-ventilated; throttling under sustained load will drop
  FPS further.

---

## 8. Troubleshooting

**"Could not open camera index 0"**
Check `ls /dev/video*` on Linux. Try `CAMERA_INDEX = 1` in `main.py`.

**"No enrolled students found"**
Run `python enroll.py` at least once. Check `database/students.json` and the
`embeddings/` folder.

**Recognition never hits "confirmed"**
- Lighting at the door may be too dim/backlit. Add a small warm LED panel.
- The cosine threshold may be too strict — try `MATCH_THRESHOLD = 0.78`.
- The student may not yet be enrolled, or they were enrolled under very
  different lighting. Re-enrol them.

**Everyone is recognised as the same student**
Your `embeddings/` folder contains stale or swapped `.npy` files. Delete
the offending files and re-enrol.

**First frame is very slow**
Model weights are downloaded the first time FaceNet runs. Subsequent runs
load from the local cache.

**PyTorch install fails on Pi 5**
Use the official CPU wheel index:
`pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu`

**Camera feed is laggy on the Pi**
Drop to 320x240, close other GUI apps, and ensure a 5V/5A supply.
