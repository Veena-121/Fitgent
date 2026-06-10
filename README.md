# Fitgent — Computer Vision Personal Trainer (MVP)

> ⚠️ This is an MVP / work in progress. The full version with LLM coaching, session history, and multi-angle bilateral tracking is under active development.

Real-time fitness agent that uses **MediaPipe Pose** to detect body landmarks via webcam, count exercise reps, and analyse form with posture correction feedback.

**Stack:** Python · OpenCV · MediaPipe · NumPy

---

## What it does (MVP scope)

- Detects 33 body landmarks in real time via webcam using MediaPipe Pose
- Counts reps by tracking joint angle transitions (up ↔ down phases)
- Calculates joint angles (knee, hip, elbow, shoulder) and compares against reference thresholds
- Displays posture correction feedback on-screen (e.g. "Go deeper", "Stop swinging")
- Supports 3 exercises: **Squat**, **Push-up**, **Bicep Curl**

---

## Demo

```
python fitagent.py --exercise squat
```

Stand ~2m from your webcam so your full body is visible.

---

## Quick Start

**Python 3.9 – 3.11 required** (MediaPipe 0.10 does not support 3.12 yet)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/fitagent.git
cd fitagent

# 2. Virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run
python fitagent.py                    # squat (default)
python fitagent.py --exercise pushup
python fitagent.py --exercise curl
```

**Controls while running:**

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset rep counter |

---

## How it works

### Pose Detection
MediaPipe Pose returns 33 body landmarks normalised to `[0, 1]`. FitAgent converts these to pixel coordinates and computes joint angles using the **dot-product formula**:

```
angle(A, B, C) = arccos( (BA · BC) / (|BA| × |BC|) )
```

### Rep Counting
Each exercise has two angle thresholds:

- **`down_angle`** — crossing this enters the "down" phase
- **`up_angle`** — crossing back completes one rep

A rep is counted on the `down → up` transition only, preventing double-counting on partial reps.

### Posture Correction
Each exercise compares live joint angles against reference thresholds and returns a feedback message with a severity level (`good` / `warn` / `error`), shown as a colour-coded banner at the bottom of the frame.

| Exercise | Checks |
|----------|--------|
| Squat    | Knee depth, hip angle, back lean |
| Push-up  | Elbow depth, shoulder flare, body alignment |
| Bicep Curl | Elbow extension, shoulder swing (body english) |

---

## Planned (full version)

- [ ] LLM coaching via Claude API — natural language cues from joint angle data
- [ ] Bilateral tracking — compare left vs right side symmetry
- [ ] Session history — sets, reps, form score over time
- [ ] More exercises — lunge, plank, deadlift, overhead press
- [ ] Streamlit web UI — browser-based version, no install required

---

## Project structure

```
fitagent/
├── fitgent.py     
└── requirements.txt
```


