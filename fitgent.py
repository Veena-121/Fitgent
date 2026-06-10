import cv2
import mediapipe as mp
import numpy as np
import argparse


mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
PL         = mp_pose.PoseLandmark   
GREEN = (29, 158, 117)
AMBER = (0,  165, 255)
RED   = (48,  90, 216)
WHITE = (255, 255, 255)
DARK  = (20,  20,  20)

def calc_angle(a, b, c):
    """Angle (degrees) at vertex b formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cosine  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def get_point(lm, idx, w, h):
    p = lm[idx]
    return (p.x * w, p.y * h)

EXERCISES = {
    "squat": {
        "name":        "SQUAT",
        "down_angle":  100,   
        "up_angle":    160,   
        "joint":       "knee",
    },
    "pushup": {
        "name":        "PUSH-UP",
        "down_angle":  90,
        "up_angle":    155,
        "joint":       "elbow",
    },
    "curl": {
        "name":        "BICEP CURL",
        "down_angle":  155,
        "up_angle":    55,
        "joint":       "elbow",
        "invert":      True,   
    },
}


def get_joint_angle(lm, joint, w, h):
    """Return the angle for the requested joint (left side)."""
    if joint == "knee":
        return calc_angle(
            get_point(lm, PL.LEFT_HIP,      w, h),
            get_point(lm, PL.LEFT_KNEE,     w, h),
            get_point(lm, PL.LEFT_ANKLE,    w, h),
        )
    if joint == "elbow":
        return calc_angle(
            get_point(lm, PL.LEFT_SHOULDER, w, h),
            get_point(lm, PL.LEFT_ELBOW,    w, h),
            get_point(lm, PL.LEFT_WRIST,    w, h),
        )
    if joint == "hip":
        return calc_angle(
            get_point(lm, PL.LEFT_SHOULDER, w, h),
            get_point(lm, PL.LEFT_HIP,      w, h),
            get_point(lm, PL.LEFT_KNEE,     w, h),
        )
    return 0.0


def get_form_feedback(lm, exercise_key, w, h):
    """
    Rule-based form analysis: compares joint angles against
    reference thresholds and returns (message, level).
    level: 'good' | 'warn' | 'error'
    """
    knee  = get_joint_angle(lm, "knee",  w, h)
    elbow = get_joint_angle(lm, "elbow", w, h)
    hip   = get_joint_angle(lm, "hip",   w, h)

    shoulder = calc_angle(
        get_point(lm, PL.LEFT_ELBOW,    w, h),
        get_point(lm, PL.LEFT_SHOULDER, w, h),
        get_point(lm, PL.LEFT_HIP,      w, h),
    )

    if exercise_key == "squat":
        if knee > 168:
            return "Go deeper — aim for 90° knee bend", "warn"
        if knee < 55:
            return "Too deep — ease up slightly",        "warn"
        if hip < 45:
            return "Keep back straight — chest up!",     "error"
        return "Good squat form!",                       "good"

    if exercise_key == "pushup":
        if elbow > 160:
            return "Lower chest to ground — full ROM",   "warn"
        if shoulder > 70:
            return "Tuck elbows — don't flare wide",     "error"
        return "Good push-up form!",                     "good"

    if exercise_key == "curl":
        if shoulder > 40:
            return "Stop swinging — pin your upper arm", "error"
        if elbow > 165:
            return "Full extension at the bottom",       "warn"
        if elbow < 45:
            return "Squeeze at the top!",                "good"
        return "Controlled pace — 2s up, 2s down",       "good"

    return "Keep going!", "good"




def count_rep(angle, phase, ex):
    """
    Returns (new_phase, rep_counted).
    Handles both normal (up→down) and inverted (down→up) exercises.
    """
    rep_counted = False
    invert = ex.get("invert", False)

    if not invert:
        if phase == "up"   and angle < ex["down_angle"]:
            phase = "down"
        elif phase == "down" and angle > ex["up_angle"]:
            phase = "up"
            rep_counted = True
    else:
        if phase == "up"   and angle > ex["down_angle"]:
            phase = "down"
        elif phase == "down" and angle < ex["up_angle"]:
            phase = "up"
            rep_counted = True

    return phase, rep_counted


def draw_hud(frame, reps, phase, ex_name, feedback, fps):
    h, w = frame.shape[:2]

    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), DARK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, ex_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}  |  Q=quit  R=reset",
                (w - 280, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

   
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (10, 52), (120, 140), DARK, -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "REPS", (18, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, str(reps), (22, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, WHITE, 3, cv2.LINE_AA)

  
    phase_color = GREEN if phase == "up" else AMBER
    phase_txt   = "^ UP" if phase == "up" else "v DOWN"
    cv2.putText(frame, phase_txt, (12, 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1, cv2.LINE_AA)

   
    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (0, h - 40), (w, h), DARK, -1)
    cv2.addWeighted(overlay3, 0.72, frame, 0.28, 0, frame)

    msg, level = feedback
    dot_color  = GREEN if level == "good" else (AMBER if level == "warn" else RED)
    cv2.circle(frame, (14, h - 14), 6, dot_color, -1)
    cv2.putText(frame, msg, (28, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)


def draw_angle_label(frame, lm, joint_idx, angle, w, h):
    """Draw angle value next to a landmark."""
    p = lm[joint_idx]
    if p.visibility < 0.5:
        return
    x, y = int(p.x * w) - 55, int(p.y * h) + 10
    cv2.putText(frame, f"{int(angle)}°", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)

def run(exercise_key="squat"):
    ex    = EXERCISES[exercise_key]
    reps  = 0
    phase = "up"
    feedback = ("Get into position…", "warn")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    prev_time = cv2.getTickCount()

    with mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)         
            h, w  = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            
            now      = cv2.getTickCount()
            fps      = cv2.getTickFrequency() / (now - prev_time + 1)
            prev_time = now

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

               
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=WHITE, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=GREEN, thickness=2),
                )

               
                angle        = get_joint_angle(lm, ex["joint"], w, h)
                phase, counted = count_rep(angle, phase, ex)
                if counted:
                    reps += 1

                
                feedback = get_form_feedback(lm, exercise_key, w, h)

               
                joint_lm_map = {
                    "knee":  PL.LEFT_KNEE,
                    "elbow": PL.LEFT_ELBOW,
                }
                draw_angle_label(frame, lm, joint_lm_map[ex["joint"]], angle, w, h)

            else:
                feedback = ("No pose detected — step back from camera", "warn")

            draw_hud(frame, reps, phase, ex["name"], feedback, fps)
            cv2.imshow("Fitgent — AI Personal Trainer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                reps, phase = 0, "up"

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession complete — {ex['name']}: {reps} reps\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FitAgent MVP — AI Personal Trainer")
    parser.add_argument(
        "--exercise", "-e",
        choices=list(EXERCISES.keys()),
        default="squat",
        help="Exercise to track (default: squat)",
    )
    args = parser.parse_args()

    print(f"\n🏋️  Fitgent  |  Exercise: {EXERCISES[args.exercise]['name']}")
    print("   Q = quit   |   R = reset reps\n")
    run(args.exercise)
