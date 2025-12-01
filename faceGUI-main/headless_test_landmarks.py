"""
Headless test: capture a few frames from webcam, run MediaPipe face_mesh.process,
log detected landmark counts to `face01_debug.log` and print a short status to stdout.
This script is safe to run non-interactively and exits after a few frames.
"""
import time
import cv2
import mediapipe as mp
from pathlib import Path

LOG_FILE = Path(__file__).parent / "face01_debug.log"
print("Writing logs to:", LOG_FILE)

mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        raise SystemExit(1)

    # grab a few frames
    frames_to_capture = 6
    for i in range(frames_to_capture):
        ret, frame = cap.read()
        if not ret:
            print("WARNING: failed to read frame")
            time.sleep(0.1)
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            cnt = len(lm.landmark)
            print(f"Frame {i}: detected landmarks = {cnt}")
        else:
            print(f"Frame {i}: no face detected")
        time.sleep(0.2)

    cap.release()
    print("Headless test finished.")
