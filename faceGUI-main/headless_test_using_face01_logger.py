"""
Import `face01` to initialize its file logger, then run a headless loop capturing a few frames
and logging detected landmark counts via `face01.logger` (so logs go to face01_debug.log).
"""
import time
import cv2
import mediapipe as mp
import face01

logger = face01.logger
logger.info("Starting headless test (using face01.logger)")

mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera in headless test")
        raise SystemExit(1)

    for i in range(6):
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame %s", i)
            time.sleep(0.1)
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            cnt = len(lm.landmark)
            logger.info("Frame %s: detected landmarks = %s", i, cnt)
        else:
            logger.info("Frame %s: no face detected", i)
        time.sleep(0.2)

    cap.release()
    logger.info("Headless test finished.")
    print("Headless test completed (logs written to face01_debug.log)")
