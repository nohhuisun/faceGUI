"""
Camera diagnostic: tries indices 0-3 with default and common Windows backends,
prints results to stdout and writes to `face01_debug.log` via face01.logger if available.
"""
import cv2
import time
from pathlib import Path

LOG = Path(__file__).parent / "camera_diag_output.txt"
print("Camera diagnostic — output also saved to:", LOG)

backends = [None]
if hasattr(cv2, 'CAP_DSHOW'):
    backends.append(cv2.CAP_DSHOW)
if hasattr(cv2, 'CAP_MSMF'):
    backends.append(cv2.CAP_MSMF)

results = []
for idx in range(0, 4):
    for b in backends:
        desc = f"index={idx} backend={'default' if b is None else b}"
        try:
            if b is None:
                cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx, b)
            ok = cap.isOpened()
            if ok:
                ret, frame = cap.read()
                frame_ok = ret and frame is not None
            else:
                frame_ok = False
            results.append((desc, ok, frame_ok))
            print(f"{desc} -> isOpened={ok}, read_ok={frame_ok}")
            cap.release()
        except Exception as e:
            print(f"{desc} -> exception: {e}")

with open(LOG, 'w', encoding='utf-8') as f:
    for line in results:
        f.write(str(line) + '\n')

print('Diagnostic finished.')
