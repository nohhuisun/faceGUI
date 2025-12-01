import sys, traceback
print('PYTHON EXECUTABLE:', sys.executable)
print('PYTHON PATH:')
for p in sys.path:
    print(p)
print('\nIMPORT TEST:')
try:
    import cv2
    print('OK', cv2.__version__)
except Exception:
    traceback.print_exc()
