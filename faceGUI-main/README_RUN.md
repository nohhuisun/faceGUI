실행 방법

1) Conda 환경 활성화(이미 `nhuisun_face_py310` 환경이 있으면 사용):

```cmd
C:\Users\504\miniconda3\Scripts\conda.exe activate nhuisun_face_py310
```

2) 필요한 패키지 설치(새 env인 경우):

```cmd
C:\Users\504\miniconda3\Scripts\conda.exe run -n nhuisun_face_py310 --no-capture-output python -m pip install -r requirements.txt
```

3) 프로그램 실행:

```cmd
C:\Users\504\miniconda3\Scripts\conda.exe run -n nhuisun_face_py310 --no-capture-output python c:\노희선pj\황동하교수님\nhuisun_face\faceGUI-main\face01.py
```

문제 해결 팁
- 카메라가 열리지 않으면 다른 앱이 카메라를 사용중인지 확인하고 종료하세요.
- Windows에서 카메라 권한을 확인하세요: 설정 -> 개인정보 및 보안 -> 카메라
- `face01_debug.log` 파일에 진단 로그가 남습니다. 문제가 계속되면 그 파일 내용을 붙여주세요.
