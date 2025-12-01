# faceGUI
11월 26일 파이썬 코드 작성
📸 웹캠 관상 분석 프로그램 (MediaPipe 기반)
이 프로그램은 웹캠을 통해 사용자의 얼굴을 실시간으로 인식하고, Google MediaPipe Face Mesh 기술을 활용하여 얼굴 특징점(랜드마크)을 감지합니다. 감지된 특징점의 거리와 비율을 계산하여 간단한 관상학적 분석(예시) 결과를 Tkinter 기반의 GUI 화면에 실시간으로 표시합니다.

🌟 주요 기능
실시간 웹캠 스트리밍: OpenCV를 이용한 웹캠 영상 실시간 표시.

MediaPipe Face Mesh: Dlib 대신 MediaPipe를 사용하여 빠르고 정확하게 478개의 얼굴 랜드마크 감지.

특징점 시각화: 감지된 랜드마크를 녹색 점으로 영상 위에 표시.

관상 분석: 인중 길이, 입술 두께, 눈 폭 등 주요 특징을 측정하여 분석 텍스트를 실시간으로 업데이트.

GUI: Python의 표준 라이브러리인 Tkinter를 사용하여 사용자 친화적인 그래픽 인터페이스 제공.

🛠️ 개발 환경 및 필수 라이브러리
이 프로그램을 실행하기 위해서는 다음 라이브러리들이 설치되어 있어야 합니다.

Python 3.x: (권장)

OpenCV (cv2): 카메라 및 영상 처리를 담당합니다.

MediaPipe: 얼굴 랜드마크 감지 핵심 엔진입니다.

Pillow (PIL): OpenCV 이미지를 Tkinter GUI에서 표시할 수 있도록 변환합니다.
<img width="1143" height="770" alt="image" src="https://github.com/user-attachments/assets/9ace5687-2274-47d2-8318-313ca0f1658a" />
