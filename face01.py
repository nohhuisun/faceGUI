import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp # Dlib 대신 MediaPipe 사용
from PIL import Image, ImageTk
import math

# --- 1. MediaPipe 초기화 ---
# MediaPipe Face Mesh 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,             # 한 번에 감지할 최대 얼굴 수
    refine_landmarks=True,       # 랜드마크 정밀도 개선
    min_detection_confidence=0.5,# 최소 감지 신뢰도
    min_tracking_confidence=0.5  # 최소 추적 신뢰도
)

# --- 2. 관상 분석 함수 ---
# MediaPipe는 0-467번까지 478개의 랜드마크를 제공합니다.
# 주요 특징점 인덱스 (MediaPipe Face Mesh 기준, 대략적인 위치)
LEFT_EYE_INNER = 33
LEFT_EYE_OUTER = 133
NOSE_TIP = 1
MOUTH_UPPER = 13
MOUTH_LOWER = 14
CHIN_CENTER = 199

def get_landmark_coords(landmarks, index, width, height):
    """MediaPipe 랜드마크 객체에서 픽셀 좌표를 계산합니다."""
    # 랜드마크 좌표는 0.0에서 1.0 사이의 정규화된 값입니다.
    lm = landmarks.landmark[index]
    x = int(lm.x * width)
    y = int(lm.y * height)
    return x, y

def calculate_distance(p1_x, p1_y, p2_x, p2_y):
    """두 점 사이의 유클리드 거리를 계산합니다."""
    return math.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)

def analyze_physiognomy_mp(landmarks, frame_width, frame_height):
    """
    MediaPipe Face Mesh 랜드마크를 기반으로 관상 정보를 분석하고 문자열을 반환합니다.
    """
    if not landmarks or len(landmarks.landmark) != 468:
        return "얼굴 랜드마크를 찾지 못했습니다."
    
    analysis_results = []
    
    # 픽셀 좌표 얻기
    nose_tip_x, nose_tip_y = get_landmark_coords(landmarks, NOSE_TIP, frame_width, frame_height)
    mouth_upper_x, mouth_upper_y = get_landmark_coords(landmarks, MOUTH_UPPER, frame_width, frame_height)
    mouth_lower_x, mouth_lower_y = get_landmark_coords(landmarks, MOUTH_LOWER, frame_width, frame_height)
    left_eye_inner_x, left_eye_inner_y = get_landmark_coords(landmarks, LEFT_EYE_INNER, frame_width, frame_height)
    left_eye_outer_x, left_eye_outer_y = get_landmark_coords(landmarks, LEFT_EYE_OUTER, frame_width, frame_height)
    
    # 1. 인중 길이 (코 끝 ~ 윗입술)
    philtrum_length = calculate_distance(nose_tip_x, nose_tip_y, mouth_upper_x, mouth_upper_y)
    analysis_results.append(f"🗣️ 인중 길이 (추정): {int(philtrum_length)} 픽셀")
    if philtrum_length > 30:
        analysis_results.append(" - 인중이 길어 건강하고 안정적인 삶을 추구할 수 있습니다.")
    else:
        analysis_results.append(" - 인중이 보통이어서 솔직하고 활동적인 성향이 있을 수 있습니다.")

    # 2. 입술 두께 (윗입술 중앙 ~ 아랫입술 중앙)
    lip_thickness = calculate_distance(mouth_upper_x, mouth_upper_y, mouth_lower_x, mouth_lower_y)
    analysis_results.append(f"👄 입술 두께 (추정): {int(lip_thickness)} 픽셀")
    if lip_thickness > 15:
        analysis_results.append(" - 입술이 도톰하여 인정이 많고 식복이 있을 수 있습니다.")
    else:
        analysis_results.append(" - 입술이 얇거나 보통이어서 이성적이고 섬세한 경향이 있을 수 있습니다.")

    # 3. 눈의 폭 (왼쪽 눈 안쪽 끝 ~ 바깥쪽 끝)
    eye_width = calculate_distance(left_eye_inner_x, left_eye_inner_y, left_eye_outer_x, left_eye_outer_y)
    analysis_results.append(f"👁️ 눈 폭 (추정): {int(eye_width)} 픽셀")
    if eye_width > 60:
        analysis_results.append(" - 눈이 커서 감정 표현이 풍부하고 호기심이 많을 수 있습니다.")
    else:
        analysis_results.append(" - 눈이 작거나 보통이어서 신중하고 집중력이 강할 수 있습니다.")
    
    # 최종 결과 반환
    return "✅ 관상 분석 결과 (MediaPipe 예시):\n" + "\n".join(analysis_results)

# --- 3. GUI 클래스 정의 (Tkinter) ---
class PhysiognomyApp:
    def __init__(self, master):
        self.master = master
        master.title("웹캠 관상 분석 프로그램 (MediaPipe Ver.)")
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 카메라 설정
        self.cap = cv2.VideoCapture(0) 
        if not self.cap.isOpened():
            messagebox.showerror("카메라 오류", "웹캠을 열 수 없습니다.")
            master.destroy()
            return
        
        self.width = 640
        self.height = 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # GUI 레이아웃 설정
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill="both", expand=True)

        # 비디오 프레임 (왼쪽)
        self.video_label = ttk.Label(main_frame, borderwidth=2, relief="groove")
        self.video_label.pack(side="left", padx=10, pady=10)

        # 분석 결과 프레임 (오른쪽)
        analysis_panel = ttk.Frame(main_frame, padding="10")
        analysis_panel.pack(side="right", fill="y", padx=10, pady=10)

        ttk.Label(analysis_panel, text="관상 분석 결과", font=("Helvetica", 18, "bold")).pack(pady=10)
        
        self.analysis_text_widget = tk.Text(analysis_panel, wrap="word", width=45, height=25, font=("Helvetica", 12), 
                                            borderwidth=2, relief="solid")
        self.analysis_text_widget.pack(pady=5, padx=5, fill="both", expand=True)
        self.analysis_text_widget.insert(tk.END, "MediaPipe를 사용하여 얼굴을 인식합니다.")
        
        self.btn_quit = ttk.Button(analysis_panel, text="프로그램 종료", command=self.on_closing)
        self.btn_quit.pack(pady=20)
        
        # 실시간 업데이트 루프 시작
        self.delay = 15 
        self.update_video()

    def update_video(self):
        """카메라에서 프레임을 읽고, 얼굴을 인식하여 특징점을 표시한 후 GUI에 업데이트합니다."""
        ret, frame = self.cap.read()
        if ret:
            # 1. MediaPipe 처리
            frame = cv2.flip(frame, 1) # 좌우 반전 (거울 모드)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame) # MediaPipe 분석 실행

            current_analysis_text = "얼굴을 찾지 못했습니다."
            
            if results.multi_face_landmarks:
                # 첫 번째 감지된 얼굴만 사용
                landmarks = results.multi_face_landmarks[0]
                
                # 2. 특징점 그리기
                for idx, lm in enumerate(landmarks.landmark):
                    x = int(lm.x * self.width)
                    y = int(lm.y * self.height)
                    
                    # 모든 랜드마크에 작은 원 그리기
                    cv2.circle(rgb_frame, (x, y), 1, (0, 255, 0), -1) 
                    
                # 3. 관상 분석 실행
                current_analysis_text = analyze_physiognomy_mp(landmarks, self.width, self.height)

            # 분석 결과를 GUI 텍스트 위젯에 업데이트
            self.analysis_text_widget.delete(1.0, tk.END)
            self.analysis_text_widget.insert(tk.END, current_analysis_text)
            
            # OpenCV 프레임을 Tkinter에서 표시할 수 있는 이미지로 변환
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        # 다음 업데이트 예약
        self.master.after(self.delay, self.update_video)

    def on_closing(self):
        """GUI 창이 닫힐 때 카메라와 창을 정리합니다."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        # MediaPipe 객체 해제 (선택 사항)
        face_mesh.close()
        self.master.destroy()

# --- 4. 메인 실행 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PhysiognomyApp(root)
    root.mainloop()