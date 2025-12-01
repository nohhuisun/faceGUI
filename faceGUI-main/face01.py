import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import subprocess

# Check if running in nhuisun_face_py310 environment
def check_and_rerun_in_correct_env():
    """Check for required modules; if missing, rerun in nhuisun_face_py310."""
    conda_env_name = "nhuisun_face_py310"
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    
    # If already in the correct env, return early
    if current_env == conda_env_name:
        return
    
    # Try importing required modules
    required_modules = [("cv2", "opencv-python"), ("mediapipe", "mediapipe"), ("PIL", "Pillow")]
    missing = []
    for mod_name, pkg_name in required_modules:
        try:
            __import__(mod_name)
        except ModuleNotFoundError:
            missing.append(pkg_name)
    
    # If any module is missing, rerun in the correct environment
    if missing:
        script_path = os.path.abspath(__file__)
        conda_exe = r"C:\Users\504\miniconda3\Scripts\conda.exe"
        
        # Try to rerun
        try:
            cmd = [conda_exe, "run", "-n", conda_env_name, "--no-capture-output", "python", script_path]
            print(f"Switching to {conda_env_name} environment...", file=sys.stderr)
            result = subprocess.run(cmd, check=False)
            sys.exit(result.returncode)
        except Exception as e:
            err_msg = (
                f"Missing modules: {', '.join(missing)}\n\n"
                f"Please run this script using the Conda env '{conda_env_name}':\n\n"
                f"{conda_exe} run -n {conda_env_name} --no-capture-output python \"{script_path}\""
            )
            print(err_msg, file=sys.stderr)
            try:
                messagebox.showerror("Missing dependencies", err_msg)
            except Exception:
                pass
            sys.exit(1)

# Check environment at startup
check_and_rerun_in_correct_env()

try:
    import cv2
except ModuleNotFoundError as e:
    print(f"Failed to import cv2: {e}", file=sys.stderr)
    sys.exit(1)

import mediapipe as mp # Dlib 대신 MediaPipe 사용
from PIL import Image, ImageTk
import numpy as np
import math
import logging
import os
from tkinter import filedialog
from typing import Optional, Tuple

# --- 1. MediaPipe 초기화 ---
# MediaPipe Face Mesh 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,             # 한 번에 감지할 최대 얼굴 수
    refine_landmarks=True,       # 랜드마크 정밀도 개선
    min_detection_confidence=0.5,# 최소 감지 신뢰도
    min_tracking_confidence=0.5  # 최소 추적 신뢰도
)

# --- 로거 설정 (파일에 디버그 로그 기록) ---
LOG_PATH = os.path.join(os.path.dirname(__file__), "face01_debug.log")
logger = logging.getLogger("face01")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.debug("Initialized MediaPipe FaceMesh with refine_landmarks=%s, max_num_faces=%s, min_detection_confidence=%s, min_tracking_confidence=%s",
             True, 1, 0.5, 0.5)


def try_open_camera(index: int = 0) -> Tuple[Optional[cv2.VideoCapture], str]:
    """Try opening camera using several backends. Returns (cap, description).
    If none succeed, returns (None, last_description).
    """
    backends = [None]
    # Try common Windows backends if available
    try:
        backends.append(cv2.CAP_DSHOW)
    except Exception:
        pass
    try:
        backends.append(cv2.CAP_MSMF)
    except Exception:
        pass
    last_desc = "none"
    for b in backends:
        try:
            if b is None:
                cap = cv2.VideoCapture(index)
                desc = f"index={index} (default)"
            else:
                cap = cv2.VideoCapture(index, b)
                desc = f"index={index} backend={b}"
            ok = cap.isOpened()
            logger.debug("try_open_camera: tried %s -> isOpened=%s", desc, ok)
            if ok:
                return cap, desc
            else:
                try:
                    cap.release()
                except Exception:
                    pass
            last_desc = desc
        except Exception as e:
            logger.exception("try_open_camera exception for backend %s: %s", b, e)
    return None, last_desc


def list_available_cameras(max_index: int = 5):
    """Return a list of camera indices that can be opened."""
    available = []
    for i in range(0, max_index + 1):
        try:
            cap = None
            if hasattr(cv2, 'CAP_DSHOW'):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(i)
            ok = cap.isOpened()
            if ok:
                available.append(i)
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            pass
    logger.debug("list_available_cameras -> %s", available)
    return available

# --- 2. 관상 분석 함수 ---
# MediaPipe Face Mesh는 기본 468개(0-467) 랜드마크를 제공합니다.
# `refine_landmarks=True` 설정 시 눈 관련 추가 랜드마크(홍채 등)로 총 478개가 될 수 있습니다.
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
    try:
        if not landmarks:
            return None
        lm_list = landmarks.landmark
        if index < 0 or index >= len(lm_list):
            logger.debug("get_landmark_coords: index %s out of range (len=%s)", index, len(lm_list))
            return None
        lm = lm_list[index]
        x = int(lm.x * width)
        y = int(lm.y * height)
        return x, y
    except Exception as e:
        logger.exception("get_landmark_coords error: %s", e)
        return None

def calculate_distance(p1_x, p1_y, p2_x, p2_y):
    """두 점 사이의 유클리드 거리를 계산합니다."""
    return math.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)

def analyze_physiognomy_mp(landmarks, frame_width, frame_height):
    """
    MediaPipe Face Mesh 랜드마크를 기반으로 관상 정보를 분석하고 문자열을 반환합니다.
    """
    # 기본 468개 이상의 랜드마크가 있어야 정상 처리합니다.
    # (refine_landmarks=True일 때는 478개가 될 수 있으므로 엄격한 등호 검사는 제거)
    if not landmarks or len(landmarks.landmark) < 468:
        logger.debug("analyze_physiognomy_mp: insufficient landmarks (%s)",
                     0 if not landmarks else len(landmarks.landmark))
        return "얼굴 랜드마크를 찾지 못했습니다."
    
    analysis_results = []
    
    # 픽셀 좌표 얻기 (안전하게 None 체크)
    coords = {}
    for name, idx in (('nose_tip', NOSE_TIP), ('mouth_upper', MOUTH_UPPER), ('mouth_lower', MOUTH_LOWER),
                      ('left_eye_inner', LEFT_EYE_INNER), ('left_eye_outer', LEFT_EYE_OUTER)):
        val = get_landmark_coords(landmarks, idx, frame_width, frame_height)
        if val is None:
            logger.debug("analyze_physiognomy_mp: missing landmark %s (index %s)", name, idx)
            return "필요한 얼굴 랜드마크가 충분하지 않아 분석할 수 없습니다."
        coords[name] = val

    nose_tip_x, nose_tip_y = coords['nose_tip']
    mouth_upper_x, mouth_upper_y = coords['mouth_upper']
    mouth_lower_x, mouth_lower_y = coords['mouth_lower']
    left_eye_inner_x, left_eye_inner_y = coords['left_eye_inner']
    left_eye_outer_x, left_eye_outer_y = coords['left_eye_outer']
    
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
        # 창을 앞으로 올려 사용자가 비디오 영역을 바로 볼 수 있게 합니다.
        try:
            master.lift()
            master.attributes('-topmost', True)
            # 잠깐 최상위로 만든 뒤 원래대로 돌려놓기
            master.after(200, lambda: master.attributes('-topmost', False))
        except Exception:
            pass

        # 카메라 목록 확인 및 선택 UI
        cameras = list_available_cameras(4)
        self.cap = None
        if cameras:
            # 기본으로 첫 카메라 선택해서 오픈
            selected_idx = cameras[0]
            self.cap, cap_desc = try_open_camera(selected_idx)
            if self.cap is None or not self.cap.isOpened():
                logger.warning("Could not open default camera %s, cameras list=%s", selected_idx, cameras)
            else:
                logger.info("Camera opened successfully: %s", cap_desc)
        else:
            logger.warning("No available cameras detected: %s", cameras)
        
        self.width = 640
        self.height = 480
        if self.cap is not None:
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            except Exception:
                pass
        logger.debug("Camera opened: %s", False if self.cap is None else self.cap.isOpened())
        logger.debug("Requested frame size: %sx%s", self.width, self.height)
        logger.debug("Camera opened: %s", self.cap.isOpened())
        logger.debug("Requested frame size: %sx%s", self.width, self.height)

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
        # 카메라 선택 콤보 + 이미지 로드 버튼
        cam_controls = ttk.Frame(analysis_panel)
        cam_controls.pack(pady=5)
        ttk.Label(cam_controls, text="카메라 선택:").grid(row=0, column=0, sticky="w")
        self.cam_var = tk.StringVar()
        cam_list = [str(c) for c in list_available_cameras(6)]
        self.cam_combo = ttk.Combobox(cam_controls, textvariable=self.cam_var, values=cam_list, width=8)
        if cam_list:
            self.cam_combo.set(cam_list[0])
        self.cam_combo.grid(row=0, column=1, padx=5)
        self.btn_select_cam = ttk.Button(cam_controls, text="선택", command=self.select_camera)
        self.btn_select_cam.grid(row=0, column=2, padx=5)

        self.btn_load_image = ttk.Button(cam_controls, text="이미지 불러오기", command=self.load_image)
        self.btn_load_image.grid(row=1, column=0, columnspan=3, pady=6)
        # 상태 레이블: 카메라 연결/랜드마크 수를 표시
        self.status_var = tk.StringVar(value="카메라 상태: 확인 중...")
        self.status_label = ttk.Label(analysis_panel, textvariable=self.status_var, foreground="blue")
        self.status_label.pack(pady=6)
        
        # 실시간 업데이트 루프 시작
        self.delay = 15 
        # 디버그 플래그: 파일에 감지 상태를 기록합니다.
        self.DEBUG = True
        self.update_video()

    def update_video(self):
        """카메라에서 프레임을 읽고, 얼굴을 인식하여 특징점을 표시한 후 GUI에 업데이트합니다."""
        # 안전하게 카메라에서 프레임 읽기
        frame = None
        ret = False
        if self.cap is not None and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
            except Exception:
                ret = False

        current_analysis_text = "얼굴을 찾지 못했습니다."

        if ret and frame is not None:
            # 1. MediaPipe 처리
            frame = cv2.flip(frame, 1) # 좌우 반전 (거울 모드)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame) # MediaPipe 분석 실행

            if results and results.multi_face_landmarks:
                # 첫 번째 감지된 얼굴만 사용
                landmarks = results.multi_face_landmarks[0]
                if self.DEBUG:
                    try:
                        cnt = len(landmarks.landmark)
                        logger.debug("Face landmarks detected: %s", cnt)
                        coords_sample = [(round(lm.x,3), round(lm.y,3)) for lm in landmarks.landmark[:3]]
                        logger.debug("Sample landmarks (normalized): %s", coords_sample)
                    except Exception as e:
                        logger.exception("Error logging landmarks: %s", e)

                # 2. 특징점 그리기 (실제 프레임 크기 사용)
                fh, fw = rgb_frame.shape[:2]
                for lm in landmarks.landmark:
                    x = int(lm.x * fw)
                    y = int(lm.y * fh)
                    cv2.circle(rgb_frame, (x, y), 1, (0, 255, 0), -1)

                # 3. 관상 분석 실행 (픽셀 계산에 실제 프레임 크기를 전달)
                current_analysis_text = analyze_physiognomy_mp(landmarks, fw, fh)
                # 상태 표시 업데이트
                try:
                    self.status_var.set(f"카메라: 연결됨  랜드마크: {len(landmarks.landmark)}")
                except Exception:
                    self.status_var.set("카메라: 연결됨  랜드마크: ?")

            # 분석 결과를 GUI 텍스트 위젯에 업데이트
            self.analysis_text_widget.delete(1.0, tk.END)
            self.analysis_text_widget.insert(tk.END, current_analysis_text)

            # OpenCV 프레임을 Tkinter에서 표시할 수 있는 이미지로 변환
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        else:
            # 카메라 프레임이 없을 때는 빈 회색 이미지로 표시 및 상태 업데이트
            blank = Image.new('RGB', (self.width, self.height), (120,120,120))
            imgtk = ImageTk.PhotoImage(image=blank)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            # 카메라가 열려 있지 않거나 프레임을 못 받아왔을 때 상태 표시
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.status_var.set("카메라: 연결되지 않음")
                else:
                    self.status_var.set("카메라: 연결됨(프레임 없음)")
            except Exception:
                self.status_var.set("카메라: 상태 알 수 없음")

        # 다음 업데이트 예약
        self.master.after(self.delay, self.update_video)

    def select_camera(self):
        """Reopen camera from combobox selection."""
        sel = self.cam_var.get()
        try:
            idx = int(sel)
        except Exception:
            messagebox.showwarning("카메라 선택", "유효한 카메라 인덱스를 선택하세요.")
            return
        # close previous
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap, desc = try_open_camera(idx)
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("카메라 오류", f"카메라를 열 수 없습니다: {desc}")
            logger.error("select_camera failed: %s", desc)
        else:
            logger.info("select_camera opened: %s", desc)

    def load_image(self):
        """Open an image file and run analysis once (headless path)."""
        path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp"), ("All files","*.*")])
        if not path:
            return
        # 일부 Windows/OpenCV 빌드에서는 한글(유니코드) 경로에서 cv2.imread가 실패합니다.
        # 안정적으로 열기 위해 먼저 PIL로 시도하고, 실패하면 numpy+cv2.imdecode로 fallback 합니다.
        try:
            pil_img = Image.open(path).convert('RGB')
            rgb = np.array(pil_img)
        except Exception as e:
            logger.exception("load_image: PIL failed to open image: %s", e)
            try:
                # cv2.imread가 유니코드 경로에서 실패하면 fromfile+imdecode 방식 사용
                data = np.fromfile(path, dtype=np.uint8)
                img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise ValueError('cv2.imdecode returned None')
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e2:
                logger.exception("load_image: cv2 fallback also failed: %s", e2)
                messagebox.showerror("이미지 오류", f"이미지를 읽을 수 없습니다.\n{e2}")
                return
        results = face_mesh.process(rgb)
        analysis_text = "얼굴을 찾지 못했습니다."
        if results and results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            # use actual image size
            h, w = rgb.shape[:2]
            analysis_text = analyze_physiognomy_mp(lm, w, h)
            # draw landmarks for display
            for lm_pt in lm.landmark:
                x = int(lm_pt.x * w)
                y = int(lm_pt.y * h)
                cv2.circle(rgb, (x, y), 1, (0,255,0), -1)
        # show in GUI
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.analysis_text_widget.delete(1.0, tk.END)
        self.analysis_text_widget.insert(tk.END, analysis_text)

    def on_closing(self):
        """GUI 창이 닫힐 때 카메라와 창을 정리합니다."""
        try:
            if self.cap and self.cap.isOpened():
                try:
                    self.cap.release()
                except Exception:
                    pass
        except Exception:
            pass
        # MediaPipe 객체 해제 (선택 사항)
        try:
            face_mesh.close()
        except Exception:
            pass
        self.master.destroy()

# --- 4. 메인 실행 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PhysiognomyApp(root)
    root.mainloop()