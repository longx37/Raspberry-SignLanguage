"""手语识别公共工具（手部 + 鼻子/肩膀版本 / Mediapipe 0.10.x Tasks API）
特征维度: 21点×3(xyz)×2手 + 鼻子+双肩×3 = 135
"""

import cv2
import numpy as np
import os
import time
import threading
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

_MODEL_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__))),
] + ([r'D:\temp_models'] if os.name == 'nt' else [])

def _find_model(filename):
    for directory in _MODEL_PATHS:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return os.path.join(_MODEL_PATHS[0], filename)

HAND_MODEL_PATH = _find_model('hand_landmarker.task')
POSE_MODEL_PATH = _find_model('pose_landmarker.task')
HAND_SINGLE_FEATURES = 21 * 3   # 单手: 21个关键点 × (x,y,z)
BODY_FEATURES = 3 * 3          # 鼻子 + 双肩: 3个关键点 × (x,y,z)
FEATURE_DIM = HAND_SINGLE_FEATURES * 2 + BODY_FEATURES  # 135 维

# 姿态关键点索引 (Mediapipe Pose)
POSE_NOSE = 0
POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12

# 21 个手部关键点连线（用于可视化骨架）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

_CHINESE_FONT_PATHS = [
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/System/Library/Fonts/PingFang.ttc',
]
_chinese_fonts = {}  # 按字号缓存 PIL 字体对象

def _get_chinese_font(size=32):
    if size not in _chinese_fonts:
        font = None
        for fp in _CHINESE_FONT_PATHS:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, size)
                break
        if font is None:
            font = ImageFont.load_default()
        _chinese_fonts[size] = font
    return _chinese_fonts[size]


def get_chinese_text_size(text, font_size=32):
    """返回 (width, height) — 用于居中布局"""
    font = _get_chinese_font(font_size)
    bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def put_chinese_text(img, text, position, font_size=32, color=(255, 255, 255),
                    outline_color=None):
    """在 OpenCV 图像上绘制中文（PIL 渲染 + alpha 合成，原地修改）
    position 为文字左下角坐标，与 cv2.putText 一致。
    outline_color 可指定描边颜色，用于字幕风格白字黑边。
    """
    if not text:
        return

    font = _get_chinese_font(font_size)
    rgb_color = (color[2], color[1], color[0])
    bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad = 6 if outline_color else 4
    if outline_color:
        pad += 2

    layer = Image.new('RGBA', (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    text_x = pad - bbox[0]
    text_y = pad - bbox[1]

    if outline_color:
        rgb_outline = (outline_color[2], outline_color[1], outline_color[0])
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)]:
            draw.text((text_x + dx, text_y + dy), text, font=font,
                      fill=(*rgb_outline, 255))

    draw.text((text_x, text_y), text, font=font, fill=(*rgb_color, 255))

    layer_arr = np.array(layer)
    fg_rgb = layer_arr[:, :, :3]
    alpha = layer_arr[:, :, 3:4] / 255.0
    fg_bgr = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    x, y = position
    x1 = max(0, x)
    y1 = max(0, y - th - pad)
    x2 = min(w, x + tw + pad * 2)
    y2 = min(h, y + pad)

    pw, ph = x2 - x1, y2 - y1
    if pw <= 0 or ph <= 0:
        return

    lx1 = x1 - x
    ly1 = y1 - (y - th - pad)
    lx2 = lx1 + pw
    ly2 = ly1 + ph

    roi = img[y1:y2, x1:x2]
    a = alpha[ly1:ly2, lx1:lx2]
    fg = fg_bgr[ly1:ly2, lx1:lx2]

    blended = (a * fg + (1 - a) * roi).astype(np.uint8)
    img[y1:y2, x1:x2] = blended


def create_detectors():
    """创建 Mediapipe Hand Landmarker + Pose Landmarker 检测器

    Returns:
        (hand_detector, pose_detector)
    """
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    return hand_detector, pose_detector


def detect_frame(image_bgr, hand_detector):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    hand_result = hand_detector.detect(mp_image)

    return hand_result


def detect_pose(image_bgr, pose_detector):
    """对一帧 BGR 图像运行 Mediapipe 姿态检测（提取肩膀坐标）"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    return pose_detector.detect(mp_image)


def extract_keypoints(hand_result, pose_result=None):
    """从检测结果提取 135 维特征向量 [左手63|右手63|鼻子3|左肩3|右肩3]
    未检测到时对应部分为全零。
    """
    lh = np.zeros(HAND_SINGLE_FEATURES)
    rh = np.zeros(HAND_SINGLE_FEATURES)

    if hand_result.hand_landmarks:
        for idx, landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].category_name
            coords = np.array([[lm.x, lm.y, lm.z]
                               for lm in landmarks]).flatten()
            if handedness == 'Left':
                lh = coords
            else:
                rh = coords

    body = np.zeros(BODY_FEATURES)
    if pose_result is not None and pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks[0]
        nose = np.array([lm[POSE_NOSE].x, lm[POSE_NOSE].y, lm[POSE_NOSE].z])
        ls = np.array([lm[POSE_L_SHOULDER].x, lm[POSE_L_SHOULDER].y, lm[POSE_L_SHOULDER].z])
        rs = np.array([lm[POSE_R_SHOULDER].x, lm[POSE_R_SHOULDER].y, lm[POSE_R_SHOULDER].z])
        body = np.concatenate([nose, ls, rs])

    return np.concatenate([lh, rh, body])


def draw_hand_landmarks(image, landmarks, is_left=True):
    """在图像上绘制单手 21 个关键点 + 骨架连线"""
    h, w = image.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    line_color = (121, 22, 76) if is_left else (245, 117, 66)
    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(image, pts[a], pts[b], line_color, 2)

    for i, (x, y) in enumerate(pts):
        if i == 0:
            color = (0, 255, 255)
        elif i in (4, 8, 12, 16, 20):
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.circle(image, (x, y), 3, color, -1)


def draw_all_landmarks(image, hand_result):
    if hand_result.hand_landmarks:
        for idx, landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].category_name
            draw_hand_landmarks(image, landmarks, handedness == 'Left')


def draw_shoulders(image, pose_result):
    """在图像上绘制鼻子 + 肩膀关键点 + 连线（鼻子品红, 肩膀青色）"""
    if pose_result is None or not pose_result.pose_landmarks:
        return
    h, w = image.shape[:2]
    lm = pose_result.pose_landmarks[0]
    nose = (int(lm[POSE_NOSE].x * w), int(lm[POSE_NOSE].y * h))
    ls = (int(lm[POSE_L_SHOULDER].x * w), int(lm[POSE_L_SHOULDER].y * h))
    rs = (int(lm[POSE_R_SHOULDER].x * w), int(lm[POSE_R_SHOULDER].y * h))
    cv2.circle(image, nose, 7, (255, 0, 255), -1)     # 鼻子 品红
    cv2.circle(image, ls, 7, (255, 255, 0), -1)       # 左肩 青色
    cv2.circle(image, rs, 7, (255, 255, 0), -1)       # 右肩 青色
    cv2.line(image, ls, rs, (255, 255, 0), 2)         # 肩膀连线


# ══════════════════════════════════════════════════════════════
#  异步检测器 —— 后台线程跑 Mediapipe，主线程不再阻塞等待
# ══════════════════════════════════════════════════════════════

class AsyncDetector:
    """在后台线程持续运行 Mediapipe 检测，主线程无阻塞获取最新结果。

    用法:
        det = AsyncDetector(hand_detector, pose_detector)
        while True:
            frame = cap.read()
            det.submit(frame)                     # 非阻塞
            hand, pose = det.get_results()        # 非阻塞，拿最新结果
            ...
        det.stop()
    """

    def __init__(self, hand_detector, pose_detector):
        self._hand = hand_detector
        self._pose = pose_detector
        self._lock = threading.Lock()
        self._pending_frame = None
        self._hand_result = None
        self._pose_result = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame_bgr):
        """提交新帧用于后台检测（拷贝帧以避免竞争）"""
        with self._lock:
            self._pending_frame = frame_bgr.copy()

    def get_results(self):
        """立即返回最近一次检测结果 (hand_result, pose_result)"""
        with self._lock:
            return self._hand_result, self._pose_result

    def _loop(self):
        while self._running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is not None:
                hand = detect_frame(frame, self._hand)
                pose = detect_pose(frame, self._pose)
                with self._lock:
                    self._hand_result = hand
                    self._pose_result = pose
            else:
                time.sleep(0.002)

    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
