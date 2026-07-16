"""
03_实时检测.py（纯手部版本 / 树莓派优化）
============================
加载训练好的 LSTM 模型，使用摄像头进行实时手势识别。
右侧按钮面板 + 底部历史记录（可滚动）。
"""

import cv2
import numpy as np
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    create_detectors, extract_keypoints,
    draw_all_landmarks, draw_shoulders,
    put_chinese_text, get_chinese_text_size, FEATURE_DIM,
    AsyncDetector,
)
from oled_display import OledDisplay

ACTIONS = np.array(['你', '想', '买', '什么', '我', '看看', '那',
                    '衣服', '请', '问', '钱', '多少', '太贵了', '便宜点', '可以（吗）'])

TFLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'action_int8.tflite')
SEQUENCE_LENGTH = 30
CONFIRM_DURATION = 1.75
SENTENCE_TIMEOUT = 7.5
WINDOW_W, WINDOW_H = 960, 720

# 布局常量
CAM_W, CAM_H = 640, 480
PANEL_X = CAM_W + 10
PANEL_W = WINDOW_W - PANEL_X - 10
BTN_W, BTN_H = 220, 52
BTN_X = PANEL_X + (PANEL_W - BTN_W) // 2

BTN_CLEAR_RECT = (BTN_X, 50, BTN_W, BTN_H)
BTN_EXIT_RECT  = (BTN_X, 130, BTN_W, BTN_H)

HIST_Y = CAM_H + 10
HIST_H = WINDOW_H - HIST_Y - 10
HIST_X = 10
HIST_W = CAM_W - 20
HIST_LINE_H = 28

INFO_X = HIST_X + HIST_W + 10
INFO_W = WINDOW_W - INFO_X - 10
INFO_Y = HIST_Y

SCROLL_BTN_SIZE = 22
SCROLL_UP_RECT   = (HIST_X + HIST_W - SCROLL_BTN_SIZE - 6, HIST_Y + 5, SCROLL_BTN_SIZE, SCROLL_BTN_SIZE)
SCROLL_DOWN_RECT = (HIST_X + HIST_W - SCROLL_BTN_SIZE - 6, HIST_Y + HIST_H - SCROLL_BTN_SIZE - 5, SCROLL_BTN_SIZE, SCROLL_BTN_SIZE)


def _semitrans_bar(frame, x, y, w, h, progress):
    """半透明进度条，直接画在 frame 上"""
    roi = frame[y:y+h, x:x+w].astype(np.float32)
    track = np.full((h, w, 3), (50, 50, 50), dtype=np.float32)
    frame[y:y+h, x:x+w] = (track * 0.30 + roi * 0.70).astype(np.uint8)
    fill_w = max(0, int(w * progress))
    if fill_w > 0:
        roi_f = frame[y:y+h, x:x+fill_w].astype(np.float32)
        fill = np.full((h, fill_w, 3), (160, 160, 160), dtype=np.float32)
        frame[y:y+h, x:x+fill_w] = (fill * 0.45 + roi_f * 0.55).astype(np.uint8)


def _is_inside(rect, px, py):
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def _draw_panel_bg(canvas, x, y, w, h, color=(245, 245, 245)):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 200, 200), 1)


def draw_button(canvas, rect, text):
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (210, 210, 210), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (160, 160, 160), 1)
    tw, th = get_chinese_text_size(text, font_size=22)
    put_chinese_text(canvas, text, (x + (w - tw) // 2, y + (h + th) // 2),
                     font_size=22, color=(50, 50, 50))


def draw_scroll_arrow(canvas, rect, direction):
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (230, 230, 230), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (180, 180, 180), 1)
    cx, cy = x + w // 2, y + h // 2
    pts = np.array([[cx, cy - 4], [cx - 5, cy + 3], [cx + 5, cy + 3]]) if direction == 'up' \
         else np.array([[cx, cy + 4], [cx - 5, cy - 3], [cx + 5, cy - 3]])
    cv2.fillPoly(canvas, [pts], (100, 100, 100))


def draw_info_panel(canvas, x, y, w, h, sentence_count):
    _draw_panel_bg(canvas, x, y, w, h)
    put_chinese_text(canvas, '信息', (x + 10, y + 30), font_size=22, color=(60, 60, 60))
    for i, line in enumerate([f'已识别句子: {sentence_count}', f'类别数: {len(ACTIONS)}']):
        put_chinese_text(canvas, line, (x + 10, y + 60 + i * 28),
                         font_size=16, color=(90, 90, 90))


def draw_history_panel(canvas, x, y, w, h, history, scroll_offset):
    _draw_panel_bg(canvas, x, y, w, h)
    put_chinese_text(canvas, '历史记录', (x + 10, y + 28), font_size=22, color=(60, 60, 60))
    cv2.line(canvas, (x + 10, y + 34), (x + w - 10, y + 34), (200, 200, 200), 1)

    content_y = y + 36
    visible = (h - 40) // HIST_LINE_H
    total = len(history)
    max_scroll = max(0, total - visible)
    scroll_offset = max(0, min(scroll_offset, max_scroll))

    if total == 0:
        put_chinese_text(canvas, '(暂无记录)', (x + 15, content_y + 40),
                         font_size=18, color=(160, 160, 160))
    else:
        for i in range(visible):
            idx = total - 1 - scroll_offset - i
            if idx < 0:
                break
            ts, sent = history[idx]
            put_chinese_text(canvas, f'{ts}  {sent}',
                             (x + 14, content_y + (i + 1) * HIST_LINE_H - 4),
                             font_size=18, color=(50, 50, 50))

    if scroll_offset > 0:
        draw_scroll_arrow(canvas, SCROLL_UP_RECT, 'up')
    if scroll_offset < max_scroll:
        draw_scroll_arrow(canvas, SCROLL_DOWN_RECT, 'down')
    return scroll_offset


def main():
    if not os.path.exists(TFLITE_PATH):
        print(f"错误: 模型文件不存在: {TFLITE_PATH}")
        sys.exit(1)

    # ── 加载 TFLite 模型 ──
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    num_model_classes = output_details[0]['shape'][-1]
    print(f"模型加载成功: {TFLITE_PATH} (TFLite int8)")

    if num_model_classes != len(ACTIONS):
        print(f"错误: 模型输出 {num_model_classes} 类，但 ACTIONS 定义了 {len(ACTIONS)} 个")
        print(f"请修改 ACTIONS 数组使二者匹配后重试")
        sys.exit(1)
    print(f"识别类别: {list(ACTIONS)}")

    # 热身推理
    print("预热模型...")
    dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()
    print("预热完成")

    # ── OLED 显示屏 ──
    oled = OledDisplay()

    # ── 摄像头 ──
    picam2 = None
    cap = None
    use_picam2 = False

    try:
        from picamera2 import Picamera2
        cameras = Picamera2.global_camera_info()
        if cameras:
            picam2 = Picamera2(camera_num=cameras[0].get('Num', 0))
            picam2.configure(picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "BGR888"}))
            picam2.set_controls({"AwbEnable": True, "AwbMode": 0})
            picam2.start()
            time.sleep(5.0)
            use_picam2 = True
            print(f"摄像头: picamera2 ({cameras[0].get('Model', '')})")
        else:
            print("picamera2 未检测到摄像头，回退 OpenCV...")
    except ImportError:
        print("picamera2 不可用，回退 OpenCV...")

    if not use_picam2:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误: 无法打开摄像头。请检查摄像头连接。")
            sys.exit(1)

    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_AUTOSIZE)

    # ── 鼠标回调 ──
    click_pos = None
    wheel_dir = 0

    def on_mouse(event, x, y, flags, param):
        nonlocal click_pos, wheel_dir
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            wheel_dir = 1 if flags > 0 else -1

    cv2.setMouseCallback('Sign Language Recognition', on_mouse)

    print("加载 Mediapipe 手部+姿态检测器（异步模式）...")
    hand_detector, pose_detector = create_detectors()
    det = AsyncDetector(hand_detector, pose_detector)
    print("就绪。按 'q' 键退出。\n")

    # ── 状态 ──
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    for _ in range(SEQUENCE_LENGTH):
        sequence.append(np.zeros(FEATURE_DIM, dtype=np.float32))
    last_label = ''
    last_confidence = 0.0

    candidate_word = None
    candidate_start = 0.0
    confirmed_sentence = []
    last_confirm_time = 0.0
    sentence_active = False
    last_confirmed_word = None

    history = []          # [(timestamp_str, sentence_str), ...]
    scroll_offset = 0

    fps = 0.0
    fps_timer = time.time()
    fps_counter = 0
    now = time.time()

    exit_flag = False

    try:
        while True:
            fps_counter += 1
            if fps_counter >= 10:
                t = time.time()
                fps = fps_counter / (t - fps_timer)
                fps_timer = t
                fps_counter = 0

            # ---- 取帧 ----
            if use_picam2:
                frame = picam2.capture_array("main")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                ret, frame = cap.read()
                if not ret:
                    continue

            h, w = frame.shape[:2]

            # ---- 提交异步检测 + 拿最新结果 ----
            det.submit(frame)
            hand_result, pose_result = det.get_results()

            if hand_result:
                draw_all_landmarks(frame, hand_result)
            if pose_result:
                draw_shoulders(frame, pose_result)

            hand_detected = bool(hand_result and hand_result.hand_landmarks)
            if hand_detected:
                keypoints = extract_keypoints(hand_result, pose_result)
                sequence.append(keypoints)

            now = time.time()

            # ---- 处理鼠标 ----
            if click_pos is not None:
                px, py = click_pos
                click_pos = None

                if _is_inside(BTN_CLEAR_RECT, px, py):
                    # 清除当前句子
                    confirmed_sentence = []
                    sentence_active = False
                    candidate_word = None
                    candidate_start = 0.0
                    last_confirmed_word = None
                    print("[按钮] 已清除当前句子")

                elif _is_inside(BTN_EXIT_RECT, px, py):
                    print("[按钮] 结束识别")
                    exit_flag = True

                elif _is_inside(SCROLL_UP_RECT, px, py):
                    scroll_offset += 1

                elif _is_inside(SCROLL_DOWN_RECT, px, py):
                    scroll_offset = max(0, scroll_offset - 1)

            # 鼠标滚轮
            if wheel_dir != 0:
                scroll_offset -= wheel_dir  # 向上滚 = 看更旧的 = offset+
                if scroll_offset < 0:
                    scroll_offset = 0
                wheel_dir = 0

            if exit_flag:
                break

            # ---- 推理 ----
            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(sequence), axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])[0]
                predicted_idx = np.argmax(predictions)
                predicted_label = ACTIONS[predicted_idx]
                confidence = predictions[predicted_idx]

                last_label = predicted_label
                last_confidence = confidence

                # 句子状态机
                if sentence_active and (now - last_confirm_time) > SENTENCE_TIMEOUT:
                    # 句子结束 → 保存到历史
                    sent_text = ' '.join(confirmed_sentence)
                    ts = time.strftime('%H:%M:%S', time.localtime(now))
                    history.append((ts, sent_text))
                    print(f"句子结束: {sent_text}")
                    confirmed_sentence = []
                    sentence_active = False
                    candidate_word = None
                    candidate_start = 0.0
                    last_confirmed_word = None
                    scroll_offset = 0  # 回到最新

                # 当前句子中已出现的词不允许再次成为候选（防止模型振荡导致重复确认）
                already_in_sentence = predicted_label in confirmed_sentence

                if candidate_word is None:
                    if predicted_label != last_confirmed_word and not already_in_sentence:
                        candidate_word = predicted_label
                        candidate_start = now
                elif predicted_label == candidate_word:
                    if now - candidate_start >= CONFIRM_DURATION:
                        if not sentence_active:
                            confirmed_sentence = []
                        confirmed_sentence.append(candidate_word)
                        last_confirm_time = now
                        last_confirmed_word = candidate_word
                        sentence_active = True
                        print(f"确认: {candidate_word}  →  句子: {' '.join(confirmed_sentence)}")
                        candidate_word = None
                        candidate_start = 0.0
                else:
                    if predicted_label != last_confirmed_word and not already_in_sentence:
                        candidate_word = predicted_label
                        candidate_start = now
                    else:
                        candidate_word = None
                        candidate_start = 0.0

                if not hand_detected:
                    candidate_word = None
                    candidate_start = 0.0

            # OLED
            if oled.available:
                if hand_detected and last_label:
                    oled.set_word(last_label, show_progress=(candidate_word is not None))
                else:
                    oled.set_word('-', show_progress=False)

            # ── frame 叠加：顶部词 ──
            if hand_detected and last_label:
                put_chinese_text(frame, f'{last_label}  {last_confidence:.0%}',
                                 (12, 38), font_size=30,
                                 color=(255, 255, 255), outline_color=(0, 0, 0))
            else:
                put_chinese_text(frame, '-', (12, 38), font_size=30,
                                 color=(200, 200, 200), outline_color=(0, 0, 0))

            # ── frame 叠加：底部句子 + 8s 进度条 ──
            if sentence_active and confirmed_sentence:
                sent_text = ' '.join(confirmed_sentence)
                tw, _ = get_chinese_text_size(sent_text, font_size=28)
                put_chinese_text(frame, sent_text, ((CAM_W - tw) // 2, CAM_H - 18),
                                 font_size=28, color=(255, 255, 255), outline_color=(0, 0, 0))
                remaining = SENTENCE_TIMEOUT - (now - last_confirm_time)
                if remaining > 0:
                    _semitrans_bar(frame, 12, CAM_H - 8, CAM_W - 24, 4,
                                   remaining / SENTENCE_TIMEOUT)
            elif sequence:
                placeholder = '等待手势...'
                pw, _ = get_chinese_text_size(placeholder, font_size=24)
                put_chinese_text(frame, placeholder, ((CAM_W - pw) // 2, CAM_H - 18),
                                 font_size=24, color=(180, 180, 180), outline_color=(0, 0, 0))

            # FPS（画面右上角）
            cv2.putText(frame, f'{fps:.0f}', (w - 48, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 合成画布
            canvas = np.full((WINDOW_H, WINDOW_W, 3), (255, 255, 255), dtype=np.uint8)
            canvas[0:CAM_H, 0:CAM_W] = frame

            scroll_offset = draw_history_panel(canvas, HIST_X, HIST_Y, HIST_W, HIST_H,
                                               history, scroll_offset)
            draw_button(canvas, BTN_CLEAR_RECT, '清除当前句子')
            draw_button(canvas, BTN_EXIT_RECT, '结束识别')
            draw_info_panel(canvas, INFO_X, INFO_Y, INFO_W,
                            HIST_Y + HIST_H - INFO_Y, len(history))

            cv2.imshow('Sign Language Recognition', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n检测已停止。")
    finally:
        det.stop()
        if picam2:
            picam2.stop()
            picam2.close()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        hand_detector.close()
        pose_detector.close()
        oled.clear()
        print("资源已释放，退出。")


if __name__ == '__main__':
    main()
