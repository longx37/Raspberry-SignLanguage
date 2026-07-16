"""
collect_testset.py — 树莓派上采集测试集
=========================================
参考 01_collect_dataset.py 的采集流程，适配树莓派（picamera2 + 中文 overlay）。
每个手势采集 N 次（默认 5 次），每次 30 帧，保存为 .npy 文件。

目录结构:
  TestData/
    ├── 你/
    │   ├── 0/
    │   │   ├── 0.npy   ← 第 0 帧 (135 维)
    │   │   ├── 1.npy
    │   │   └── ...     (共 30 帧)
    │   ├── 1/
    │   └── ...         (共 N 次)
    ├── 想/
    └── ...

用法:
  python collect_testset.py                # 每手势 5 次，全部 15 个手势
  python collect_testset.py --trials 3     # 每手势 3 次
  python collect_testset.py --start 3      # 从试次 3 开始（断点续采）
"""

import cv2
import numpy as np
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    create_detectors, detect_frame, detect_pose, extract_keypoints,
    draw_all_landmarks, draw_shoulders,
    put_chinese_text, FEATURE_DIM,
)

# ── 配置 ──────────────────────────────────────────────

ACTIONS = np.array(['你', '想', '买', '什么', '我', '看看', '那',
                    '衣服', '请', '问', '钱', '多少', '太贵了', '便宜点', '可以（吗）'])

SEQUENCE_LENGTH = 30
DEFAULT_TRIALS = 5
FRAME_INTERVAL_MS = 67          # 帧间隔（毫秒），与 PC 端训练数据采集一致
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TestData')


# ── 工具函数 ──────────────────────────────────────────

def setup_directories(args):
    """为每种手势 × 每次采集创建目录"""
    for action in ACTIONS:
        for trial in range(args.start, args.start + args.trials):
            dir_path = os.path.join(DATA_PATH, action, str(trial))
            os.makedirs(dir_path, exist_ok=True)
    print(f"测试数据目录已就绪: {DATA_PATH}")
    print(f"手势类别 ({len(ACTIONS)} 个): {list(ACTIONS)}")
    print(f"每种手势 {args.trials} 次采集 (编号 {args.start}~{args.start + args.trials - 1})")
    print(f"每次 {SEQUENCE_LENGTH} 帧, 帧间隔 {FRAME_INTERVAL_MS}ms\n")


def init_camera():
    """初始化摄像头：picamera2 优先，OpenCV 回退"""
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
            time.sleep(5.0)  # AWB 稳定
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
        print("摄像头: OpenCV")

    return picam2, cap, use_picam2


def read_frame(picam2, cap, use_picam2):
    """读取一帧（picamera2 返回 BGR888 需转 RGB，OpenCV 直接返回 BGR）"""
    if use_picam2:
        frame = picam2.capture_array("main")
        if frame is None:
            return False, None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return True, frame
    else:
        return cap.read()


# ── 主程序 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='树莓派手语测试集采集')
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS,
                        help=f'每种手势采集次数 (默认: {DEFAULT_TRIALS})')
    parser.add_argument('--start', type=int, default=0,
                        help='起始试次编号 (默认: 0)，用于断点续采')
    args = parser.parse_args()

    setup_directories(args)

    # ── 初始化摄像头 ──
    picam2, cap, use_picam2 = init_camera()

    # ── 初始化 Mediapipe ──
    print("加载 Mediapipe 手部+姿态检测器...")
    hand_detector, pose_detector = create_detectors()
    print("模型加载完毕，准备开始采集。\n")

    cv2.namedWindow('Test Data Collection', cv2.WINDOW_AUTOSIZE)

    # ── 三层循环: 手势 → 采集次数 → 帧 ──
    try:
        for action_idx, action in enumerate(ACTIONS):
            for trial in range(args.start, args.start + args.trials):
                frame_num = 0
                countdown_shown = False

                while frame_num < SEQUENCE_LENGTH:
                    # ---- 读取摄像头 ----
                    ret, frame = read_frame(picam2, cap, use_picam2)
                    if not ret:
                        print("警告: 摄像头读取失败，尝试继续...")
                        continue

                    # ---- Mediapipe 同步检测 ----
                    hand_result = detect_frame(frame, hand_detector)
                    pose_result = detect_pose(frame, pose_detector)

                    # ---- 绘制骨架 ----
                    draw_all_landmarks(frame, hand_result)
                    draw_shoulders(frame, pose_result)

                    # ---- 每轮开始时的倒计时提示 ----
                    if not countdown_shown:
                        h, w = frame.shape[:2]
                        put_chinese_text(frame, '准备采集',
                                         (w // 2 - 80, 180),
                                         font_size=36, color=(0, 255, 0), outline_color=(0, 0, 0))
                        info = f'手势: {action}  |  试次 #{trial}  |  {action_idx+1}/{len(ACTIONS)}'
                        put_chinese_text(frame, info,
                                         (15, 35), font_size=22,
                                         color=(0, 0, 255), outline_color=(0, 0, 0))
                        cv2.imshow('Test Data Collection', frame)
                        cv2.waitKey(1500)  # 1.5 秒准备时间
                        countdown_shown = True
                        continue  # 重新检测（倒计时后用户可能还没摆好手势）

                    # ---- 检查是否检测到手部 ----
                    if not hand_result.hand_landmarks:
                        h, w = frame.shape[:2]
                        put_chinese_text(frame, '未检测到手! 请调整位置',
                                         (w // 2 - 160, h - 60),
                                         font_size=28, color=(0, 0, 255), outline_color=(0, 0, 0))
                        info = f'手势: {action}  |  试次 #{trial}  |  等待手部...'
                        put_chinese_text(frame, info,
                                         (15, 35), font_size=22,
                                         color=(0, 0, 255), outline_color=(0, 0, 0))
                        cv2.imshow('Test Data Collection', frame)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            print("\n用户手动中止采集。")
                            return
                        continue  # 不保存，不递增帧数，重新尝试

                    # ---- 显示采集进度 ----
                    remaining_s = (SEQUENCE_LENGTH - frame_num) * FRAME_INTERVAL_MS / 1000.0
                    h, w = frame.shape[:2]
                    put_chinese_text(frame,
                                     f'手势: {action}  |  试次 #{trial}  '
                                     f'|  帧 {frame_num+1}/{SEQUENCE_LENGTH}  '
                                     f'(~{remaining_s:.1f}秒)',
                                     (15, 35), font_size=22,
                                     color=(0, 255, 0), outline_color=(0, 0, 0))

                    # 进度条
                    progress = frame_num / SEQUENCE_LENGTH
                    bar_w = int((w - 100) * progress)
                    cv2.rectangle(frame, (50, h - 45), (50 + bar_w, h - 30),
                                  (0, 255, 0), -1)
                    cv2.rectangle(frame, (50, h - 45), (w - 50, h - 30),
                                  (255, 255, 255), 1)

                    cv2.imshow('Test Data Collection', frame)

                    # ---- 提取并保存关键点 ----
                    keypoints = extract_keypoints(hand_result, pose_result)
                    npy_path = os.path.join(DATA_PATH, action, str(trial), str(frame_num))
                    np.save(npy_path, keypoints)
                    frame_num += 1  # 只有成功检测到手才递增

                    # ---- 固定帧间隔 ----
                    if cv2.waitKey(FRAME_INTERVAL_MS) & 0xFF == ord('q'):
                        print("\n用户手动中止采集。")
                        return

                # 每次采集完成提示
                print(f"  [{action}] 试次 #{trial} 完成")

            print(f"\n>>> 手势 '{action}' 全部 {args.trials} 次采集完成！\n")

        print("=" * 50)
        print("  测试数据采集完毕！")
        print(f"  数据路径: {DATA_PATH}")
        print("  下一步: 运行 evaluate_accuracy.py 评估准确率")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n采集被中断。")
    finally:
        if picam2:
            picam2.stop()
            picam2.close()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        hand_detector.close()
        pose_detector.close()
        print("资源已释放，退出。")


if __name__ == '__main__':
    main()
