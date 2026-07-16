"""
01_采集数据集.py（纯手部版本）
==============================
使用摄像头实时采集手部关键点数据，保存为 .npy 文件。
仅检测双手 21 个关键点 (x, y, z)，每帧 126 维特征。

文件结构:
  MP_Data/
    ├── 手势名1/
    │   ├── 0/        ← 第 0 次采集
    │   │   ├── 0.npy  ← 第 0 帧
    │   │   ├── 1.npy
    │   │   └── ...   (共 30 帧)
    │   ├── 1/
    │   └── ...       (共 30 次)
    └── 手势名2/
        └── ...

用法:
  conda activate hand
  python 01_采集数据集.py
  python 01_采集数据集.py --start 30   # 从第 30 号开始采集，不会覆盖之前的

操作说明:
  - 每轮采集前有 2 秒准备倒计时
  - 采集过程中请在摄像头前稳定做出目标手势
  - 按 'q' 可随时中止采集
  - 修改下方 ACTIONS / NO_SEQUENCES / SEQUENCE_LENGTH 可调整采集参数
"""

import cv2
import numpy as np
import os
import sys
import argparse

# 确保能找到同目录的 utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    create_detectors, detect_frame, detect_pose, extract_keypoints,
    draw_all_landmarks, draw_shoulders, FEATURE_DIM,
)

# ══════════════════════════════════════════════════════════════
#  可修改的采集参数
# ══════════════════════════════════════════════════════════════

# 训练的手势名称（即 label / action）
ACTIONS = np.array(['你', '想', '买', '什么', '我', '看看', '那',
                    '衣服', '请', '问', '钱', '多少', '太贵了', '便宜点', '可以（吗）'])

# 每种手势采集的次数（即 sequence / video 数量）
NO_SEQUENCES = 30

# 每次采集的帧数（即 sequence length）
SEQUENCE_LENGTH = 30

# 采集帧间隔（毫秒）：每两帧之间的固定延迟
# 30 帧 × 67ms ≈ 2 秒完成一轮采集，给你充足的反应时间
FRAME_INTERVAL_MS = 67

# 手部重新检测到后的准备延迟（毫秒）：避免手一回到画面就立刻采集
REACQUIRE_DELAY_MS = 300

# 数据保存目录
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MP_Data')


# ══════════════════════════════════════════════════════════════
#  创建目录结构
# ══════════════════════════════════════════════════════════════

def setup_directories(start_seq=0):
    """为每种手势 × 每次采集创建目录"""
    for action in ACTIONS:
        for sequence in range(start_seq, start_seq + NO_SEQUENCES):
            dir_path = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(dir_path, exist_ok=True)
    print(f"数据目录已就绪: {DATA_PATH}")
    print(f"手势类别: {list(ACTIONS)}")
    print(f"每种手势 {NO_SEQUENCES} 次采集 (编号 {start_seq}~{start_seq + NO_SEQUENCES - 1}) × {SEQUENCE_LENGTH} 帧")


# ══════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════

def main():
    # ── 命令行参数 ──
    parser = argparse.ArgumentParser(description='手语数据采集')
    parser.add_argument('--start', type=int, default=0,
                        help=f'起始采集序号 (默认 0)，每次采集 {NO_SEQUENCES} 轮')
    args = parser.parse_args()
    start_seq = args.start

    # ── 初始化目录 ──
    setup_directories(start_seq)

    # ── 初始化摄像头 ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头。请检查摄像头连接。")
        sys.exit(1)

    # ── 初始化 Mediapipe ──
    print("正在加载 Mediapipe 手部+姿态检测模型...")
    hand_detector, pose_detector = create_detectors()
    print("模型加载完毕，准备开始采集。\n")

    # ── 三层循环: 手势 → 采集次数 → 帧 ──
    try:
        for action_idx, action in enumerate(ACTIONS):
            for sequence in range(start_seq, start_seq + NO_SEQUENCES):
                frame_num = 0
                countdown_shown = False
                hand_was_lost = False       # 标记手部是否刚丢失→恢复
                while frame_num < SEQUENCE_LENGTH:
                    # ---- 读取摄像头 ----
                    ret, frame = cap.read()
                    if not ret:
                        print("警告: 摄像头读取失败，尝试继续...")
                        continue

                    # ---- Mediapipe 检测 ----
                    hand_result = detect_frame(frame, hand_detector)
                    pose_result = detect_pose(frame, pose_detector)

                    # ---- 绘制骨架 ----
                    draw_all_landmarks(frame, hand_result)
                    draw_shoulders(frame, pose_result)

                    # ---- 每轮开始时的倒计时提示 ----
                    if not countdown_shown:
                        cv2.putText(frame, 'STARTING COLLECTION',
                                    (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame,
                                    f'Collecting: {action}  |  Video #{sequence}',
                                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('Data Collection', frame)
                        cv2.waitKey(1500)  # 2 秒准备时间
                        countdown_shown = True
                        continue  # 重新检测（倒计时后用户可能还没摆好手势）

                    # ---- 检查是否检测到手部 ----
                    if not hand_result.hand_landmarks:
                        hand_was_lost = True  # 标记：手部曾丢失
                        cv2.putText(frame,
                                    'NO HAND DETECTED! Please adjust position!',
                                    (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.putText(frame,
                                    f'Collecting: {action}  |  Video #{sequence}  |  Waiting for hand...',
                                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('Data Collection', frame)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            print("\n用户手动中止采集。")
                            cap.release()
                            cv2.destroyAllWindows()
                            hand_detector.close()
                            pose_detector.close()
                            return
                        continue  # 不保存，不递增帧数，重新尝试

                    # ---- 手部刚恢复时的缓冲（给你重新摆好手势的时间） ----
                    if hand_was_lost:
                        hand_was_lost = False
                        # 倒计时提示
                        for countdown_sec in range(3, 0, -1):
                            ret2, frame2 = cap.read()
                            if not ret2:
                                continue
                            hand_result2 = detect_frame(frame2, hand_detector)
                            pose_result2 = detect_pose(frame2, pose_detector)
                            draw_all_landmarks(frame2, hand_result2)
                            draw_shoulders(frame2, pose_result2)
                            if not hand_result2.hand_landmarks:
                                # 倒计时期间手又丢了，跳出重新等待
                                hand_was_lost = True
                                break
                            cv2.putText(frame2,
                                        f'Hand back! Resuming in {countdown_sec}...',
                                        (80, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (0, 255, 255), 3, cv2.LINE_AA)
                            cv2.putText(frame2,
                                        f'Collecting: {action}  |  Video #{sequence}  |  Frame {frame_num+1}/{SEQUENCE_LENGTH}',
                                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.imshow('Data Collection', frame2)
                            if cv2.waitKey(1000) & 0xFF == ord('q'):
                                print("\n用户手动中止采集。")
                                cap.release()
                                cv2.destroyAllWindows()
                                hand_detector.close()
                                pose_detector.close()
                                return
                        if hand_was_lost:
                            continue  # 倒计时期间手又丢了，回到等待状态

                    # ---- 显示采集进度 ----
                    remaining_s = (SEQUENCE_LENGTH - frame_num) * FRAME_INTERVAL_MS / 1000.0
                    cv2.putText(frame,
                                f'Collecting: {action}  |  Video #{sequence}  |  Frame {frame_num+1}/{SEQUENCE_LENGTH}  (~{remaining_s:.1f}s left)',
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Data Collection', frame)

                    # ---- 提取并保存关键点 ----
                    keypoints = extract_keypoints(hand_result, pose_result)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    frame_num += 1  # 只有成功检测到手才递增

                    # ---- 固定帧间隔（控制采集速度，不随运行速度变化） ----
                    if cv2.waitKey(FRAME_INTERVAL_MS) & 0xFF == ord('q'):
                        print("\n用户手动中止采集。")
                        cap.release()
                        cv2.destroyAllWindows()
                        hand_detector.close()
                        pose_detector.close()
                        return

                # 每次采集完成提示
                print(f"  [{action}] 采集 #{sequence} 完成")

            print(f"\n>>> 手势 '{action}' 全部 {NO_SEQUENCES} 次采集完成！\n")

        print("=" * 50)
        print("  所有数据采集完毕！")
        print(f"  数据路径: {DATA_PATH}")
        print("  下一步: 运行 02_训练模型.py 开始训练")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n采集被中断。")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_detector.close()
        pose_detector.close()


if __name__ == '__main__':
    main()
