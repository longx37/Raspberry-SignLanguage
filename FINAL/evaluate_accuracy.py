"""
evaluate_accuracy.py — 批量评估模型准确率
==========================================
加载 TestData/ 中的测试集，用 TFLite 模型批量推理，输出：
  - 总体准确率 / 各类别准确率
  - 混淆矩阵（控制台 + CSV）
  - 详细结果（JSON）

用法:
  python evaluate_accuracy.py                     # 评估全部 TestData
  python evaluate_accuracy.py --data TestData     # 指定数据目录
  python evaluate_accuracy.py --model action_int8.tflite  # 指定模型
"""

import numpy as np
import os
import sys
import json
import csv
import time
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import FEATURE_DIM

# ── 配置 ──────────────────────────────────────────────

ACTIONS = np.array(['你', '想', '买', '什么', '我', '看看', '那',
                    '衣服', '请', '问', '钱', '多少', '太贵了', '便宜点', '可以（吗）'])

SEQUENCE_LENGTH = 30
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TestData')
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'action_int8.tflite')
# 如果 tflite 不存在，自动找 action.keras 或同目录其他模型
if not os.path.exists(DEFAULT_MODEL_PATH):
    for candidate in ['action.keras', 'action.h5']:
        c = os.path.join(os.path.dirname(os.path.abspath(__file__)), candidate)
        if os.path.exists(c):
            DEFAULT_MODEL_PATH = c
            break


# ── 数据加载 ──────────────────────────────────────────

def load_test_data(data_dir):
    """加载 TestData/ 下所有 .npy 序列，返回 (X, y, meta)

    Returns:
        X: np.ndarray, shape (n_samples, 30, 135)
        y: np.ndarray, shape (n_samples,), integer class indices
        meta: list of dict, 每个样本的元信息
    """
    X_list = []
    y_list = []
    meta_list = []

    if not os.path.isdir(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        sys.exit(1)

    for action in ACTIONS:
        action_dir = os.path.join(data_dir, action)
        if not os.path.isdir(action_dir):
            print(f"  跳过: 未找到 '{action}' 目录")
            continue

        for trial_name in sorted(os.listdir(action_dir),
                                 key=lambda x: int(x) if x.isdigit() else 0):
            trial_dir = os.path.join(action_dir, trial_name)
            if not os.path.isdir(trial_dir):
                continue

            # 加载该试次的所有帧
            sequence = []
            frames_found = 0
            for frame_idx in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(trial_dir, f'{frame_idx}.npy')
                if os.path.exists(npy_path):
                    kp = np.load(npy_path)
                    sequence.append(kp)
                    frames_found += 1
                else:
                    # 缺失帧填充零
                    sequence.append(np.zeros(FEATURE_DIM, dtype=np.float32))

            # 检查实际帧数
            if frames_found == 0:
                print(f"  跳过 [{action}]/{trial_name}: 无任何帧数据")
                continue
            elif frames_found < SEQUENCE_LENGTH:
                print(f"  注意 [{action}]/{trial_name}: "
                      f"仅 {frames_found}/{SEQUENCE_LENGTH} 帧")

            X_list.append(np.stack(sequence, axis=0))
            y_list.append(np.where(ACTIONS == action)[0][0])
            meta_list.append({
                'gesture': action,
                'trial': trial_name,
                'frames_found': frames_found,
            })

    if not X_list:
        print("错误: TestData/ 中没有找到任何有效数据。请先运行 collect_testset.py")
        sys.exit(1)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"\n加载完成: {X.shape[0]} 个样本, {X.shape[1]} 帧, {X.shape[2]} 维")
    class_counts = defaultdict(int)
    for m in meta_list:
        class_counts[m['gesture']] += 1
    print(f"各类别样本数: {dict(class_counts)}")

    return X, y, meta_list


# ── 模型推理 ──────────────────────────────────────────

def load_model(model_path):
    """加载模型（TFLite 或 Keras），返回统一的 model_dict

    Returns:
        dict: {'type': 'tflite'|'keras', 'interpreter'?, 'keras_model'?,
               'input_details'?, 'output_details'?, 'num_classes'}
    """
    import tensorflow as tf

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print(f"  当前目录: {os.getcwd()}")
        print(f"  FINAL/: {os.path.dirname(os.path.abspath(__file__))}")
        nearby = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__)))
                  if f.endswith(('.tflite', '.keras', '.h5'))]
        if nearby:
            print(f"  同目录下的模型文件: {nearby}")
        sys.exit(1)

    ext = os.path.splitext(model_path)[1].lower()

    # ── 路径 1: TFLite 模型 ──
    if ext == '.tflite':
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
        except Exception as e:
            raise RuntimeError(
                f"TFLite 模型加载失败，文件可能损坏:\n"
                f"  文件: {model_path}\n"
                f"  大小: {os.path.getsize(model_path)} 字节\n"
                f"  错误: {e}\n"
                f"  请用二进制模式重新传输模型文件。"
            ) from e

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        num_classes = output_details[0]['shape'][-1]

        # 热身
        dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()

        print(f"模型加载成功: {model_path} (TFLite)")
        print(f"  输入 shape: {input_details[0]['shape']}")
        print(f"  输出 shape: {output_details[0]['shape']}")
        print(f"  预热完成\n")

        return {
            'type': 'tflite',
            'interpreter': interpreter,
            'input_details': input_details,
            'output_details': output_details,
            'num_classes': num_classes,
        }

    # ── 路径 2: Keras 模型 (.keras / .h5) ──
    elif ext in ('.keras', '.h5'):
        keras_model = tf.keras.models.load_model(model_path)
        num_classes = keras_model.output_shape[-1]

        # 热身
        dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        _ = keras_model.predict(dummy, verbose=0)

        print(f"模型加载成功: {model_path} (Keras)")
        print(f"  输入 shape: {keras_model.input_shape}")
        print(f"  输出 shape: {keras_model.output_shape}")
        print(f"  预热完成\n")

        return {
            'type': 'keras',
            'keras_model': keras_model,
            'num_classes': num_classes,
        }

    else:
        print(f"错误: 不支持的模型格式 '{ext}'，请使用 .tflite / .keras / .h5")
        sys.exit(1)


def run_inference(model, X):
    """批量推理，返回 (predicted_indices, confidences, all_probabilities, inference_times_ms)"""
    n = X.shape[0]
    num_classes = model['num_classes']
    predicted_indices = np.zeros(n, dtype=np.int32)
    confidences = np.zeros(n, dtype=np.float32)
    all_probs = np.zeros((n, num_classes), dtype=np.float32)
    inference_times = np.zeros(n, dtype=np.float32)

    for i in range(n):
        input_data = np.expand_dims(X[i], axis=0).astype(np.float32)

        t0 = time.perf_counter()

        if model['type'] == 'tflite':
            interpreter = model['interpreter']
            interpreter.set_tensor(model['input_details'][0]['index'], input_data)
            interpreter.invoke()
            predictions = interpreter.get_tensor(
                model['output_details'][0]['index'])[0]
        elif model['type'] == 'keras':
            predictions = model['keras_model'].predict(input_data, verbose=0)[0]
        else:
            raise RuntimeError(f"未知模型类型: {model['type']}")

        elapsed = (time.perf_counter() - t0) * 1000

        predicted_indices[i] = np.argmax(predictions)
        confidences[i] = predictions[predicted_indices[i]]
        all_probs[i] = predictions
        inference_times[i] = elapsed

    return predicted_indices, confidences, all_probs, inference_times


# ── 统计计算 ──────────────────────────────────────────

def compute_statistics(y_true, y_pred, confidences, inference_times, actions, meta_list):
    """计算所有评估指标"""
    n = len(y_true)
    correct = (y_true == y_pred)
    overall_accuracy = correct.mean()

    # 每类准确率
    per_class = {}
    for idx, action in enumerate(actions):
        mask = y_true == idx
        if mask.sum() > 0:
            per_class[action] = {
                'accuracy': correct[mask].mean(),
                'correct': int(correct[mask].sum()),
                'total': int(mask.sum()),
            }
        else:
            per_class[action] = {'accuracy': None, 'correct': 0, 'total': 0}

    # 混淆矩阵
    n_classes = len(actions)
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        if t < n_classes and p < n_classes:
            cm[t, p] += 1

    # 平均置信度
    avg_conf_correct = confidences[correct].mean() if correct.sum() > 0 else 0.0
    avg_conf_wrong = confidences[~correct].mean() if (~correct).sum() > 0 else 0.0

    stats = {
        'n_samples': n,
        'n_correct': int(correct.sum()),
        'n_wrong': int((~correct).sum()),
        'overall_accuracy': float(overall_accuracy),
        'per_class_accuracy': per_class,
        'confusion_matrix': cm.tolist(),
        'avg_confidence_correct': float(avg_conf_correct),
        'avg_confidence_wrong': float(avg_conf_wrong),
        'avg_confidence_overall': float(confidences.mean()),
        'avg_inference_time_ms': float(inference_times.mean()),
        'min_inference_time_ms': float(inference_times.min()),
        'max_inference_time_ms': float(inference_times.max()),
    }
    return stats


# ── 格式化输出 ────────────────────────────────────────

def print_report(stats, actions, meta_list):
    """终端美观报告"""
    print("\n" + "=" * 60)
    print("  手语识别模型准确率评估报告")
    print("=" * 60)

    print(f"\n  测试样本数:   {stats['n_samples']}")
    print(f"  正确:         {stats['n_correct']}")
    print(f"  错误:         {stats['n_wrong']}")
    print(f"  总体准确率:   {stats['overall_accuracy']:.2%}")

    # 各类别准确率
    print(f"\n  {'─' * 40}")
    print(f"  {'手势':<12s} {'准确率':>8s}  {'样本数':>6s}")
    print(f"  {'─' * 40}")
    for action in actions:
        info = stats['per_class_accuracy'][action]
        if info['total'] > 0:
            bar = '█' * int(info['accuracy'] * 10)
            pct = f"{info['accuracy']:.0%}"
            print(f"  {action:<12s} {pct:>6s}  {bar:<10s}  {info['total']:>3d}")
        else:
            print(f"  {action:<12s} {'—':>6s}  {'(无数据)':<10s}")
    print(f"  {'─' * 40}")

    # 混淆矩阵（ASCII）
    print(f"\n  混淆矩阵 (行=实际, 列=预测)")
    print(f"  {'':>6s}", end="")
    for a in actions:
        print(f"{a:>4s}", end="")
    print()

    cm = np.array(stats['confusion_matrix'])
    for i, a in enumerate(actions):
        print(f"  {a:>6s}", end="")
        for j in range(len(actions)):
            val = cm[i, j]
            if val > 0:
                print(f"{val:>4d}", end="")
            else:
                print(f"{'·':>4s}", end="")
        # 行合计
        row_sum = cm[i].sum()
        print(f"  | {row_sum}")

    # 列合计
    print(f"  {'':>6s}", end="")
    for j in range(len(actions)):
        print(f"{cm[:, j].sum():>4d}", end="")
    print()

    # 推理性能
    print(f"\n  推理性能:")
    print(f"    平均: {stats['avg_inference_time_ms']:.1f} ms")
    print(f"    最小: {stats['min_inference_time_ms']:.1f} ms")
    print(f"    最大: {stats['max_inference_time_ms']:.1f} ms")

    # 置信度
    print(f"\n  置信度分析:")
    print(f"    总体平均:       {stats['avg_confidence_overall']:.2%}")
    print(f"    正确预测时平均: {stats['avg_confidence_correct']:.2%}")
    print(f"    错误预测时平均: {stats['avg_confidence_wrong']:.2%}")

    # 最易混淆的手势对 (off-diagonal top N)
    print(f"\n  最易混淆的 5 对手势:")
    n_classes = len(actions)
    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], actions[i], actions[j]))
    pairs.sort(reverse=True)
    for count, gt, pred in pairs[:5]:
        print(f"    {gt} → {pred}: {count} 次")

    print("\n" + "=" * 60)


def save_results(stats, meta_list, y_true, y_pred, all_probs, actions, output_dir):
    """保存 JSON 和 CSV 结果文件"""
    os.makedirs(output_dir, exist_ok=True)

    # ── JSON ──
    json_path = os.path.join(output_dir, 'accuracy_results.json')

    # 把 meta 信息一并写入（转换为可序列化格式）
    trials_output = []
    for i, (t, p, conf, meta) in enumerate(
            zip(y_true, y_pred,
                [all_probs[i, p] for i, p in enumerate(y_pred)],
                meta_list)):
        trials_output.append({
            'ground_truth': actions[int(t)],
            'ground_truth_idx': int(t),
            'predicted': actions[int(p)],
            'predicted_idx': int(p),
            'confidence': float(conf),
            'correct': int(t) == int(p),
            **meta,
        })

    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gestures': list(actions),
        'statistics': stats,
        'trials': trials_output,
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到: {json_path}")

    # ── CSV 混淆矩阵 ──
    cm_csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
    cm = np.array(stats['confusion_matrix'])
    with open(cm_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['实际\\预测'] + list(actions))
        for i, a in enumerate(actions):
            writer.writerow([a] + list(cm[i]))
    print(f"混淆矩阵 CSV 已保存到: {cm_csv_path}")

    # ── CSV 逐试次详情 ──
    detail_csv_path = os.path.join(output_dir, 'trial_details.csv')
    with open(detail_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['手势', '试次', '实际类别', '预测类别',
                         '置信度', '正确', '有效帧数'])
        for trial in trials_output:
            writer.writerow([
                trial['gesture'], trial['trial'],
                trial['ground_truth'], trial['predicted'],
                f"{trial['confidence']:.4f}",
                'Yes' if trial['correct'] else 'No',
                trial.get('frames_found', '?'),
            ])
    print(f"逐试次详情已保存到: {detail_csv_path}")


# ── 主程序 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='批量评估手语识别模型准确率')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_DIR,
                        help=f'测试数据目录 (默认: {DEFAULT_DATA_DIR})')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'模型路径 (.tflite / .keras / .h5) (默认: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--output', type=str, default=None,
                        help='结果输出目录 (默认: 数据目录)')
    args = parser.parse_args()

    # ── 加载数据 ──
    print("=" * 60)
    print("  手语识别准确率评估")
    print("=" * 60)
    print(f"\n数据目录: {args.data}")
    print(f"模型路径: {args.model}")

    X, y, meta_list = load_test_data(args.data)

    # ── 加载模型 ──
    model = load_model(args.model)

    if model['num_classes'] != len(ACTIONS):
        print(f"警告: 模型输出 {model['num_classes']} 类，"
              f"但 ACTIONS 定义了 {len(ACTIONS)} 个")

    # ── 批量推理 ──
    print(f"开始批量推理 ({X.shape[0]} 个样本)...")
    t_start = time.perf_counter()
    y_pred, confidences, all_probs, inference_times = run_inference(model, X)
    total_time = (time.perf_counter() - t_start) * 1000
    print(f"推理完成: 总耗时 {total_time:.0f} ms, "
          f"平均 {inference_times.mean():.1f} ms/样本")

    # ── 计算统计 ──
    stats = compute_statistics(y, y_pred, confidences, inference_times,
                               ACTIONS, meta_list)

    # ── 输出报告 ──
    print_report(stats, ACTIONS, meta_list)

    # ── 保存结果 ──
    output_dir = args.output if args.output else args.data
    save_results(stats, meta_list, y, y_pred, all_probs, ACTIONS, output_dir)


if __name__ == '__main__':
    main()
