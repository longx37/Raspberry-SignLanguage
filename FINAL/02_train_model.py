"""
02_训练模型.py（手部 + 身体姿态版本）
============================
加载采集好的 .npy 数据，搭建 LSTM 时序神经网络进行手势分类训练。

网络结构:
  LSTM(32, tanh) → Dropout → LSTM(64, tanh) → Dropout
  → LSTM(32, tanh) → Dropout → Dense(32, relu) → Dropout
  → Dense(16, relu) → Dense(num_actions, softmax)

输入维度: (30 帧, 135 特征/帧)
  135 = 21×3 (left hand) + 21×3 (right hand) + 3×3 (nose + shoulders)

用法:
  conda activate hand
  python 02_训练模型.py

  tensorboard --logdir Logs   # 可选：查看训练曲线
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# matplotlib 非交互后端 + 中文字体配置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
from utils import FEATURE_DIM

# ══════════════════════════════════════════════════════════════
#  可修改参数
# ══════════════════════════════════════════════════════════════

# 手势名称 —— 必须与采集脚本中的 ACTIONS 保持一致
ACTIONS = np.array(['你',
    '想',
    '买',
    '什么',
    '我',
    '看看',
    '那',
    '衣服',
    '请',
    '问',
    '钱',
    '多少',
    '太贵了',
    '便宜点',
    '可以（吗）'])

# 采集参数 —— 必须与采集脚本中的设置一致
NO_SEQUENCES = 90
SEQUENCE_LENGTH = 30

# 数据目录
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MP_Data')

# 训练参数
EPOCHS = 500                 # 最大训练轮数（EarlyStopping 会自动提前停止）
BATCH_SIZE = 8               # 批次大小
AUGMENTATION_FACTOR = 2      # 数据增强倍数（每个样本生成几个增强副本，0=不增强）

# 数据集划分 —— 按序列编号拆分（三人采集，每人30段，避免同人数据泄露）
TRAIN_SEQ_START = 0          # 训练集起始编号
TRAIN_SEQ_END = 59           # 训练集结束编号（包含）
VAL_SEQ_START = 60           # 验证集起始编号
VAL_SEQ_END = 89             # 验证集结束编号（包含）

# 输出
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Logs')
MODEL_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'action.keras')


# ══════════════════════════════════════════════════════════════
#  数据增强
# ══════════════════════════════════════════════════════════════

def augment_sequence(sequence, scale_range=0.3, rotation_range=15):
    """对单个关键点序列施加随机缩放和旋转增强

    在所有帧上应用相同的变换参数，保持时序一致性。
    仅变换 x, y 坐标；z（深度）保持不变。
    以 (0.5, 0.5) 为变换中心（归一化图像坐标系中心）。

    Args:
        sequence:    numpy array, shape = (30, 135)
        scale_range: 缩放幅度，0.3 表示 0.7× ∼ 1.3×
        rotation_range: 旋转幅度（度），15 表示 -15° ∼ +15°

    Returns:
        augmented: numpy array, shape = (30, 135)
    """
    # 随机缩放因子: 1.0 ± scale_range
    scale = 1.0 + np.random.uniform(-scale_range, scale_range)

    # 随机旋转角度（弧度）
    angle_deg = np.random.uniform(-rotation_range, rotation_range)
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    augmented = sequence.copy()

    for t in range(sequence.shape[0]):
        # 重塑为 (45, 3) —— 每行一个关键点的 (x, y, z)
        frame = sequence[t].reshape(-1, 3)

        # 以 (0.5, 0.5) 为中心
        x = frame[:, 0] - 0.5
        y = frame[:, 1] - 0.5

        # 缩放
        x_scaled = x * scale
        y_scaled = y * scale

        # 旋转
        x_rot = x_scaled * cos_a - y_scaled * sin_a
        y_rot = x_scaled * sin_a + y_scaled * cos_a

        # 平移回原坐标系，clip 到有效范围
        frame[:, 0] = np.clip(x_rot + 0.5, 0.0, 1.0)
        frame[:, 1] = np.clip(y_rot + 0.5, 0.0, 1.0)

        augmented[t] = frame.flatten()

    return augmented


def augment_dataset(X, y, augmentation_factor=2):
    """对数据集施加数据增强，生成额外样本

    每个原始样本生成 augmentation_factor 个增强副本。
    增强方式: 随机缩放 (±30%) + 随机旋转 (±15°)。

    Args:
        X: 原始特征, shape = (N, 30, 135)
        y: 原始标签 (one-hot), shape = (N, num_classes)
        augmentation_factor: 每个样本生成的增强副本数

    Returns:
        X_aug: 增强后的特征, shape = (N*(1+factor), 30, 135)
        y_aug: 增强后的标签, shape = (N*(1+factor), num_classes)
    """
    X_list = [X]
    y_list = [y]

    for _ in range(augmentation_factor):
        aug_batch = np.array([augment_sequence(seq) for seq in X])
        X_list.append(aug_batch)
        y_list.append(y)

    X_aug = np.concatenate(X_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)

    print(f"\n数据增强完成:")
    print(f"  原始样本: {X.shape[0]}")
    print(f"  增强倍数: {augmentation_factor}×")
    print(f"  增强后总样本: {X_aug.shape[0]}")
    return X_aug, y_aug


# ══════════════════════════════════════════════════════════════
#  加载数据
# ══════════════════════════════════════════════════════════════

def load_data():
    """从 MP_Data 目录加载所有 .npy 文件，构建训练数据集

    Returns:
        X: numpy array, shape = (num_samples, 30, 135)
        y: numpy array, shape = (num_samples, num_actions)  one-hot 编码
    """
    label_map = {label: num for num, label in enumerate(ACTIONS)}

    sequences, labels = [], []

    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            raise FileNotFoundError(
                f"数据目录不存在: {action_path}\n"
                f"请先运行 01_采集数据集.py 采集数据！"
            )

        for sequence in range(NO_SEQUENCES):
            window = []
            seq_path = os.path.join(DATA_PATH, action, str(sequence))

            if not os.path.exists(seq_path):
                print(f"  警告: 缺失 {seq_path}，跳过")
                continue

            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(seq_path, f"{frame_num}.npy")
                res = np.load(frame_path)
                window.append(res)

            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(ACTIONS)).astype(int)

    print(f"数据集加载完成:")
    print(f"  X 形状: {X.shape}  (样本数, 帧数, 特征维度)")
    print(f"  y 形状: {y.shape}  (样本数, 类别数)")
    print(f"  各类别分布: {np.sum(y, axis=0).astype(int)}")
    return X, y


# ══════════════════════════════════════════════════════════════
#  搭建 LSTM 模型
# ══════════════════════════════════════════════════════════════

def build_model(num_actions):
    """搭建三层 LSTM + Dropout + 两层全连接的手势分类模型（防过拟合精简版）

    Args:
        num_actions: 分类类别数

    Returns:
        编译好的 Keras Sequential 模型
    """
    model = Sequential([
        LSTM(32, return_sequences=True,
             input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(num_actions, activation='softmax'),
    ])

    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy',
                 Precision(name='precision'),
                 Recall(name='recall')]
    )

    return model


# ══════════════════════════════════════════════════════════════
#  绘制训练指标曲线
# ══════════════════════════════════════════════════════════════

def plot_training_curves(history, output_dir):
    """生成训练指标曲线图：损失、准确率、F1、精确率（2×2 网格）

    Args:
        history: model.fit() 返回的 History 对象
        output_dir: 图片输出目录
    """
    epochs = range(1, len(history.history['loss']) + 1)

    # ── 从 Precision/Recall 计算 F1 ──
    train_precision = history.history['precision']
    val_precision = history.history['val_precision']
    train_recall = history.history['recall']
    val_recall = history.history['val_recall']

    train_f1 = [2 * p * r / (p + r + 1e-8)
                for p, r in zip(train_precision, train_recall)]
    val_f1 = [2 * p * r / (p + r + 1e-8)
              for p, r in zip(val_precision, val_recall)]

    # ── 创建 2×2 画布 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上：损失曲线
    axes[0, 0].plot(epochs, history.history['loss'],
                    'b-', label='训练损失', linewidth=1.5)
    axes[0, 0].plot(epochs, history.history['val_loss'],
                    'r-', label='验证损失', linewidth=1.5)
    axes[0, 0].set_title('损失曲线 (Loss)', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 右上：准确率曲线
    axes[0, 1].plot(epochs, history.history['categorical_accuracy'],
                    'b-', label='训练准确率', linewidth=1.5)
    axes[0, 1].plot(epochs, history.history['val_categorical_accuracy'],
                    'r-', label='验证准确率', linewidth=1.5)
    axes[0, 1].set_title('准确率曲线 (Accuracy)', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 左下：F1 分数曲线
    axes[1, 0].plot(epochs, train_f1, 'b-',
                    label='训练 F1', linewidth=1.5)
    axes[1, 0].plot(epochs, val_f1, 'r-',
                    label='验证 F1', linewidth=1.5)
    axes[1, 0].set_title('F1 分数曲线 (F1 Score)', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 右下：精确率曲线
    axes[1, 1].plot(epochs, train_precision,
                    'b-', label='训练精确率', linewidth=1.5)
    axes[1, 1].plot(epochs, val_precision,
                    'r-', label='验证精确率', linewidth=1.5)
    axes[1, 1].set_title('精确率曲线 (Precision)', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # ── 保存 ──
    output_path = os.path.join(output_dir, 'training_metrics.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n训练指标曲线已保存到: {output_path}")

    # ── 打印最终指标 ──
    print(f"\n{'='*50}")
    print(f"  最终验证指标")
    print(f"{'='*50}")
    print(f"  损失:     {history.history['val_loss'][-1]:.4f}")
    print(f"  准确率:   {history.history['val_categorical_accuracy'][-1]:.4f}")
    print(f"  精确率:   {val_precision[-1]:.4f}")
    print(f"  召回率:   {val_recall[-1]:.4f}")
    print(f"  F1 分数:  {val_f1[-1]:.4f}")


# ══════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════

def main():
    # ── 加载数据 ──
    print("=" * 50)
    print("  加载数据...")
    print("=" * 50)
    X, y = load_data()

    # ── 按序列编号划分训练/验证集 ──
    # load_data 按 [action0_seq0..89, action1_seq0..89, ...] 顺序存储
    # 每人采集 30 段 (0-29, 30-59, 60-89)，这样划分保证不同人的数据不会跨集泄露
    seq_indices = np.tile(np.arange(NO_SEQUENCES), len(ACTIONS))
    train_mask = (seq_indices >= TRAIN_SEQ_START) & (seq_indices <= TRAIN_SEQ_END)
    val_mask = (seq_indices >= VAL_SEQ_START) & (seq_indices <= VAL_SEQ_END)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"\n训练集 (序列 {TRAIN_SEQ_START}-{TRAIN_SEQ_END}): {X_train.shape[0]} 样本")
    print(f"验证集 (序列 {VAL_SEQ_START}-{VAL_SEQ_END}): {X_val.shape[0]} 样本")

    # ── 数据增强（仅对训练集）──
    if AUGMENTATION_FACTOR > 0:
        X_train, y_train = augment_dataset(
            X_train, y_train, augmentation_factor=AUGMENTATION_FACTOR
        )

    # ── 搭建模型 ──
    print("\n" + "=" * 50)
    print("  搭建 LSTM 模型...")
    print("=" * 50)
    model = build_model(len(ACTIONS))
    model.summary()

    # ── 回调 ──
    callbacks = [
        TensorBoard(log_dir=LOG_DIR),
        ModelCheckpoint(
            MODEL_OUTPUT,
            monitor='val_categorical_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_categorical_accuracy',
            patience=80,
            mode='max',
            verbose=1,
            restore_best_weights=True,
        ),
    ]

    # ── 训练 ──
    print("\n" + "=" * 50)
    print(f"  开始训练 ({EPOCHS} epochs)...")
    print(f"  TensorBoard: tensorboard --logdir {LOG_DIR}")
    print("=" * 50)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    # ── 最终保存 ──
    model.save(MODEL_OUTPUT)
    print(f"\n模型已保存到: {MODEL_OUTPUT}")

    # ── 绘制训练指标曲线 ──
    plot_training_curves(history, os.path.dirname(os.path.abspath(__file__)))

    # ── 结果摘要 ──
    train_acc = history.history['categorical_accuracy'][-1]
    val_acc = history.history['val_categorical_accuracy'][-1]
    print(f"\n最终训练准确率: {train_acc:.4f}")
    print(f"最终验证准确率: {val_acc:.4f}")
    print(f"实际训练轮数: {len(history.history['loss'])}")
    print(f"\n下一步: 运行 03_实时检测.py 使用模型进行实时手势识别")


if __name__ == '__main__':
    main()
