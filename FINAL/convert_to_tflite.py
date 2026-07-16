"""
模型量化转换: .keras → TFLite (动态范围量化)
==========================================
在 PC 上运行，生成 .tflite 文件部署到树莓派。

用法: python convert_to_tflite.py
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'action.keras')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'action_int8.tflite')


def main():
    print("=" * 50)
    print("  Keras → TFLite 动态范围量化")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型不存在: {MODEL_PATH}")
        sys.exit(1)

    print(f"\n[1/3] 加载模型: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print(f"\n[2/3] 转换中...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    print(f"\n[3/3] 保存: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)

    keras_size = os.path.getsize(MODEL_PATH)
    tflite_size = os.path.getsize(OUTPUT_PATH)
    print(f"\n完成!")
    print(f"  原始  : {keras_size / 1024:.0f} KB")
    print(f"  量化后: {tflite_size / 1024:.0f} KB")
    print(f"  压缩比: {keras_size / tflite_size:.1f}x")
    print(f"\n部署到树莓派:")
    print(f"  scp {OUTPUT_PATH} longx@raspberry:~/sign_language/")


if __name__ == '__main__':
    main()
