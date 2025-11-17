from pathlib import Path
import openvino as ov
import nncf

# --- 1. 配置您的模型路径 ---
# 注意：ONNX模型应该是FP32精度的
ONNX_MODEL_PATH = Path("D:\Code\OpenVINOCode\Model\yolov8n\yolov8n.onnx") # <<< 确保这里是原始的FP32模型路径
FP16_MODEL_PATH = Path("D:\Code\OpenVINOCode\Model\yolov8n\yolov8n_fp16_nncf.xml")

# --- 2. 主执行函数 (已修改为仅转换和保存) ---
def main():
    """
    加载一个FP32 ONNX模型，将其权重转换为FP16，并保存为OpenVINO IR格式。
    """
    print("--- OpenVINO FP16 权重压缩工具 ---")

    if not ONNX_MODEL_PATH.exists():
        print(f"错误: ONNX模型文件未找到: {ONNX_MODEL_PATH}")
        return
        
    # --- 定义输出路径 ---
    output_ir_path = FP16_MODEL_PATH

    try:
        core = ov.Core()
        
        # --- 加载原始FP32模型 ---
        print(f"[1/3] 正在加载FP32模型: {ONNX_MODEL_PATH}")
        model_fp32 = core.read_model(ONNX_MODEL_PATH)
        
        # --- 将模型权重压缩为FP16 ---
        print(f"[2/3] 正在将模型权重压缩为FP16...")
        # 这个过程很快，因为它只改变权重的数据类型，不涉及校准
        model_fp16 = nncf.compress_weights(model_fp32)
        
        # --- 保存转换后的FP16模型 ---
        # ov.serialize 会自动创建 .xml 和 .bin 两个文件
        print(f"[3/3] 正在保存FP16模型到: {output_ir_path}")
        ov.serialize(model_fp16, str(output_ir_path))
        
        print("\n转换成功!")
        print(f"  - XML 文件: {output_ir_path}")
        print(f"  - BIN 文件: {output_ir_path.with_suffix('.bin')}")

    except Exception as e:
        print(f"\n在处理过程中发生错误: {e}")


if __name__ == "__main__":
    main()