import openvino as ov
from pathlib import Path

# --- 1. 配置您的路径 ---
# 指向您想要转换的 OpenVINO IR 模型的 .xml 文件
ir_model_path = Path("D:\Code\OpenVINOCode\Model\yolov8n\yolov8n.xml")

# 定义转换后 ONNX 模型的输出路径
# onnx_output_path = ir_model_path.with_suffix(".onnx")
onnx_output_path = Path("D:/Code/OpenVINOCode/Model/yolov8n/yolov8n_test.onnx")

# --- 2. 执行转换 ---
try:
    print(f"正在加载 OpenVINO IR 模型: {ir_model_path}")
    
    # 初始化 OpenVINO 核心对象
    core = ov.Core()
    
    # 读取 IR 模型到内存中，这将返回一个 openvino.Model 对象
    model = core.read_model(ir_model_path)
    
    print(f"开始将模型转换为 ONNX 格式...")
    
    # 使用 ov.save_model() 函数将模型对象保存为 .onnx 文件
    # 这是 OpenVINO 2023.1 及以后版本支持的关键功能
    ov.save_model(model, str(onnx_output_path))
    
    print("-" * 50)
    print("转换成功!")
    print(f"OpenVINO IR 模型: {ir_model_path}")
    print(f"已成功转换为 ONNX 模型: {onnx_output_path}")
    print("-" * 50)

except Exception as e:
    print(f"\n转换过程中发生错误: {e}")