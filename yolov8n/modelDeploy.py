import openvino as ov
import numpy as np
import time
from pathlib import Path
import onnx

# --- 1. 配置区域 ---
# ONNX模型文件路径
ONNX_MODEL_PATH = Path("/home/ghost/Code/OpenVINOCode/Model/yolov8n/yolov8n.onnx")
# 转换后的IR模型文件路径 (OpenVINO格式)
IR_MODEL_PATH = ONNX_MODEL_PATH.with_suffix(".xml")
# 推理设备
DEVICE = "CPU"

def convert_onnx_to_ir(onnx_path: Path, ir_path: Path):
    """
    将ONNX模型转换为OpenVINO IR格式。
    如果IR文件已存在，则跳过转换。
    """
    if ir_path.exists():
        print(f"IR model {ir_path} already exists. Skipping conversion.")
        return
        
    print(f"Converting ONNX model {onnx_path} to IR format...")
    # 使用OpenVINO的模型转换API
    model = ov.convert_model(str(onnx_path))
    # 序列化并保存模型到.xml和.bin文件
    ov.save_model(model, str(ir_path))
    print(f"IR model saved to {ir_path}")

def get_profiling_info(infer_request: ov.InferRequest):
    """
    从推理请求中提取并打印性能分析信息。
    """
    profiling_info = infer_request.get_profiling_info()
    
    total_time_ms = 0
    perf_data = []
    
    for info in profiling_info:
        # info.status: 'EXECUTED', 'NOT_RUN'
        # info.node_name: 算子的名称
        # info.real_time: 实际执行时间 (microseconds)
        # info.cpu_time: CPU执行时间 (microseconds)
        # info.node_type: 算子的类型 (e.g., 'Convolution', 'ReLU')
        if info.status == ov.runtime.ProfilingInfo.Status.EXECUTED:
            perf_data.append({
                "name": info.node_name,
                "type": info.node_type,
                "real_time_ms": info.real_time.total_seconds() * 1000,
                "cpu_time_ms": info.cpu_time.total_seconds() * 1000,
            })
            total_time_ms += info.real_time.total_seconds() * 1000

    print("-" * 50)
    print(f"Total Inference Time: {total_time_ms:.4f} ms")
    print("-" * 50)
    print(f"{'Operator Name':<30} | {'Type':<20} | {'Time (ms)':<15}")
    print("-" * 70)
    
    # 按耗时降序排序
    perf_data.sort(key=lambda x: x['real_time_ms'], reverse=True)
    
    for item in perf_data:
        print(f"{item['name']:<30} | {item['type']:<20} | {item['real_time_ms']:.4f}")

    print("-" * 70)
    return perf_data
    
def main():
    # --- 步骤 1: 模型转换 (ONNX -> OpenVINO IR) ---
    convert_onnx_to_ir(ONNX_MODEL_PATH, IR_MODEL_PATH)

    # --- 步骤 2: 初始化OpenVINO Core ---
    print("Initializing OpenVINO Core...")
    core = ov.Core()
    
    # 打印可用设备
    print("Available devices:", core.available_devices)

    # --- 步骤 3: 加载并编译模型，并启用性能分析 ---
    print(f"Loading and compiling model {IR_MODEL_PATH} for {DEVICE}...")
    
    # 加载模型
    model = core.read_model(model=str(IR_MODEL_PATH))
    
    # 编译模型时，通过配置字典启用性能计数器
    # ov.properties.hint.performance_mode() 是 v2023.1 之后推荐的配置方式
    # ov.properties.enable_profiling(True) 是之前的配置方式，两者皆可
    config = {
        ov.properties.enable_profiling(): True
    }
    
    compiled_model = core.compile_model(model=model, device_name=DEVICE, config=config)
    
    # --- 步骤 4: 创建推理请求和准备输入数据 ---
    # 创建推理请求对象
    infer_request = compiled_model.create_infer_request()
    
    # 获取模型的输入节点信息
    input_tensor = compiled_model.input(0)
    input_shape = input_tensor.shape # e.g., [1, 3, 640, 640]
    input_type = input_tensor.element_type # e.g., <Type: 'float32'>
    
    print(f"Model input shape: {input_shape}")
    print(f"Model input type: {input_type}")
    
    # 生成一个符合输入形状和类型的随机数据作为虚拟输入
    # YOLOv8通常需要0-1范围的浮点数
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # --- 步骤 5: 执行推理 ---
    print("Performing a single inference run...")
    start_time = time.perf_counter()
    
    # infer方法会自动将数据填充到正确的输入张量
    infer_request.infer([dummy_input])
    
    end_time = time.perf_counter()
    print(f"End-to-end inference latency (from Python): {(end_time - start_time) * 1000:.4f} ms")

    # --- 步骤 6: 提取并分析性能数据 ---
    print("\nExtracting operator-level performance data...")
    operator_perf_data = get_profiling_info(infer_request)
    
    # 你可以将 `operator_perf_data` 保存为 JSON 文件，供后续平台使用
    import json
    with open("operator_performance.json", "w") as f:
        json.dump(operator_perf_data, f, indent=4)
    print("Operator performance data saved to operator_performance.json")

if __name__ == "__main__":
    main()