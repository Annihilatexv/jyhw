import onnxruntime as ort
import numpy as np
import time
import os

# --- 1. 配置 ---
ONNX_MODEL_PATH = "/home/ghost/Code/OpenVINOCode/Model/yolov8n/yolov8n_int8.onnx"
INPUT_IMAGE = np.random.randn(1, 3, 640, 640).astype(np.float32)
# 增加运行次数以获得更稳定的统计数据
NUM_RUNS = 1000 

# --- 2. 设置 ONNX Runtime 的会话选项 ---
sess_options = ort.SessionOptions()
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# --- 3. 设置 OpenVINO EP 的特定选项 ---
provider_options = {
    'device_type' : 'CPU', # 'CPU_FP32'是设备标识，OpenVINO会自动检测INT8能力
    'num_of_threads': os.cpu_count(), # 让OpenVINO使用所有可用的CPU核心
    'cache_dir': './ov_cache', # 启用模型缓存，可以加快后续启动速度
}

# --- 4. 创建推理会话 ---
print(f"正在加载 INT8 ONNX 模型: {ONNX_MODEL_PATH}")
print(f"Provider Options: {provider_options}")

try:
    session = ort.InferenceSession(
        ONNX_MODEL_PATH, 
        sess_options=sess_options, 
        providers=['CPUExecutionProvider'],
        provider_options=[provider_options]
    )
    print(f"成功加载模型。当前使用的EP: {session.get_providers()}")
except Exception as e:
    print(f"加载模型失败。请确保 'onnxruntime-openvino' 已正确安装。")
    print(f"错误: {e}")
    exit()

# 获取模型输入输出
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- 5. 预热模型 ---
# 预热次数可以适当增加
print(f"\n正在预热模型 (运行 {min(10, NUM_RUNS // 10)} 次)...")
for _ in range(min(10, NUM_RUNS // 10)):
    _ = session.run([output_name], {input_name: INPUT_IMAGE})
print("预热完成。")

# --- 6. 执行推理并收集所有延迟数据 ---
print(f"开始性能测试 (运行 {NUM_RUNS} 次)...")
latencies = []
# 记录总的基准测试时间来计算吞吐量
start_benchmark_time = time.perf_counter()

for _ in range(NUM_RUNS):
    start_inference_time = time.perf_counter()
    _ = session.run([output_name], {input_name: INPUT_IMAGE})
    end_inference_time = time.perf_counter()
    # 将单次延迟（秒）转换为毫秒并存入列表
    latencies.append((end_inference_time - start_inference_time) * 1000)

end_benchmark_time = time.perf_counter()
print("性能测试完成。")

# --- 7. 计算并打印所有性能指标 ---

# a. 计算总耗时（秒）
total_benchmark_time_sec = end_benchmark_time - start_benchmark_time

# b. 基于延迟列表计算统计数据 (单位: 毫秒)
avg_latency_ms = np.mean(latencies)
min_latency_ms = np.min(latencies)
max_latency_ms = np.max(latencies)
median_latency_ms = np.median(latencies)
p95_latency_ms = np.percentile(latencies, 95)

# c. 计算吞吐量 (单位: FPS)
# 吞吐量 = 总运行次数 / 总耗时
throughput_fps = NUM_RUNS / total_benchmark_time_sec

print("\n--- INT8 ONNX 推理性能分析 ---")
print(f"模型: {ONNX_MODEL_PATH}")
print(f"执行提供者: {session.get_providers()[0]}")
print(f"执行次数: {NUM_RUNS}")

print("\n延迟 (Latency) 统计 (单位: 毫秒):")
print(f"  - 最 小 值 (Min):    {min_latency_ms:.2f} ms")
print(f"  - 最 大 值 (Max):    {max_latency_ms:.2f} ms")
print(f"  - 平 均 值 (Average): {avg_latency_ms:.2f} ms")
print(f"  - 中 位 数 (Median):  {median_latency_ms:.2f} ms")
print(f"  - 95分位延迟: {p95_latency_ms:.2f} ms")

print("\n吞吐量 (Throughput):")
print(f"  - {throughput_fps:.2f} 帧/秒 (FPS)")