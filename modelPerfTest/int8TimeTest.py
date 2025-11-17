import cv2
import numpy as np
import openvino as ov
import time

# --- 1. 初始化 OpenVINO Core ---
core = ov.Core()

# --- 2. 载入并编译模型 ---
# 直接载入你的INT8 ONNX模型
model_path = "/home/ghost/Code/OpenVINOCode/Model/yolov8n/yolov8n_int8.onnx"
# 将模型编译到指定设备，例如 "CPU" 或 "GPU"
# 编译过程会针对目标硬件进行优化，初次编译会比较耗时
print("Loading and compiling model...")
compiled_model = core.compile_model(model_path, "CPU")
print("Model compiled.")

# --- 3. 获取模型输入输出信息 ---
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
# 获取模型输入的形状信息 (例如: [1, 3, 640, 640])
input_shape = input_layer.shape
input_height, input_width = input_shape[2], input_shape[3]

# --- 4. 准备输入数据 ---
# 创建一个符合模型输入尺寸的随机图像用于测试
# 在实际应用中，你需要从文件加载并进行预处理
dummy_image = np.random.randint(0, 255, size=(input_height, input_width, 3), dtype=np.uint8)

# 预处理：Resize, BGR to RGB, HWC to CHW, Normalize
# 注意：此处的预处理必须和你的模型训练及量化校准时完全一致！
resized_image = cv2.resize(dummy_image, (input_width, input_height))
rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
normalized_image = rgb_image.astype(np.float32) / 255.0
input_tensor = np.transpose(normalized_image, (2, 0, 1))  # HWC -> CHW
input_tensor = np.expand_dims(input_tensor, axis=0)      # Add batch dimension -> [1, 3, 640, 640]

print(f"Input tensor shape: {input_tensor.shape}")

# --- 5. 执行推理并测试耗时 ---
num_iterations = 100
latencies = []

# 预热一次，确保所有初始化完成
print("Warming up...")
_ = compiled_model([input_tensor])
print("Warm-up finished.")

print(f"Running inference for {num_iterations} iterations...")
start_time = time.perf_counter()

for _ in range(num_iterations):
    iter_start_time = time.perf_counter()
    # 同步推理
    results = compiled_model([input_tensor])[output_layer]
    iter_end_time = time.perf_counter()
    latencies.append((iter_end_time - iter_start_time) * 1000) # 转换为毫秒

end_time = time.perf_counter()

# --- 6. 打印结果 ---
total_time_ms = (end_time - start_time) * 1000
average_latency_ms = np.mean(latencies)
fps = 1000 / average_latency_ms

print("\n--- Inference Performance ---")
print(f"Total time for {num_iterations} iterations: {total_time_ms:.2f} ms")
print(f"Average latency per image: {average_latency_ms:.2f} ms")
print(f"Estimated FPS: {fps:.2f}")

# results变量现在包含了模型的输出，你可以进行后续的后处理（如NMS）
print(f"Output tensor shape: {results.shape}")