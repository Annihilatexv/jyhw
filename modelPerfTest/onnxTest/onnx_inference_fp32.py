import onnxruntime as ort
import numpy as np
import time
import cv2

# --- 配置 ---
ONNX_MODEL_PATH = "/home/ghost/Code/OpenVINOCode/Model/yolov8n/yolov8n.onnx"
# 创建一个随机的输入图像用于测试，尺寸为 640x640
INPUT_IMAGE = np.random.randn(1, 3, 640, 640).astype(np.float32) 

# --- 加载模型并创建推理会话 ---
print(f"正在加载 FP32 ONNX 模型: {ONNX_MODEL_PATH}")
session_options = ort.SessionOptions()
# 可以在这里设置线程数等优化选项
# session_options.intra_op_num_threads = 4 
session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=session_options)

# 获取模型输入输出的名称
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"输入名称: {input_name}")
print(f"输出名称: {output_name}")

# --- 预热模型 ---
# 第一次推理通常会包含模型加载和初始化的时间，为了准确计时，先进行一次“预热”运行
print("正在预热模型...")
_ = session.run([output_name], {input_name: INPUT_IMAGE})
print("预热完成。")

# --- 执行推理并计时 ---
print("开始 FP32 推理...")
start_time = time.time()
# 在实际应用中，您可能会在这里循环处理多张图片
for _ in range(100): # 运行10次取平均值，使时间更具参考性
    outputs = session.run([output_name], {input_name: INPUT_IMAGE})
end_time = time.time()
print("FP32 推理完成。")

# --- 打印结果 ---
# `outputs` 是一个列表，包含模型的输出
# YOLOv8的输出通常是 [1, 84, 8400] (batch, 4 bbox coords + 80 class scores, num_proposals)
print(f"输出张量的形状: {outputs[0].shape}") 
total_time = end_time - start_time
average_time_ms = (total_time / 100) * 1000
print(f"\n--- FP32 推理性能 ---")
print(f"模型: {ONNX_MODEL_PATH}")
print(f"总计执行100次推理耗时: {total_time:.4f} 秒")
print(f"平均每次推理耗时: {average_time_ms:.2f} 毫秒")