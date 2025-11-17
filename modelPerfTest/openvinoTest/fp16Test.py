import time
from pathlib import Path
import numpy as np
import openvino as ov
from tqdm import tqdm
import nncf

# --- 1. 配置您的模型路径 ---
# 注意：ONNX模型应该是FP32精度的
ONNX_MODEL_PATH = Path("D:\Code\OpenVINOCode\Model\yolov8n\yolov8n.onnx") # <<< 确保这里是原始的FP32模型路径

# --- 2. 性能基准测试函数 (无变化) ---
def benchmark_model(model_name: str, compiled_model, num_iterations=1000):
    """
    对编译后的模型进行指定次数的推理，并计算性能指标。
    """
    input_tensor_name = compiled_model.input(0)
    input_shape = input_tensor_name.shape
    # 使用随机数据进行性能测试
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    print(f"[{model_name}] 正在进行预热运行...")
    for _ in range(10):
        compiled_model([dummy_input])

    inference_times = []
    print(f"[{model_name}] 开始进行 {num_iterations} 次推理测试...")
    start_total_time = time.perf_counter()
    for _ in tqdm(range(num_iterations), desc=f"基准测试 ({model_name})"):
        start_time = time.perf_counter()
        compiled_model([dummy_input])
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)
    
    end_total_time = time.perf_counter()
    total_inference_time = end_total_time - start_total_time
    
    # 转换为毫秒
    inference_times_ms = [t * 1000 for t in inference_times]
    
    avg_time = np.mean(inference_times_ms)
    max_time = np.max(inference_times_ms)
    min_time = np.min(inference_times_ms)
    median_time = np.median(inference_times_ms)
    throughput = num_iterations / total_inference_time
    
    return {
        "avg": avg_time,
        "max": max_time,
        "min": min_time,
        "median": median_time,
        "throughput": throughput
    }


# --- 3. 主执行函数 (已修改) ---
def main():
    print("--- YOLOv8 ONNX 模型性能对比测试 (FP32 vs FP16) ---")

    if not ONNX_MODEL_PATH.exists():
        print(f"错误: ONNX模型文件未找到: {ONNX_MODEL_PATH}")
        return

    core = ov.Core()
    
    # --- FP32 模型处理 ---
    print(f"\n[1/4] 正在加载FP32模型: {ONNX_MODEL_PATH}")
    model_fp32 = core.read_model(ONNX_MODEL_PATH)
    
    print("正在编译FP32模型...")
    compiled_model_fp32 = core.compile_model(model_fp32, "CPU")
    
    print("\n[2/4] 正在测试FP32模型性能...")
    fp32_results = benchmark_model("FP32", compiled_model_fp32)
    
    # --- FP16 模型处理 ---
    print(f"\n[3/4] 正在将模型权重压缩为FP16...")
    # 使用 nncf.compress_weights 将模型权重转换为 FP16
    # 这个过程很快，因为它只改变权重的数据类型，不涉及校准
    model_fp16 = nncf.compress_weights(model_fp32)
    
    print("权重压缩完成。正在编译FP16模型...")
    compiled_model_fp16 = core.compile_model(model_fp16, "CPU")
    
    print("\n[4/4] 正在测试FP16模型性能...")
    fp16_results = benchmark_model("FP16", compiled_model_fp16)

    # --- 打印最终对比报告 ---
    print("\n\n" + "="*60)
    print("              性能对比报告 (1000次推理)")
    print("="*60)
    print(f"{'指标':<20} | {'FP32 模型':<18} | {'FP16 模型':<18}")
    print("-" * 60)
    print(f"{'平均时间 (ms)':<20} | {fp32_results['avg']:<18.2f} | {fp16_results['avg']:<18.2f}")
    print(f"{'中位时间 (ms)':<20} | {fp32_results['median']:<18.2f} | {fp16_results['median']:<18.2f}")
    print(f"{'最小时间 (ms)':<20} | {fp32_results['min']:<18.2f} | {fp16_results['min']:<18.2f}")
    print(f"{'最大时间 (ms)':<20} | {fp32_results['max']:<18.2f} | {fp16_results['max']:<18.2f}")
    print(f"{'吞吐量 (FPS)':<20} | {fp32_results['throughput']:<18.2f} | {fp16_results['throughput']:<18.2f}")
    print("=" * 60)
    
    # --- 性能提升分析 ---
    speed_increase = ((1 / fp16_results['avg']) - (1 / fp32_results['avg'])) / (1 / fp32_results['avg']) * 100
    throughput_increase = (fp16_results['throughput'] - fp32_results['throughput']) / fp32_results['throughput'] * 100
    
    print("\n分析:")
    print(f"与FP32相比，FP16模型的平均推理时间变化: {speed_increase:+.2f}%")
    print(f"与FP32相比，FP16模型的吞吐量提升: {throughput_increase:+.2f}%")
    print("\n注意: 性能提升依赖于硬件支持。在不支持FP16原生计算的CPU上，性能可能没有显著差异，")
    print("甚至会因为需要转换回FP32进行计算而略有下降。主要优势是模型体积减半。")


if __name__ == "__main__":
    main()