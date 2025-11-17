import time
from pathlib import Path
import cv2
import numpy as np
import openvino as ov
from tqdm import tqdm
import nncf

# --- 1. 配置您的模型和数据路径 ---
# 注意：ONNX模型应该是FP32/FP16精度的，而不是已经量化过的INT8模型
ONNX_MODEL_PATH = Path("D:\Code\OpenVINOCode\Model\yolov8n\yolov8n.xml") # <<< MODIFIED: 确保这里是原始的FP32模型路径
CALIBRATION_DATA_DIR = Path("D:/Code/OpenVINOCode/cocoData")


# --- 2. YOLOv8预处理函数 (无变化) ---
def preprocess_image(image_path, input_size=(640, 640)):
    image = cv2.imread(str(image_path))
    if image is None: return None
    h, w = image.shape[:2]
    r = min(input_size[0] / h, input_size[1] / w)
    resized_h, resized_w = int(h * r), int(w * r)
    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded_image = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    top = (input_size[0] - resized_h) // 2
    left = (input_size[1] - resized_w) // 2
    padded_image[top:top+resized_h, left:left+resized_w] = resized_image
    rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    chw_image = rgb_image.transpose(2, 0, 1)
    input_tensor = np.expand_dims(chw_image, axis=0).astype(np.float32) / 255.0
    return input_tensor


# --- 3. 数据加载器 (修改) ---
# ImageLoader现在更通用，只负责产出预处理好的张量
class ImageLoader:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        
    def __len__(self):
        return len(self.image_paths)

    def __iter__(self):
        for image_path in tqdm(self.image_paths, desc="处理校准数据"):
            tensor = preprocess_image(image_path)
            if tensor is not None:
                yield tensor # <<< MODIFIED: 只 yield 张量本身，而不是字典


# --- 4. 性能基准测试函数 (无变化) ---
def benchmark_model(compiled_model, num_iterations=1000):
    """
    对编译后的模型进行指定次数的推理，并计算性能指标。
    """
    input_tensor_name = compiled_model.input(0)
    input_shape = input_tensor_name.shape
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    print("正在进行预热运行...")
    for _ in range(10):
        compiled_model([dummy_input])

    inference_times = []
    print(f"开始进行 {num_iterations} 次推理测试...")
    start_total_time = time.perf_counter()
    for _ in tqdm(range(num_iterations), desc="基准测试中"):
        start_time = time.perf_counter()
        compiled_model([dummy_input])
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)
    
    end_total_time = time.perf_counter()
    total_inference_time = end_total_time - start_total_time
    
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


# --- 5. 主执行函数 (修改) ---
def main():
    print("--- YOLOv8 ONNX 模型性能对比测试 ---")

    if not ONNX_MODEL_PATH.exists():
        print(f"错误: ONNX模型文件未找到: {ONNX_MODEL_PATH}")
        return
    if not CALIBRATION_DATA_DIR.exists():
        print(f"错误: 校准数据集文件夹未找到: {CALIBRATION_DATA_DIR}")
        return

    core = ov.Core()
    
    print(f"\n[1/4] 正在加载FP32模型: {ONNX_MODEL_PATH}")
    model_fp32 = core.read_model(ONNX_MODEL_PATH)
    
    print("正在编译FP32模型...")
    compiled_model_fp32 = core.compile_model(model_fp32, "CPU")
    
    print("\n[2/4] 正在测试FP32模型性能...")
    fp32_results = benchmark_model(compiled_model_fp32)
    
    print(f"\n[3/4] 正在准备INT8量化...")
    image_files = [p for p in CALIBRATION_DATA_DIR.glob("**/*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not image_files:
        print(f"错误: 在 {CALIBRATION_DATA_DIR} 中未找到任何图片文件。")
        return
        
    # <<< ADDED: 这是解决问题的核心部分 >>>
    # 1. 创建你的原始数据加载器实例
    image_loader = ImageLoader(image_files)

    # 2. 定义一个转换函数 (transform_fn)
    #    这个函数接收 ImageLoader 产出的每一项数据（一个张量），
    #    并将其包装成模型输入所期望的字典格式。
    #    字典的 key 必须是模型的输入层名称。
    input_name = model_fp32.input(0).any_name
    def transform_fn(data_item):
        return {input_name: data_item}

    # 3. 使用 nncf.Dataset 包装你的数据加载器和转换函数
    calibration_dataset = nncf.Dataset(image_loader, transform_fn)
    # <<< END OF ADDED SECTION >>>

    print("开始进行INT8量化 (这可能需要一些时间)...")
    quantized_model = nncf.quantize(
        model=model_fp32,
        calibration_dataset=calibration_dataset, # <<< MODIFIED: 使用包装后的 nncf.Dataset
        preset=nncf.QuantizationPreset.PERFORMANCE
    )
    
    print("量化完成。正在编译INT8模型...")
    compiled_model_int8 = core.compile_model(quantized_model, "CPU")
    
    print("\n[4/4] 正在测试INT8模型性能...")
    int8_results = benchmark_model(compiled_model_int8)

    # --- 打印最终对比报告 ---
    print("\n\n" + "="*60)
    print("              性能对比报告 (1000次推理)")
    print("="*60)
    print(f"{'指标':<20} | {'FP32 模型':<18} | {'INT8 模型':<18}")
    print("-" * 60)
    print(f"{'平均时间 (ms)':<20} | {fp32_results['avg']:<18.2f} | {int8_results['avg']:<18.2f}")
    print(f"{'中位时间 (ms)':<20} | {fp32_results['median']:<18.2f} | {int8_results['median']:<18.2f}")
    print(f"{'最小时间 (ms)':<20} | {fp32_results['min']:<18.2f} | {int8_results['min']:<18.2f}")
    print(f"{'最大时间 (ms)':<20} | {fp32_results['max']:<18.2f} | {int8_results['max']:<18.2f}")
    print(f"{'吞吐量 (FPS) ':<20} | {fp32_results['throughput']:<18.2f} | {int8_results['throughput']:<18.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()