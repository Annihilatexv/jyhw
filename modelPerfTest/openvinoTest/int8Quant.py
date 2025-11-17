from pathlib import Path
import cv2
import numpy as np
import openvino as ov
from tqdm import tqdm
import nncf

# --- 1. 配置您的模型和数据路径 ---
# 注意：ONNX模型应该是FP32/FP16精度的
ONNX_MODEL_PATH = Path("D:\Code\OpenVINOCode\Model\yolov8n\yolov8n.xml") # <<< 确保这里是原始的FP32模型路径
CALIBRATION_DATA_DIR = Path("D:/Code/OpenVINOCode/cocoData") # <<< 校准图片所在的文件夹


# --- 2. YOLOv8预处理函数 (为校准所需) ---
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


# --- 3. 数据加载器 (为校准所需) ---
class ImageLoader:
    def __init__(self, image_paths):
        # 为了加快校准过程，通常不需要使用全部数据集
        # 这里我们选择前300张图片，您可以根据需要调整
        self.image_paths = image_paths[:300]
        print(f"将使用 {len(self.image_paths)} 张图片进行校准。")
        
    def __len__(self):
        return len(self.image_paths)

    def __iter__(self):
        for image_path in tqdm(self.image_paths, desc="处理校准数据"):
            tensor = preprocess_image(image_path)
            if tensor is not None:
                yield tensor


# --- 4. 主执行函数 (已修改为仅量化和保存) ---
def main():
    """
    加载一个FP32 ONNX模型，使用校准数据集将其量化为INT8，并保存为OpenVINO IR格式。
    """
    print("--- OpenVINO INT8 量化工具 ---")

    if not ONNX_MODEL_PATH.exists():
        print(f"错误: ONNX模型文件未找到: {ONNX_MODEL_PATH}")
        return
    if not CALIBRATION_DATA_DIR.exists() or not CALIBRATION_DATA_DIR.is_dir():
        print(f"错误: 校准数据集文件夹未找到: {CALIBRATION_DATA_DIR}")
        return

    # --- 定义输出路径 ---
    # 在原始文件名后添加 "_int8" 后缀
    output_ir_path = ONNX_MODEL_PATH.with_name(f"{ONNX_MODEL_PATH.stem}_int8_nncf.xml")

    try:
        core = ov.Core()
        
        # --- 1. 加载原始FP32模型 ---
        print(f"\n[1/4] 正在加载FP32模型: {ONNX_MODEL_PATH}")
        model_fp32 = core.read_model(ONNX_MODEL_PATH)
        
        # --- 2. 准备校准数据集 ---
        print(f"\n[2/4] 正在准备校准数据集...")
        image_files = sorted([p for p in CALIBRATION_DATA_DIR.glob("**/*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        if not image_files:
            print(f"错误: 在 {CALIBRATION_DATA_DIR} 中未找到任何图片文件。")
            return
            
        image_loader = ImageLoader(image_files)

        # 定义一个转换函数，将数据加载器的输出包装成NNCF期望的字典格式
        input_name = model_fp32.input(0).any_name
        def transform_fn(data_item):
            return {input_name: data_item}

        # 使用 nncf.Dataset 包装数据加载器和转换函数
        calibration_dataset = nncf.Dataset(image_loader, transform_fn)

        # --- 3. 执行INT8量化 ---
        print("\n[3/4] 开始进行INT8量化 (这可能需要几分钟)...")
        quantized_model = nncf.quantize(
            model=model_fp32,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE,  # MIXED通常在性能和精度之间有很好的平衡  PERFORMANCE选择性能
        )
        
        # --- 4. 保存量化后的模型 ---
        print(f"\n[4/4] 正在保存INT8模型到: {output_ir_path}")
        ov.serialize(quantized_model, str(output_ir_path))
        
        print("\n量化并保存成功!")
        print(f"  - XML 文件: {output_ir_path}")
        print(f"  - BIN 文件: {output_ir_path.with_suffix('.bin')}")

    except Exception as e:
        print(f"\n在处理过程中发生错误: {e}")


if __name__ == "__main__":
    main()
