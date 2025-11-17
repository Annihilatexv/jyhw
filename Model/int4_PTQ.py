import os
import numpy as np
import cv2
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader, CalibrationMethod

class MobileViTDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        """
        初始化校准数据读取器。

        参数:
        calibration_image_folder (str): 包含校准图像的文件夹路径。
        model_path (str): ONNX模型文件路径，用于自动获取输入尺寸和名称。
        """
        self.image_folder = calibration_image_folder
        self.image_files = [f for f in os.listdir(self.image_folder) if os.path.isfile(os.path.join(self.image_folder, f))]
        self.datasize = len(self.image_files)
        
        # 动态获取模型的输入信息
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_info = session.get_inputs()[0]
        self.input_name = input_info.name
        _, _, self.height, self.width = input_info.shape
        
        print(f"成功初始化数据读取器:")
        print(f" - 模型输入节点名称: {self.input_name}")
        print(f" - 模型期望输入尺寸: ({self.height}, {self.width})")
        print(f" - 在 '{calibration_image_folder}' 中找到 {self.datasize} 个校准图像。")

        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            # 创建一个可迭代的数据生成器
            self.enum_data = iter(self._preprocess_images())
        return next(self.enum_data, None)

    def _preprocess_images(self):
        """
        数据预处理生成器。
        这里的预处理必须和模型训练时的预处理逻辑完全一致。
        """
        # ImageNet 标准归一化参数
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for image_file in self.image_files:
            try:
                image_path = os.path.join(self.image_folder, image_file)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"警告: 无法读取图像 '{image_path}'，已跳过。")
                    continue
                
                # 1. 调整尺寸
                img = cv2.resize(img, (self.width, self.height))
                
                # 2. BGR -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 3. 归一化 (先转为float32并缩放到[0,1]，然后减均值除以标准差)
                img = img.astype(np.float32) / 255.0
                img = (img - mean) / std
                
                # 4. HWC -> CHW (高度, 宽度, 通道 -> 通道, 高度, 宽度)
                img = np.transpose(img, (2, 0, 1))
                
                # 5. 增加 Batch 维度
                img = np.expand_dims(img, axis=0)

                # 6. 返回符合ONNX Runtime要求的字典
                yield {self.input_name: img}
            except Exception as e:
                print(f"处理图像 '{image_file}' 时出错: {e}")


# --- 主流程 ---

# 0. 定义文件路径
# 这是上一步固定了batch=1的模型
model_fp32 = "/home/ghost/Code/OpenVINOCode/Model/yolov8n/yolov8n.onnx" 
# 定义量化后模型的保存路径
# <<< 主要修改点 1: 修改输出文件名以反映INT4量化 >>>
model_quant = "/home/ghost/Code/OpenVINOCode/Model/yolov8n/yolov8n_int4.onnx"
# 存放校准图像的文件夹
calibration_image_folder = "/home/ghost/Code/OpenVINOCode/cocoData" # 请确保此文件夹存在并包含一些图片

# 检查所需文件和文件夹是否存在
if not os.path.exists(model_fp32):
    raise FileNotFoundError(f"错误: FP32模型 '{model_fp32}' 不存在。")
if not os.path.exists(calibration_image_folder):
    os.makedirs(calibration_image_folder)
    print(f"'{calibration_image_folder}' 不存在，已创建。请向其中添加校准图像。")
    # 为了能让脚本运行，可以提示用户或直接退出
    if not os.listdir(calibration_image_folder):
        print("错误: 校准文件夹为空，请添加图片后再运行。")
        exit()

# 1. 创建数据读取器实例
calib_dr = MobileViTDataReader(calibration_image_folder, model_fp32)

# 2. 执行静态量化
print("\n开始执行INT4静态量化，这可能需要一些时间...")
quantize_static(
    model_input=model_fp32,
    model_output=model_quant,
    calibration_data_reader=calib_dr,
    quant_format=QuantFormat.QDQ,
    # <<< 主要修改点 2: 将激活值量化类型从 QInt8 改为 QInt4 >>>
    activation_type=QuantType.QInt4,
    # <<< 主要修改点 3: 将权重量化类型从 QInt8 改为 QInt4 >>>
    # 注意: ONNX Runtime要求权重和激活值的类型是有符号或无符号的配对。
    # 例如, (QInt4, QInt4) 或 (QUInt4, QUInt4)
    weight_type=QuantType.QInt4,
    calibrate_method=CalibrationMethod.MinMax,
    # [可选] 增加一些额外参数，对于INT4量化可能有用
    # extra_options={
    #     "ActivationSymmetric": False, # 对称或非对称量化激活值
    #     "WeightSymmetric": True,     # 对称或非对称量化权重
    # }
)

print(f"\n量化完成！INT4静态量化模型已保存至: {model_quant}")