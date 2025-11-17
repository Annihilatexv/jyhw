import openvino as ov
import numpy as np
from pathlib import Path

class OpenVINOModel:
    """
    一个封装了OpenVINO模型加载、编译和推理的类。
    """
    def __init__(self, model_path: str, device: str = "CPU"):
        self.model_path = Path(model_path)
        self.device = device
        self.core = ov.Core()
        
        ir_path = self._to_ir_path(self.model_path)
        self._convert_if_needed(self.model_path, ir_path)
        
        self.model = self.core.read_model(model=str(ir_path))
        self.compiled_model = None
        self.infer_request = None

    def _to_ir_path(self, model_path: Path) -> Path:
        """根据输入模型路径确定IR路径"""
        if model_path.suffix == ".onnx":
            return model_path.with_suffix(".xml")
        return model_path

    def _convert_if_needed(self, original_path: Path, ir_path: Path):
        """如果需要，将ONNX转换为IR"""
        if original_path.suffix == ".onnx" and not ir_path.exists():
            print(f"Converting ONNX model {original_path} to IR format...")
            model = ov.convert_model(str(original_path))
            ov.save_model(model, str(ir_path))
            print(f"IR model saved to {ir_path}")

    def compile(self, enable_profiling=False):
        """编译模型并创建推理请求"""
        config = {ov.properties.enable_profiling(): enable_profiling}
        print(f"Compiling model for {self.device} with profiling={'enabled' if enable_profiling else 'disabled'}...")
        self.compiled_model = self.core.compile_model(
            model=self.model, 
            device_name=self.device, 
            config=config
        )
        self.infer_request = self.compiled_model.create_infer_request()
        print("Model compiled successfully.")

    def get_input_details(self):
        """获取模型输入细节"""
        input_tensor = self.compiled_model.input(0)
        return input_tensor.shape, input_tensor.element_type

    def create_dummy_input(self):
        """创建一个符合模型输入的随机数据"""
        shape, dtype = self.get_input_details()
        # 将OpenVINO类型转换为Numpy类型
        if dtype == ov.Type.f32:
            np_dtype = np.float32
        elif dtype == ov.Type.u8:
            np_dtype = np.uint8
        else:
            # 可根据需要添加更多类型映射
            raise ValueError(f"Unsupported data type: {dtype}")
        
        return np.random.rand(*shape).astype(np_dtype)

    def infer(self, input_data):
        """执行一次同步推理"""
        if not self.infer_request:
            raise RuntimeError("Model is not compiled. Call .compile() first.")
        
        results = self.infer_request.infer([input_data])
        return results

    def get_profiling_info(self):
        """获取性能分析数据"""
        if not self.infer_request:
            raise RuntimeError("Model is not compiled. Call .compile() first.")
        return self.infer_request.get_profiling_info()

    def get_and_save_op_performance(self, output_path: str):
        """
        提取算子性能数据并将其保存为JSON文件。
        """
        if not self.infer_request:
            raise RuntimeError("Model is not compiled with profiling enabled.")
        
        profiling_info = self.infer_request.get_profiling_info()
        
        total_time_ms = 0
        perf_data = []
        
        for info in profiling_info:
            if info.status == ov.ProfilingInfo.Status.EXECUTED:
                perf_data.append({
                    "name": info.node_name,
                    "type": info.node_type,
                    "real_time_ms": info.real_time.total_seconds() * 1000,
                    "cpu_time_ms": info.cpu_time.total_seconds() * 1000,
                })
                total_time_ms += info.real_time.total_seconds() * 1000
        
        # 按耗时降序排序
        perf_data.sort(key=lambda x: x['real_time_ms'], reverse=True)

        final_data = {
            "total_inference_time_ms": total_time_ms,
            "operator_performance": perf_data
        }

        import json
        with open(output_path, "w") as f:
            json.dump(final_data, f, indent=4)
        # print(f"Operator performance data saved to {output_path}")