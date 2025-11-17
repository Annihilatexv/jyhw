import argparse
from openvinoModel import OpenVINOModel
import time
import os
from pathlib import Path

def main(args):
    """
    使用OpenVINO自带的测试工具对模型内的单个算子的执行时间进行统计
    """
    # 从args对象中获取配置
    modelPath = args.modelPath
    device = args.device
    repeatTimes = args.repeatTimes

    # print(f"模型路径 (Model Path): {modelPath}")
    # print(f"推理设备 (Device): {device}")
    # print(f"重复次数 (Repeat Times): {repeatTimes}")
    
    # 1. 初始化并编译模型，并启用profiling
    model = OpenVINOModel(modelPath, device)
    model.compile()

    # 2. 准备输入
    dummy_input = model.create_dummy_input()

    # 3. 执行推理
    # print("\nRunning inference...", flush=True)
    model.infer(dummy_input) # 预热
    for i in range(repeatTimes):
        model.infer(dummy_input) # 分析
    # print("\nInference finished.")


if __name__ == "__main__":
    # 1. 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="使用 OpenVINO 对模型进行基准测试。")

    # 2. 添加命令行参数
    parser.add_argument('--modelPath', type=str, required=True, help="指定 .onnx 或 .xml 格式的模型文件路径。")
    parser.add_argument('--device', type=str, default="CPU", help="指定推理设备，例如 'CPU', 'GPU'。")
    parser.add_argument('--repeatTimes', type=int, default=100, help="指定推理的重复次数。")

    # 3. 解析参数
    args = parser.parse_args()
    
    # 4. 调用主函数
    main(args)