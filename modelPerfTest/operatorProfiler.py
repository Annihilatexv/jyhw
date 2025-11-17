import argparse
from datetime import datetime
import os
from openvinoModel import OpenVINOModel

def profile_operators(model: OpenVINOModel, num_inferences: int = 1):
    """
    对模型进行指定次数的推理，以采集算子性能数据。
    """
    if num_inferences <= 0:
        raise ValueError("Number of inferences must be at least 1.")

    print(f"\n--- Running Operator Profiling ({num_inferences} inference run{'s' if num_inferences > 1 else ''}) ---")
    
    # 创建一次输入数据即可，后续重复使用
    dummy_input = model.create_dummy_input()

    # 首次推理会自动触发 JIT 编译和内存分配，其耗时通常不具代表性，
    # 但对于分析来说，采集这次的性能数据也可能有用。
    print("Running inference(s) to collect profiling data...")
    for i in range(num_inferences):
        model.infer(dummy_input)
        print(f"Inference {i+1}/{num_inferences} completed.", end='\r', flush=True)
    print()
    print("\nInference runs finished. Extracting profiling information...")

def analyze_and_display_performance(model: OpenVINOModel, top_n: int = 10):
    """
    从模型中提取算子性能数据，并在终端打印出 Top-N 的耗时算子。
    """
    profiling_info = model.get_profiling_info()
    
    total_time_ms = 0
    perf_data = []

    # 解析性能数据
    for info in profiling_info:
        # 只统计执行成功的算子
        if info.status == "EXECUTED":
            real_time_ms = info.real_time.total_seconds() * 1000
            perf_data.append({
                "name": info.node_name,
                "type": info.node_type,
                "real_time_ms": real_time_ms,
            })
            total_time_ms += real_time_ms
    
    # 按耗时从高到低排序
    perf_data.sort(key=lambda x: x['real_time_ms'], reverse=True)


def main():
    parser = argparse.ArgumentParser(
        description="OpenVINO Operator Performance Profiler.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-m", "--model", required=True, help="Path to the ONNX or OpenVINO IR model (.xml) file.")
    parser.add_argument("-d", "--device", default="CPU", help="Target device for inference (e.g., CPU, GPU).")
    parser.add_argument("-o", "--output_dir", default="./results", help="Directory to save the profiling JSON report.")
    parser.add_argument("-n", "--num_inferences", type=int, default=1, 
                        help="Number of inference runs to perform for profiling.\n"
                             "Note: OpenVINO aggregates times over runs, so this value\n"
                             "doesn't average the time but sums it up.")
    parser.add_argument("--top_n", type=int, default=15, help="Number of top time-consuming operators to display.")

    args = parser.parse_args()

    # --- 1. 初始化和编译模型 (开启性能分析) ---
    print(f"Initializing model: {args.model}")
    # 关键步骤：实例化模型
    model = OpenVINOModel(args.model, args.device)
    # 关键步骤：编译时必须将 enable_profiling 设置为 True
    model.compile(enable_profiling=True)

    # --- 2. 运行推理以采集数据 ---
    profile_operators(model, args.num_inferences)

    # --- 3. 分析、显示并保存结果 ---
    # 在终端显示 Top-N 结果
    analyze_and_display_performance(model, args.top_n)

    # 创建输出目录并保存完整的 JSON 报告
    # os.makedirs(args.output_dir, exist_ok=True)
    # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # output_filename = f"op_perf_{model.model_path.stem}_{timestamp}.json"
    output_path = args.output_dir
    
    # 使用您在 OpenVINOModel 类中已经实现的方法来保存文件
    model.get_and_save_op_performance(output_path)
    
    # print(f"\nFull profiling report has been saved to: {output_path}")


if __name__ == "__main__":
    main()