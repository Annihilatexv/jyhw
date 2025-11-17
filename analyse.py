import re
from collections import defaultdict
import os

def analyze_benchmark_log(file_path):
    """
    Parses an OpenVINO benchmark log file to aggregate execution time and
    counts for each layer type.

    Args:
        file_path (str): The path to the log file.

    Returns:
        dict: A dictionary with layer types as keys and another dictionary
              containing 'total_time' and 'count' as values.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None

    # Use defaultdict to simplify aggregation
    layer_stats = defaultdict(lambda: {'total_time': 0.0, 'count': 0})
    
    parsing_started = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Start parsing only after this specific line is found
            if "[ INFO ] Performance counts for 0-th infer request:" in line:
                parsing_started = True
                continue
            
            if not parsing_started:
                continue

            # Stop parsing when we reach the total time summary
            if "Total time:" in line:
                break

            # Process only lines that represent an executed layer
            if 'EXECUTED' in line:
                # Use regex to robustly find layerType and realTime
                # This is more stable than splitting by spaces
                layer_type_match = re.search(r'layerType:\s+([\w\.-]+)', line)
                real_time_match = re.search(r'realTime \(ms\):\s+([\d\.]+)', line)

                if layer_type_match and real_time_match:
                    layer_type = layer_type_match.group(1)
                    real_time = float(real_time_match.group(1))
                    
                    layer_stats[layer_type]['total_time'] += real_time
                    layer_stats[layer_type]['count'] += 1

    return layer_stats

def print_stats_table(layer_stats):
    """Formats and prints the aggregated statistics in a Markdown table."""
    if not layer_stats:
        print("No statistics were generated. Please check the log file.")
        return

    # Sort the results by total execution time in descending order
    sorted_stats = sorted(layer_stats.items(), key=lambda item: item[1]['total_time'], reverse=True)

    total_time_all = 0.0
    total_count_all = 0

    # --- Print Header ---
    print("| Layer Type      | 执行时间 (ms) | 执行次数 |")
    print("| :-------------- | :------------ | :------- |")
    
    # --- Print Rows ---
    for layer_type, stats in sorted_stats:
        total_time = stats['total_time']
        count = stats['count']
        print(f"| {layer_type:<15} | {total_time:<13.3f} | {count:<8} |")
        total_time_all += total_time
        total_count_all += count
    
    # --- Print Footer (Total) ---
    print("| :-------------- | :------------ | :------- |")
    print(f"| **合计 (累加值)** | **{total_time_all:<11.3f}** | **{total_count_all:<6}** |")


# --- Main execution ---
# 请将 'fp32_pc_yolov8n.txt' 替换为您的日志文件的实际路径
log_file = "D:\\Code\\OpenVINOCode\\fp32_pc_yolov8n.txt"
stats = analyze_benchmark_log(log_file)
if stats:
    print_stats_table(stats)