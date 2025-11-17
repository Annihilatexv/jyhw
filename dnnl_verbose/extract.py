#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import csv
import os

def parse_exec_log_to_csv(input_file_path):
    """
    解析 onednn_verbose 日志文件，提取 'exec' 阶段的信息，并保存为 CSV 文件。

    Args:
        input_file_path (str): 输入的日志文件路径。
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f"错误: 输入文件 '{input_file_path}' 不存在。")
        return

    # 根据输入文件名生成输出文件名，例如 a.txt -> a_exec.csv
    base_name = os.path.splitext(input_file_path)[0]
    output_file_path = f"{base_name}_exec.csv"

    # 定义要匹配的日志行前缀
    # 注意：这里的 v1, primitive 是根据 oneDNN v3.x 的格式，旧版可能不同
    prefix_to_match = "onednn_verbose,v1,primitive,exec,"

    # CSV 文件的表头
    csv_header = [
        "engine", "primitive", "implementation", "prop_kind",
        "memory_descriptors", "attributes", "auxiliary", "problem_desc", "exec_time"
    ]

    # 用于记录成功写入的行数
    lines_written = 0

    print(f"--- 正在读取日志文件: '{input_file_path}' ---")

    try:
        # 使用 with 语句确保文件被正确关闭
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:

            # 创建一个 CSV writer 对象
            writer = csv.writer(outfile)

            # 1. 写入表头
            writer.writerow(csv_header)

            # 2. 逐行读取输入文件
            for line in infile:
                # 去除行首尾的空白字符
                stripped_line = line.strip()

                # 检查行是否以我们关心的前缀开头
                if stripped_line.startswith(prefix_to_match):
                    # 截取前缀之后的内容
                    data_part = stripped_line[len(prefix_to_match):]
                    
                    # 使用 csv 模块来解析这一行数据，可以正确处理带引号的逗号
                    # 我们需要把它包装成一个列表，因为 csv.reader 需要可迭代对象
                    parsed_row = next(csv.reader([data_part]))
                    
                    # 写入解析后的数据行
                    writer.writerow(parsed_row)
                    lines_written += 1

    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return

    print(f"--- 处理完成 ---")
    if lines_written > 0:
        print(f"成功提取并写入 {lines_written} 条 'exec' 记录到文件: '{output_file_path}'")
    else:
        print(f"未在输入文件中找到任何匹配 '{prefix_to_match}' 的行。")


def main():
    """
    脚本主入口
    """
    # 检查是否提供了文件名作为命令行参数
    if len(sys.argv) < 2:
        print("\n用法: python extract_exec_log.py <你的日志文件名.txt>")
        print("例如: python extract_exec_log.py yolo_int8_pc_test.txt\n")
        sys.exit(1)

    log_file = sys.argv[1]
    parse_exec_log_to_csv(log_file)


if __name__ == "__main__":
    main()