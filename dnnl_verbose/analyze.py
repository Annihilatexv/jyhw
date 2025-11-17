#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd

def analyze_and_report(csv_file_path):
    """
    读取解析后的 onednn exec 日志 CSV 文件，
    按 primitive 分类统计总时间，并在终端打印报告。

    Args:
        csv_file_path (str): CSV 文件的路径。
    """
    # 检查输入文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"\n错误: 输入文件 '{csv_file_path}' 不存在。")
        print("请确认文件名是否正确，或先运行 extract_exec_log.py 脚本生成 CSV 文件。\n")
        return

    print(f"\n--- 正在读取和分析 CSV 文件: '{csv_file_path}' ---")

    try:
        # 1. 使用 pandas 读取 CSV 文件
        # 我们只关心 'primitive' 和 'exec_time' 这两列
        df = pd.read_csv(csv_file_path, usecols=['primitive', 'exec_time'])

        # 确保 'exec_time' 列是数值类型，对于无法转换的值，将其设为 NaN
        df['exec_time'] = pd.to_numeric(df['exec_time'], errors='coerce')
        # 删除含有 NaN 值的行，确保数据干净
        df.dropna(subset=['exec_time'], inplace=True)

        if df.empty:
            print("\n错误：CSV 文件中没有有效的执行时间数据。\n")
            return

        # 2. 按 'primitive' 分组，并计算每个组的 'exec_time' 总和
        time_summary = df.groupby('primitive')['exec_time'].sum()

        # 3. 对结果按时间降序排序
        time_summary = time_summary.sort_values(ascending=False)

        # 计算总执行时间，用于后续计算百分比
        total_time = time_summary.sum()
        
        if total_time == 0:
            print("\n警告：总执行时间为零。\n")
            # 即使为零，也打印表头和总计行
            print_report(pd.Series(), 0) # 传入空的Series
            return

        # 4. 打印格式化的报告
        print_report(time_summary, total_time)

    except FileNotFoundError:
        print(f"\n错误: 文件未找到 '{csv_file_path}'\n")
    except KeyError as e:
        print(f"\n错误: CSV 文件中缺少必需的列: {e}。请检查CSV文件是否正确生成。\n")
    except Exception as e:
        print(f"\n处理数据时发生未知错误: {e}\n")


def print_report(time_summary, total_time):
    """
    将统计结果格式化并打印到终端。

    Args:
        time_summary (pd.Series): 按 primitive 分组并求和后的数据。
        total_time (float): 总执行时间。
    """
    print("\n" + "="*55)
    print("      oneDNN Primitive Performance Summary")
    print("="*55)
    
    # 打印表头
    # < 左对齐, > 右对齐, ^ 居中
    print(f"{'Primitive Type':<20} | {'Total Time (ms)':>18} | {'Percentage':>10}")
    print("-"*55)

    # 逐行打印每个 primitive 的统计数据
    if not time_summary.empty:
        for primitive, time in time_summary.items():
            percentage = (time / total_time) * 100 if total_time > 0 else 0
            # 格式化输出：
            # {primitive:<20} -> 字符串，左对齐，占20个字符位
            # {time:>18.4f} -> 浮点数，右对齐，占18个字符位，保留4位小数
            # {percentage:>9.2f}% -> 浮点数，右对齐，占9个字符位，保留2位小数，最后加个 '%'
            print(f"{primitive:<20} | {time:>18.4f} | {percentage:>9.2f}%")

    # 打印分隔符和总计行
    print("-"*55)
    print(f"{'Total':<20} | {total_time:>18.4f} | {100.00:>9.2f}%")
    print("="*55 + "\n")


def main():
    """
    脚本主入口
    """
    if len(sys.argv) < 2:
        print("\n用法: python analyze_exec_time_cli.py <你的CSV文件名.csv>")
        print("例如: python analyze_exec_time_cli.py yolo_int8_pc_test_exec.csv\n")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyze_and_report(csv_file)


if __name__ == "__main__":
    main()