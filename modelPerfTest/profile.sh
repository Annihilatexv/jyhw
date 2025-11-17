#!/bin/bash

# ==============================================================================
# 
#   OpenVINO 模型性能分析脚本 (最终精简版)
#
#   策略:
#   1. 使用 perf stat 进行宏观性能统计。
#   2. 使用 perf record 和 FlameGraph 生成深度分析的火焰图。
#   3. 所有产物保存在带时间戳的目录中，不自动删除。
#   4. 自动管理和恢复系统安全设置。
#
# ==============================================================================

# 当任何命令执行失败时，立即退出脚本
set -e

# --- 用户配置区 ---
OPENVINO_SETUP_SCRIPT="/home/ghost/openvinoInstall/setupvars.sh"
TARGET_SCRIPT="./inferProfiler.py"
OPERATOR_SCRIPT="./operatorProfiler.py"  # 算子性能测试文件路径
OPERATOR_SCRIPT_MODEL_PATH="/home/ghost/Code/OpenVINOCode/Model/yolov8n/yolov8n_int4.onnx"  # 测试模型路径
FLAMEGRAPH_DIR="/home/ghost/FlameGraph"
PERF_FREQ=99

# --- 脚本主逻辑 ---

# 捕获脚本退出信号，并执行清理函数
trap cleanup EXIT

# 清理函数，只负责恢复安全设置
cleanup() {
    echo -e "\n=== 恢复系统安全设置 ==="
    
    if [ -n "$original_kptr_restrict" ]; then
        echo "正在将 kernel.kptr_restrict 恢复为原始值: $original_kptr_restrict"
        echo "$original_kptr_restrict" | sudo tee /proc/sys/kernel/kptr_restrict > /dev/null
    fi

    if [ -n "$original_perf_paranoid" ]; then
        echo "正在将 kernel.perf_event_paranoid 恢复为原始值: $original_perf_paranoid"
        echo "$original_perf_paranoid" | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null
    fi
    echo "安全设置已恢复。"
}

echo "=== 准备工作：加载环境并调整权限 ==="

# 保存并临时修改安全参数
original_kptr_restrict=$(cat /proc/sys/kernel/kptr_restrict)
original_perf_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
echo "临时放宽 perf 安全限制..."
echo 0 | sudo tee /proc/sys/kernel/kptr_restrict > /dev/null
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid > /dev/null

# 加载 OpenVINO 环境变量
source "$OPENVINO_SETUP_SCRIPT"
echo "OpenVINO 环境已加载。"

# 获取 Python 解释器路径并创建结果目录
PYTHON_PATH=$(which python3)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
DIRECTORY_PATH="./results/profile_${TIMESTAMP}"
mkdir -p "$DIRECTORY_PATH"
echo "所有分析结果将保存在: $DIRECTORY_PATH"

# 定义所有产物的文件路径
STAT_REPORT_FILE="$DIRECTORY_PATH/perf_stat_report.txt"
PERF_DATA_FILE="$DIRECTORY_PATH/perf.data"
FOLDED_STACKS_FILE="$DIRECTORY_PATH/out.perf-folded"
OUTPUT_SVG="$DIRECTORY_PATH/flamegraph.svg"
OPERATOR_TOP_FILE="$DIRECTORY_PATH/operator_perf.json"

# =================================================
#               开始执行两阶段性能分析
# =================================================

echo -e "\n--- 分析阶段 1: [perf stat] 宏观性能统计 ---"
perf stat \
    -e instructions,cycles,cpu-clock,branch-instructions,branch-misses,cache-references,cache-misses,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,context-switches,page-faults,duration_time  \
    -o "$STAT_REPORT_FILE" \
    -- \
    "$PYTHON_PATH" "$TARGET_SCRIPT" \
    --modelPath "$OPERATOR_SCRIPT_MODEL_PATH" \
    --repeatTimes 1000 


echo "宏观统计报告已保存至: $STAT_REPORT_FILE"
echo "--- 报告预览 (前20行) ---"
cat "$STAT_REPORT_FILE" | head -n 20
echo "--------------------------"


echo -e "\n--- 分析阶段 2: [perf record] 深度采样与火焰图生成 ---"
echo "正在进行深度采样..."
perf record \
    --call-graph lbr \
    -F "$PERF_FREQ" \
    -o "$PERF_DATA_FILE" \
    -- \
    "$PYTHON_PATH" "$TARGET_SCRIPT" \
    --modelPath "$OPERATOR_SCRIPT_MODEL_PATH" \
    --repeatTimes 1000 

echo "正在处理数据并生成火焰图..."
perf script -i "$PERF_DATA_FILE" | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" > "$FOLDED_STACKS_FILE"
"$FLAMEGRAPH_DIR/flamegraph.pl" "$FOLDED_STACKS_FILE" > "$OUTPUT_SVG"


echo -e "\n--- 分析阶段 3: [OpenVINO] 算子执行时间统计 ---"
"$PYTHON_PATH" "$OPERATOR_SCRIPT" \
    -m "$OPERATOR_SCRIPT_MODEL_PATH" \
    -d "CPU" \
    -n "100" \
    --top_n 20 \
    -o "$OPERATOR_TOP_FILE"
            
echo "算子统计报告已保存至: $OPERATOR_TOP_FILE"



echo -e "\n=========================================================="
echo "✅ 所有性能分析阶段已完成！"
echo "📂 所有产物已保存至目录: $(realpath "$DIRECTORY_PATH")"
echo "    - 宏观统计报告: $(basename "$STAT_REPORT_FILE")"
echo "    - 原始采样数据: $(basename "$PERF_DATA_FILE")"
echo "    - 折叠后的栈信息: $(basename "$FOLDED_STACKS_FILE")"
echo "    - 算子执行时间统计报告: $(basename "$OPERATOR_TOP_FILE")"
echo "    - 🔥 可视化火焰图: $(basename "$OUTPUT_SVG")"
echo "=========================================================="

# 脚本正常结束后，trap 会自动调用 cleanup 函数来恢复系统安全设置











