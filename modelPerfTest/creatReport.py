import argparse
import json
import re
from pathlib import Path

def parse_perf_stat(file_path: Path) -> dict:
    """
    【修复版】增强版解析函数：能健壮地处理 perf stat 输出不完整的情况。
    """
    stats = {
        'performance_counter_stats': file_path.read_text(),
        'key_metrics': {}
    }
    content = stats['performance_counter_stats']
    
    patterns = {
        'task_clock': r'([0-9,]+\.[0-9]+)\s+msec\s+cpu-clock',
        'cycles': r'([0-9,]+)\s+cycles',
        'instructions': r'([0-9,]+)\s+instructions',
        'ipc': r'([0-9]+\.[0-9]+)\s+insn\s+per\s+cycle',
        'cache_references': r'([0-9,]+)\s+cache-references',
        'cache_misses': r'([0-9,]+)\s+cache-misses',
        'cache_miss_percent': r'([0-9]+\.[0-9]+)\s+%\s+of\s+all\s+cache\s+refs',
        'branch_instructions': r'([0-9,]+)\s+branch-instructions',
        'branch_misses': r'([0-9,]+)\s+branch-misses',
        'llc_loads': r'([0-9,]+)\s+LLC-loads',
        'llc_load_misses': r'([0-9,]+)\s+LLC-load-misses',
        'wall_time': r'([0-9,]+\.[0-9]+)\s+seconds\s+time\s+elapsed'
    }

    # 提取原始数值
    raw_metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            raw_metrics[key] = float(match.group(1).replace(',', ''))
        else:
            raw_metrics[key] = None # 如果找不到，值为 None

    # --- 关键修复区 ---
    # 填充最终要显示的指标，并在格式化前检查值是否为 None
    km = stats['key_metrics']
    
    # 定义一个辅助函数来安全地格式化
    def safe_format(key, format_str, default_val="N/A"):
        val = raw_metrics.get(key)
        return format_str.format(val) if val is not None else default_val

    km['ipc'] = safe_format('ipc', "{:.2f}")
    km['wall_time'] = safe_format('wall_time', "{:.2f}")
    km['task_clock'] = safe_format('task_clock', "{:.2f}")
    km['cache_miss_percent'] = safe_format('cache_miss_percent', "{:.2f}")

    # 计算并安全地格式化分支预测失败率
    branch_instructions = raw_metrics.get('branch_instructions')
    branch_misses = raw_metrics.get('branch_misses')
    if branch_instructions and branch_misses and branch_instructions > 0:
        branch_miss_rate = (branch_misses / branch_instructions) * 100
        km['branch_miss_rate'] = f"{branch_miss_rate:.2f}"
    else:
        km['branch_miss_rate'] = "N/A"

    # 计算并安全地格式化L3缓存未命中率
    llc_loads = raw_metrics.get('llc_loads')
    llc_load_misses = raw_metrics.get('llc_load_misses')
    if llc_loads and llc_load_misses and llc_loads > 0:
        llc_miss_rate = (llc_load_misses / llc_loads) * 100
        km['llc_miss_rate'] = f"{llc_miss_rate:.2f}"
    else:
        km['llc_miss_rate'] = "N/A"

    return stats

# ... (脚本的其余部分保持不变) ...
def parse_operator_perf(file_path: Path) -> dict:
    """读取 operator_perf.json 文件。"""
    with file_path.open('r') as f:
        return json.load(f)

def embed_svg(file_path: Path) -> str:
    """读取 flamegraph.svg 文件内容用于内嵌。"""
    return file_path.read_text()

def generate_html_report(report_data: dict, output_path: Path):
    """根据解析的数据生成最终的 HTML 报告 (升级版)。"""
    
    html_template = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>模型性能分析报告 (增强版)</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; background-color: #f4f7f9; color: #333; }}
            .container {{ max-width: 1200px; margin: 20px auto; padding: 20px; }}
            .header {{ text-align: center; border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; margin-bottom: 30px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-top: 40px; }}
            .card {{ background-color: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; }}
            .metric {{ text-align: center; padding: 15px; border-radius: 5px; background-color: #ecf0f1; }}
            .metric .value {{ font-size: 2em; font-weight: bold; }}
            .metric .label {{ font-size: 0.9em; color: #7f8c8d; margin-top: 5px; }}
            .value.ipc {{ color: #2980b9; }}
            .value.time {{ color: #27ae60; }}
            .value.miss-rate {{ color: #c0392b; }} /* 红色突出未命中率 */
            pre {{ background-color: #2d2d2d; color: #f1f1f1; padding: 15px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; font-family: "Courier New", Courier, monospace; }}
            .table-container {{ max-height: 600px; overflow: auto; border: 1px solid #ddd; }} /* 新增：表格滑动容器 */
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; white-space: nowrap; }} /* white-space: nowrap 防止表格内容换行 */
            thead th {{ background-color: #3498db; color: white; cursor: pointer; position: sticky; top: 0; z-index: 1; }} /* position: sticky 让表头吸顶 */
            th .sort-indicator {{ position: absolute; right: 10px; top: 50%; transform: translateY(-50%); }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .svg-container {{ width: 100%; overflow-x: auto; border: 1px solid #ddd; padding: 10px; background-color: #fff; box-sizing: border-box; }}
        </style>
        <script>
            function sortTable(n, tableId) {{
                let table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
                table = document.getElementById(tableId);
                switching = true;
                dir = "asc";
                let headers = table.getElementsByTagName("TH");
                for (let j = 0; j < headers.length; j++) {{
                    if (j !== n) {{
                        let indicator = headers[j].querySelector(".sort-indicator");
                        if (indicator) indicator.innerHTML = "";
                    }}
                }}
                let currentIndicator = headers[n].querySelector(".sort-indicator");
                if (currentIndicator.innerHTML.includes("▲")) {{
                    dir = "desc";
                    currentIndicator.innerHTML = "▼";
                }} else {{
                    dir = "asc";
                    currentIndicator.innerHTML = "▲";
                }}
                while (switching) {{
                    switching = false;
                    rows = table.rows;
                    for (i = 1; i < (rows.length - 1); i++) {{
                        shouldSwitch = false;
                        x = rows[i].getElementsByTagName("TD")[n];
                        y = rows[i + 1].getElementsByTagName("TD")[n];
                        let xContent = x.innerHTML.replace('%', '');
                        let yContent = y.innerHTML.replace('%', '');
                        if (!isNaN(parseFloat(xContent)) && !isNaN(parseFloat(yContent))) {{
                            xContent = parseFloat(xContent);
                            yContent = parseFloat(yContent);
                        }} else {{
                            xContent = xContent.toLowerCase();
                            yContent = yContent.toLowerCase();
                        }}
                        if (dir == "asc") {{
                            if (xContent > yContent) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }} else if (dir == "desc") {{
                            if (xContent < yContent) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }}
                    }}
                    if (shouldSwitch) {{
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount++;
                    }} else {{
                        if (switchcount == 0 && dir == "asc") {{
                            dir = "desc";
                            switching = true;
                        }}
                    }}
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>模型性能分析报告</h1>
                <p>目录: <code>{directory}</code></p>
            </div>

            <div class="card">
                <h2>宏观性能指标 (perf stat)</h2>
                <div class="metrics-grid">
                    <div class="metric"><div class="value ipc">{ipc}</div><div class="label">IPC (每周期指令数)</div></div>
                    <div class="metric"><div class="value time">{wall_time}s</div><div class="label">总耗时</div></div>
                    <div class="metric"><div class="value time">{task_clock}ms</div><div class="label">CPU 时间</div></div>
                    <div class="metric"><div class="value miss-rate">{branch_miss_rate}%</div><div class="label">分支预测失败率</div></div>
                    <div class="metric"><div class="value miss-rate">{llc_miss_rate}%</div><div class="label">L3 缓存未命中率</div></div>
                </div>
                <h3>完整 `perf stat` 输出:</h3>
                <pre>{perf_stat_raw}</pre>
            </div>

            <div class="card">
                <h2>算子性能详情 (OpenVINO Profiling)</h2>
                <p>总推理时间 (所有算子累加): <strong>{total_inference_time:.4f} ms</strong></p>
                <div class="table-container">
                    <table id="opTable">
                        <thead>
                            <tr>
                                <th onclick="sortTable(0, 'opTable')">排名<span class="sort-indicator"></span></th>
                                <th onclick="sortTable(1, 'opTable')">算子名<span class="sort-indicator"></span></th>
                                <th onclick="sortTable(2, 'opTable')">类型<span class="sort-indicator"></span></th>
                                <th onclick="sortTable(3, 'opTable')">执行时间 (ms)<span class="sort-indicator">▼</span></th>
                                <th onclick="sortTable(4, 'opTable')">占总时间比<span class="sort-indicator"></span></th>
                            </tr>
                        </thead>
                        <tbody>
                            {operator_table_rows}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <h2>火焰图 (perf record)</h2>
                <div class="svg-container">
                    {flamegraph_svg}
                </div>
            </div>

        </div>
    </body>
    </html>
    """
    
    op_perf_data = report_data['operator_perf']['operator_performance']
    total_time = report_data['operator_perf']['total_inference_time_ms']
    
    op_rows_html = ""
    for i, op in enumerate(op_perf_data, 1):
        percentage = (op['real_time_ms'] / total_time) * 100 if total_time > 0 else 0
        op_rows_html += f"""
        <tr>
            <td>{i}</td>
            <td>{op['name']}</td>
            <td>{op['type']}</td>
            <td>{op['real_time_ms']:.4f}</td>
            <td>{percentage:.2f}%</td>
        </tr>
        """
    
    km = report_data['perf_stat']['key_metrics']
    context = {
        "directory": report_data['directory'],
        "ipc": km.get('ipc', 'N/A'),
        "wall_time": km.get('wall_time', 'N/A'),
        "task_clock": km.get('task_clock', 'N/A'),
        "branch_miss_rate": km.get('branch_miss_rate', 'N/A'),
        "llc_miss_rate": km.get('llc_miss_rate', 'N/A'),
        "perf_stat_raw": report_data['perf_stat']['performance_counter_stats'],
        "total_inference_time": total_time,
        "operator_table_rows": op_rows_html,
        "flamegraph_svg": report_data['flamegraph_svg'],
    }
    
    final_html = html_template.format(**context)
    with output_path.open('w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"🎉 性能报告已成功生成: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="根据性能分析产物生成一个可视化的 HTML 报告。")
    parser.add_argument("directory", type=Path, help="包含分析结果文件的目录路径。")
    args = parser.parse_args()

    results_dir = args.directory
    if not results_dir.is_dir():
        print(f"错误: 提供的路径不是一个有效的目录 -> {results_dir}")
        return

    required_files = {
        "perf_stat": "perf_stat_report.txt",
        "operator_perf": "operator_perf.json",
        "flamegraph_svg": "flamegraph.svg",
    }
    
    for key, filename in required_files.items():
        if not (results_dir / filename).exists():
            print(f"错误: 缺少必需的文件 -> {filename}")
            return

    print("正在解析性能数据...")
    report_data = {
        "directory": results_dir.resolve(),
        "perf_stat": parse_perf_stat(results_dir / required_files['perf_stat']),
        "operator_perf": parse_operator_perf(results_dir / required_files['operator_perf']),
        "flamegraph_svg": embed_svg(results_dir / required_files['flamegraph_svg']),
    }

    print("正在生成 HTML 报告...")
    output_html_path = results_dir / "performance_report.html"
    generate_html_report(report_data, output_html_path)

if __name__ == "__main__":
    main()