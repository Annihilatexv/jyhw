import onnx
from onnx import numpy_helper

def cleanup_model_for_quantization(input_model_path: str, output_model_path: str):
    """
    预处理 ONNX 模型以解决 "Expected bias ... to be an initializer" 警告。

    该脚本会查找作为其他节点输入的 Constant 节点，并将其转换为
    图的 initializer，从而让量化工具能更好地处理它们。

    参数:
    input_model_path (str): 原始 ONNX 模型的路径。
    output_model_path (str): 清理后要保存的新 ONNX 模型的路径。
    """
    print(f"开始预处理模型: {input_model_path}")
    model = onnx.load(input_model_path)
    graph = model.graph

    # Constant节点名称 -> ONNX ValueInfo 对象
    all_nodes_as_inputs = set()
    for node in graph.node:
        for input_name in node.input:
            all_nodes_as_inputs.add(input_name)

    # 找到所有作为其他节点输入的 Constant 节点
    # 这些是需要被转换的候选节点
    constants_to_convert = []
    for node in graph.node:
        if node.op_type == 'Constant' and node.output[0] in all_nodes_as_inputs:
            constants_to_convert.append(node)
    
    if not constants_to_convert:
        print("未找到需要转换的 Constant 节点。模型无需修改。")
        # 如果不需要修改，可以直接保存一份副本或不操作
        onnx.save(model, output_model_path)
        return

    print(f"找到 {len(constants_to_convert)} 个 Constant 节点需要转换为 initializer。")

    # 执行转换
    for const_node in constants_to_convert:
        # 1. 从 Constant 节点中提取权重数据
        tensor_value = numpy_helper.to_array(const_node.attribute[0].t)
        
        # 2. 创建一个新的 initializer
        initializer_tensor = numpy_helper.from_array(tensor_value, const_node.output[0])
        
        # 3. 将 initializer 添加到图中
        graph.initializer.append(initializer_tensor)
        
        # 4. 从图中移除这个 Constant 节点
        graph.node.remove(const_node)

    # 保存清理后的模型
    onnx.save(model, output_model_path)
    print(f"预处理完成！清理后的模型已保存至: {output_model_path}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 这是你之前固定了 batch=1 的模型
    original_model = "/home/ghost/Code/OpenVINOCode/Model/mobileVitv/mobilevitv2_1x3x256x256.onnx"
    # 定义预处理后模型的保存路径
    preprocessed_model = "/home/ghost/Code/OpenVINOCode/Model/mobileVitv/mobilevitv2_1x3x256x256_preQuant.onnx"

    cleanup_model_for_quantization(original_model, preprocessed_model)