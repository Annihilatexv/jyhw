import onnx

def set_batch_size(model_path: str, output_path: str, batch_size: int):
    """
    修改 ONNX 模型的批量大小。

    该脚本将模型输入和输出的第一个维度（假定为批量维度）设置为固定值。
    这对于动态批量大小（通常表示为 "batch" 或 "-1"）的模型尤其有用。

    参数:
    model_path (str): 原始 ONNX 模型的路径。
    output_path (str): 修改后要保存的新 ONNX 模型的路径。
    batch_size (int): 要设置的固定批量大小。
    """
    try:
        # 1. 加载 ONNX 模型
        model = onnx.load(model_path)
        graph = model.graph

        print(f"原始模型的输入: {[inp.name for inp in graph.input]}")
        print(f"原始模型的输出: {[out.name for out in graph.output]}")

        # 2. 修改模型输入的批量维度
        for inp in graph.input:
            # 获取输入的形状
            shape = inp.type.tensor_type.shape
            # 将第一个维度（批量大小）修改为固定值
            shape.dim[0].dim_value = batch_size
            print(f"已将输入 '{inp.name}' 的批量维度修改为 {batch_size}")

        # 3. 修改模型输出的批量维度
        for out in graph.output:
            # 获取输出的形状
            shape = out.type.tensor_type.shape
            # 将第一个维度（批量大小）修改为固定值
            shape.dim[0].dim_value = batch_size
            print(f"已将输出 '{out.name}' 的批量维度修改为 {batch_size}")
            
        # (可选) 检查模型是否有效
        onnx.checker.check_model(model)

        # 4. 保存修改后的模型
        onnx.save(model, output_path)
        print(f"模型已成功修改并保存到: {output_path}")

    except Exception as e:
        print(f"处理模型时出错: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 定义模型路径
    input_model_name = "/home/ghost/Code/OpenVINOCode/Model/mobileVitv/mobilevitv2_256x256.onnx"
    output_model_name = "/home/ghost/Code/OpenVINOCode/Model/mobileVitv/mobilevitv2_1x3x256x256.onnx"

    # 设置固定的批量大小
    fixed_batch_size = 1

    # 调用函数进行转换
    set_batch_size(input_model_name, output_model_name, fixed_batch_size)