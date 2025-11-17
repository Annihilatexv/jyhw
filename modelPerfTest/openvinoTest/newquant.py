# --- 4. 主执行函数 (已修改为仅量化和保存) ---
def main():
    # ... (前面的检查和路径设置不变) ...
    output_ir_path = ONNX_MODEL_PATH.with_name(f"{ONNX_MODEL_PATH.stem}_int8_nncf.xml")
    INPUT_SIZE = (640, 640) # YOLOv8 模型的标准输入宽高

    try:
        core = ov.Core()
        
        # --- 1. 加载原始FP32模型 ---
        print(f"\n[1/5] 正在加载FP32模型: {ONNX_MODEL_PATH}")
        model_fp32 = core.read_model(ONNX_MODEL_PATH)
        
        # 获取输入层名称
        input_name = model_fp32.input(0).any_name
        
        # 🚀 新增步骤：固定动态输入形状
        print(f"\n[2/5] 正在固定模型输入形状为: [1, 3, {INPUT_SIZE[0]}, {INPUT_SIZE[1]}]")
        # 假设模型的输入层是第一个 (索引 0)
        # 形状应该与您的 `preprocess_image` 函数生成的张量形状完全匹配
        target_shape = [1, 3, INPUT_SIZE[0], INPUT_SIZE[1]]
        
        # 使用 reshape 来固定输入形状
        model_fp32.reshape({input_name: target_shape})
        
        # 检查是否成功
        print(f"    新的输入形状: {model_fp32.input(0).shape}")
        
        # --- 3. 准备校准数据集 ---
        print("\n[3/5] 正在准备校准数据集...")
        # ... (数据加载器和准备代码不变) ...
        image_files = sorted([p for p in CALIBRATION_DATA_DIR.glob("**/*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        if not image_files:
            print(f"错误: 在 {CALIBRATION_DATA_DIR} 中未找到任何图片文件。")
            return
            
        image_loader = ImageLoader(image_files)

        # 定义一个转换函数，将数据加载器的输出包装成NNCF期望的字典格式
        def transform_fn(data_item):
            # 注意: 这里的键必须是 reshape 之前获取到的 `input_name`
            return {input_name: data_item}

        calibration_dataset = nncf.Dataset(image_loader, transform_fn)

        # --- 4. 执行INT8量化 ---
        print("\n[4/5] 开始进行INT8量化 (这可能需要几分钟)...")
        quantized_model = nncf.quantize(
            model=model_fp32, # 使用形状固定的模型
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE, 
        )
        
        # --- 5. 保存量化后的模型 ---
        print(f"\n[5/5] 正在保存INT8模型到: {output_ir_path}")
        ov.serialize(quantized_model, str(output_ir_path))
        
        print("\n量化并保存成功!")
        print(f"  - XML 文件: {output_ir_path}")
        print(f"  - BIN 文件: {output_ir_path.with_suffix('.bin')}")

    except Exception as e:
        print(f"\n在处理过程中发生错误: {e}")
