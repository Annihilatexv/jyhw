import openvino as ov 

# 转换为 openvino 模型
modelPath = "../Model/onnx/yolov8n.onnx"
ovModel = ov.convert_model(modelPath)

ov.save_model(ovModel, "../Model/openvino/yolov8n_fp32.xml", compress_to_fp16 = False)

print("---------模型输入输出大小-------")
print(ovModel.inputs)
print(ovModel.outputs)