import openvino as ov 

core = ov.Core()

modelPath = "../Model/onnx/yolov8n.onnx"
ovModel = ov.convert_model(modelPath)

compileModel = core.compile_model(ovModel, "CPU")

print("---------模型输入输出大小-------")
print(ovModel.inputs)
print(ovModel.outputs)
print("---------编译模型输入输出大小-------")
print(compileModel.inputs)
print(compileModel.outputs)

