# export DNNL_VERBOSE=1
/home/ghost/openvino_cpp_samples_build/intel64/Release/benchmark_app \
  -m /home/ghost/Code/OpenVINOCode/Model/yolov8n/yolov8n_int4.onnx \
  -d CPU -api sync -pc > int4_pc_yolov8n.txt 
  # 2>&1

E:\OpenVINO\openvino\bin\intel64\Release\benchmark_app.exe -m D:\Code\OpenVINOCode\Model\yolov8n\yolov8n.xml -d CPU -api sync -pc -niter 1000 > fp32_pc_yolov8n.txt 

E:\OpenVINO\openvino\bin\intel64\Release\benchmark_app.exe -m D:\Code\OpenVINOCode\Model\yolov8n\yolov8n_int8.onnx -d CPU -api sync -pc -niter 1000 > int8_pc_yolov8n.txt 


vtune -collect memory-access -- ^
E:\OpenVINO\openvino\bin\intel64\Release\benchmark_app.exe -m D:\Code\OpenVINOCode\Model\yolov8n\yolov8n_int8_nncf.xml -d CPU -api sync -pc -niter 1000 -hint latency

vtune -collect hotspots -knob sampling-interval=1 -- ^
E:\OpenVINO\openvino\bin\intel64\Release\benchmark_app.exe -m D:\Code\OpenVINOCode\Model\yolov8n\yolov8n_int8.onnx -d CPU -api sync -pc -hint latency


vtune -collect memory-access -- ^
E:\OpenVINO\openvino\bin\intel64\Release\benchmark_app.exe -m D:\Code\OpenVINOCode\Model\yolov8n\yolov8n_int8.onnx -d CPU -api sync -pc -hint latency


vtune -collect hotspots -- ^
python D:\Code\OpenVINOCode\modelPerfTest\openvinoTest\int8Test.py


