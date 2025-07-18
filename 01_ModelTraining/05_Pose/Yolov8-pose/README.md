# Yolov8-pose

## 环境安装
1. Clone repo and install [requirements.txt](requirements.txt) in a python>=3.8.0, including pytorch>=1.8
   ```
   git clone https://github.com/AIDrive-Research/EdgeAI-Toolkit.git
   cd EdgeAI-Toolkit/train/pose/yolov8-pose
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

## 模型导出

1. ONNX导出

   ```bash
   yolo mode=export model=yolov8n-pose.pt format=onnx opset=12 simplify=True
   ```

## 模型转换
**注意**：该操作适用于KS968产品，KS988无需执行。

1. 模型转换

   运行：

   ```
    python convert.py onnx_model_path platform fp output_rknn_path
   ```

   其中：

   - onnx_model_path：训练后导出的onnx模型文件位置
   - platform：[rk3568,rk3588]
   - fp：fp代表不量化
   - output_rknn_path：转换后模型的保存路径