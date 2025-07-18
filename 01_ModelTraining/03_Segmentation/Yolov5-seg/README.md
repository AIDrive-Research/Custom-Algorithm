# Yolov5-seg

## 环境安装

   1. Clone repo and install [requirements.txt](requirements.txt) in a python>=3.8.0, including pytorch>=1.8
      ```
       git clone https://github.com/AIDrive-Research/Custom-Algorithm.git
       cd Custom-Algorithm/01_ModelTraining/03_Segmentation/Yolov5-seg
       pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
      ```

## 数据准备

1. 制作数据集，目录结构如下：

   ```
    JPEGImages:
       xxx.jpg
    labels:
    	xxx.txt
   ```

2. 将数据拆分训练集和验证集

   ```
    python tools/1_split_train_val.py --input-images 图片文件路径 --input-labels YOLO格式标注文件存储路径 --output YOLO数据集存储路径
   ```

   目录结构如下图:

   ```
    images:
    	train
    		xxx.jpg
    	val
    		xxx.jpg
    labels:
    	train
    		xxx.txt
    	val
    		xxx.txt	
   ```

3. 新建训练数据yaml文件，参考data/coco128.yaml编辑自己的yaml文件

   ```
    cd data
    cp coco128.yaml custom.yaml
   ```

   修改custom.yaml:

   - path: 2中YOLO数据集存储路径
   - train: images/train
   - val: image/val
   - names: 类别名称

## 训练

1. 单卡训练

   ```
    python segment/train.py --model yolov5s-seg.pt --data data/custom.yaml --epochs 300 --img 640 --batch-size 128
   ```

2. 推荐使用多卡训练

   ```python
    python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --model yolov5s-seg.pt --data data/custom.yaml --epochs 300 --img 640 --device 0,1,2,3
   ```

## 模型导出

1. ONNX导出

   ```
   python export.py --weights xxx.py --include onnx --simplify --opset 12
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
