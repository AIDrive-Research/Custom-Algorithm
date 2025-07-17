# Resnet18

## 环境安装

   1. Clone repo and install [requirements.txt](train/classify/resnet18/requirements.txt) in a python>=3.8.0 environment, including pytorch>=1.8

      ```bash
      git clone https://github.com/AIDrive-Research/Custom-Algorithm.git
      cd Custom-Algorithm/01_ModelTraining/01_Detection/Yolov5
      pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
      ```

## 数据准备

数据集结构如下：

```bash
|-dataset
   |-images
      |- 0
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |- 1
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |...
   |-train
      |- 0
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |- 1
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |...
   |-test
      |- 0
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |- 1
         |-xxxxx.jpg
         |-xxxxx.jpg
         |...
      |...
```

## 数据划分

将数据集放置于dataset下的images文件夹下，按照类别进行放置，0文件夹表示第一类，1文件夹表示第二类，依次类推。执行如下代码，进行数据集划分，得到训练集与验证集。注：将`src_path`，`train_path`，`test_path`修改为自己数据集路径。

```bash
python split_train_test.py
```

## 模型训练

将`train.py`配置参数中的`n_classes`修改为自己数据集类别数，将`train_dataset`，`test_dataset`修改为自己的数据路径。执行如下代码。

```
python train.py
```

**模型导出**

将`convert_onnx.py`中的`input_path`，`output_path`分别修改为自己的模型权重路径，`onnx`文件导出路径。执行如下代码。

```bash
python convert_onnx.py
```

**模型量化**  
**注意**：该操作适用于KS968产品，KS988无需执行。

1. 在训练集中随机选取图片进行模型量化，精度校准，数量在80-120之间，目录结构如下：

   ```
    images:
    	xxx.jpg
   ```

2. 把图片路径保存至xxx.txt

   ```
    find ./images/ -name "*.jpg">custom.txt
   ```

3. 模型量化

   修改convert.py：

   - DATASET_PATH：量化图片路径

   运行：

   ```
    python convert.py onnx_model_path platform i8/fp output_rknn_path
   ```

   其中：

   - onnx_model_path：训练后导出的onnx模型文件位置
   - platform：[rk3562,rk3566,rk3568,rk3588]
   - i8/fp：i8代表使用图片量化；fp代表不量化
   - output_rknn_path：量化后模型的保存路径