## Tired Driving Detecting 

### 1.Requirement

Windows + 摄像头 即可运行

必要的软件包：

- Python 3

- pytorch == 1.3

- torchvision == 0.4.1

- PySide2

- csv

- dlib == 19.6.1

- opencv-python

- scipy

- imutils

- onnx

- onnxruntime

未说明版本的python包，直接pip安装即可

### 2.How to use

默认参数：

```
parser.add_argument('-o','--output_path', default = "./output", type=str, help='store detection result')
parser.add_argument('-i','--input_video',default="",type=str,help='input video path')
parser.add_argument('-s', '--save_frame', action='store_true', help="save frames or not")
parser.add_argument('-t', '--time_step', default=30, type=int, help="interval per detection")
```

output_path 表示输出文件的默认路径（如果不存在会自动创建）

input_video 表示输入源，如果是""表示使用的是摄像头输入，如果要使用video输入，使用video的绝对路径

save_frame 表示是否保存摄像头获取到的原始图片，布尔值

time_step 表示保存结果的时间间隔，秒为单位

以下命令为例，表示把结果保存在"./output_path"中，且保存所有原始图像，每十秒保存一次输出值。

```
python main.py -o "./output_path" -s -t 10
```



修改这一行，把anno.csv保存为你想要的名字，对于第一次摄像头采集，取名为1.csv，第二次，取名为2.csv，以此类推，采集你想要的实验数据。

```
anno_file = os.path.join(output_path, "anno.csv")
```

