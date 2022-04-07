## Camera Monitoring System 

### 1.Requirement

Windows + camera to run

Necessary software packages:

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

Python package of unspecified version can be installed directly through pip

### 2.How to use

Default parameters:

```
parser.add_argument('-o','--output_path', default = "./output", type=str, help='store detection result')
parser.add_argument('-i','--input_video',default="",type=str,help='input video path')
parser.add_argument('-s', '--save_frame', action='store_true', help="save frames or not")
parser.add_argument('-t', '--time_step', default=30, type=int, help="interval per detection")
```

- output_ path indicates the default path of the output file (automatically created if it does not exist)

- input_ video indicates the input source. If "" indicates the camera input is used. If you want to use video input, use the absolute path of video

- save_ frame indicates whether to save the original picture obtained by the camera, Boolean value

- time_ step indicates the time interval for saving the results, in seconds

  Taking the following command as an example, it means to save the results in ". / output_path", save all original images, and save the output value every ten seconds.

```
python main.py -o "./output_path" -s -t 10
```

​	Modify this line and put anno CSV is saved as the name you want. For the first camera acquisition, it is named 1 CSV, the second time, named 2 CSV, and so on, collect the experimental data you want.

```
anno_file = os.path.join(output_path, "anno.csv")
```

