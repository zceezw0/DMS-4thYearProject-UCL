## Driver Monitoring System based on CSCNN&DCN

### 1.Requirement

Necessary software packages:

- Python 3
- pytorch == 1.3
- torchvision == 0.4.1
- csv
- opencv-python
- sklearn
- PIL
- json

Python package of unspecified version can be installed directly through pip

### 2.Code introduction

```css
CSCNN&DCN
├── CSCNN&DCN.md
├── checkpoint
│   └── DCN_best.pth
├── dataset
│   └── DCN_dataset.py
├── model
│   └── network.py
├── preprocess
│   └── count_data.py
│   └── feature_engine.py
└── result
│   └── auc.png
│   └── loss.png
└── confusion_matrix.py
└── main.py
```

- CSCNN&DCN.md: **This markdown file**
- checkpoint: **stored the best weight during training**
- dataset: **our pytorch dataset**
- model: **our model**
- preprocess: **some script for preprocess**
- result: **some curve during training**
- confusion_matrix.py: **draw confusion matrix**
- main.py: **Main function entry**

### 3.How to use

#### 3.1 Train

```
python main.py
```

#### 3.2 Test

```
python confusion_matrix.py
```

