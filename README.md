# Fastai-Manager
Maximize the convenience of fastai experiment, additionally try eXplainable AI

## Introduction

[fastai](https://github.com/fastai/fastai)는 Pytorch 기반의 간편한 딥러닝 라이브러리로 높은 편의성을 제공합니다. (좋은 [강의](https://course.fast.ai/)도 제공)

하지만 반복되는 실험과 결과 해석 측면에서는 무엇인가 부족한 부분도 느끼게 되어, 해당 부분을 보충하여 클래스 형태로 구성하게 되었습니다.

현재 **Image Classification and Regression** 파트를 담당하는 `FastVision` 쪽이 동작하고 있습니다.

- V0.0.1: Only Convolutional Neural Network arch and ResNet

## Computer Vision

- 어떤 아키텍쳐를 사용할 것인가? (**ConvNet** or **Unet**)
  + **ConvNet**: 어떤 작업을 수행할 것인가? (**Classification** or **Regression**), `Size(Input) != Size(Output)`
  + **Unet**: **Image Segmentation**, **Colorization **, **Super-Resolution**, `Size(Input) == Size(Output)` 

### Classification using Image 
사전 준비 (클래스 생성)와 학습 그리고 해석 과정이 존재합니다. 

#### 사전준비 example

필요한 파라미터 값들을 dictionary 형태로 만들어줍니다.

```python
parameters = {
    "path" : root_dir  / 'EcoSatellite' / 'data' / "Land",
    "bs" : 32, 
    "size" : 256, 
    "valid_pct" : 0.2,

    "metrics" : [accuracy],
    "pretrained" : True, 
    "load_dir": None
}

vis = FastVision(models.resnet34, parameters, task="classification")
```
- For `DataBunch`
  + `path`: 이미지 분류 작업을 위한 데이터들이 모여있는 상위 폴더로, 하위 폴더는 각각 분류하고자 하는 클래스 폴더들로 구성되어 안에 이미지들이 존재해야합니다.
  + `bs`: batch size 로, 한번 학습할때 몇개의 데이터를 볼지 결정합니다.
  + `size`: 이미지 한 축의 픽셀 크기 입니다.
  + `valid_pct`: 데이터의 train-validation 분할을 위한 validation set **%** 입니다.
- For `Learner`
  + `metrics`: 모델의 평가 지표입니다.
  + `pretrained`: 나중에 파이토치에서 미리 구성된 아키텍처를 이용할 때, 사전학습된 파라미터를 사용할지를 결정합니다.
  + `load_dir`: 학습된 결과물이 있다면 이에 대한 경로를 지정하여 바로 불러올 수 있으며, `None` 값이면 무시합니다.


#### 학습 example

```python
vis.find_alpha()
vis.fit_model_cyc(3, 5e-03)
```
- `find_alpha()`: Cyclical Learning과 관련된 learning rate alpha 탐색 과정으로 alpha에 따른 loss를 보여주며, 최적의 alpha 선택을 보조합니다.
- `fit_model_cyc()`: 몇 epoch를 돌지, 얼마만큼의 learning rate alpha를 줄지 선택하여 학습을 진행합니다.


### Regression using Image

이미지를 활용한 회귀 모델을 생성 및 학습하도록 합니다.

#### 사전준비 example

필요한 파라미터 값들을 dictionary 형태로 만들어줍니다.

```python
parameters = {
    "df": "meta_dhs.csv",
    "colnames": ["image_name", "wealth"],
    "parent_path": data_path / "repli",
    "folder": "poverty",
    "bs" : 32, 
    "size" : 256, 
    "valid_pct" : 0.2,

    "metrics" : [root_mean_squared_error, r2_score],
    "pretrained" : True, 
    "load_dir": None,

    "final_size": 1024,
    "y_range": None

    
}

vis = FastVision(models.resnet34, parameters, task="regression")
```
- For `DataBunch`
  + `df`: 이미지 이름과 이에 대한 실수형 라벨이 열 형태로 존재하는 데이터프레임의 이름으로, 자동으로 다른 경로 변수들과 조합되어 사용됩니다.
  + `colnames`: 이미지 이름과 이에 대한 실수형 라벨의 열이름입니다.
  + `parent_path`: 라벨링을 표시한 데이터프레임이 존재하는 폴더 이름으로, 특정 이름을 가진 하위 폴더에 이미지들이 저장되어 있습니다.
  + `folder`: 이미지 회귀 작업을 위한 이미지들이 모여있는 폴더입니다.
  + `bs`: batch size 로, 한번 학습할때 몇개의 데이터를 볼지 결정합니다.
  + `size`: 이미지 한 축의 픽셀 크기 입니다.
  + `valid_pct`: 데이터의 train-validation 분할을 위한 validation set **%** 입니다.
- For `Learner`
  + `metrics`: 모델의 평가 지표입니다.
  + `pretrained`: 나중에 파이토치에서 미리 구성된 아키텍처를 이용할 때, 사전학습된 파라미터를 사용할지를 결정합니다.
  + `load_dir`: 학습된 결과물이 있다면 이에 대한 경로를 지정하여 바로 불러올 수 있으며, `None` 값이면 무시합니다.
  + `final_size`: 기존 모델에서 Final Layer 계층만 분리하고 회귀 모델로 교쳬하기 위한 중간 계층의 output 사이즈입니다. 
  + `y_range`: ...



#### 학습 example

```python
vis.find_alpha()
vis.fit_model_cyc(3, 5e-03)
```
- `find_alpha()`: Cyclical Learning과 관련된 learning rate alpha 탐색 과정으로 alpha에 따른 loss를 보여주며, 최적의 alpha 선택을 보조합니다.
- `fit_model_cyc()`: 몇 epoch를 돌지, 얼마만큼의 learning rate alpha를 줄지 선택하여 학습을 진행합니다.






