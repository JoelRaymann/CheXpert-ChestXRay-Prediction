# CheXpert-ChestXRay-Prediction
A A.I Approach towards prediction of 14 disease classification with CheXpert Dataset

# Introduction
This is a repo that contains implementation of different A.I Model powered by TF 2.0
The first iteration uses the densenet-121 model which is recommended by the CheXpert
Paper. 

# Pre-requisities
We need a set of few requirements before we can use these code. First, we need to download
the CheXpert Dataset from the Stanford ML website. You can get it from [here](https://stanfordmlgroup.github.io/competitions/chexpert/).
Next, we need the following software and hardware requirements
## Software Requirements:
### For Windows - 7, 8, 8.1, 10 - Recommended OS
 - Python 3.5.x or greater or Python 2.7.x or greater but python 2.7.x series not recommended
 - Nvidia Drivers Updated (Only needed if you are training the model from scratch)
 - CUDA - 10.0 (Only needed if you are training the model from scratch)
 - CuDNN - 7.x (Only needed if you are training the model from scratch)

### For Linux - Ubuntu - 16, 18; Fedora; OpenSUSE; RHEL
 - Python 3.5.x or greater or Python 2.7.x or greater but python 2.7.x series not recommended
 - Nvidia Drivers Updated (Only needed if you are training the model from scratch)
 - CUDA - 10.0 (Only needed if you are training the model from scratch)
 - CuDNN - 7.x (Only needed if you are training the model from scratch)

### For MacOS - Sierra, High Sierra, Mojave - Training is not supported ;_;
 - Python 3.5.x or greater or Python 2.7.x or greater but python 2.7.x series not recommended

## Hardware Requirements:
 - Atleast any core with atleast 4 logical cores
 - Atleast 8 GB RAM 
 - Nvidia GPU Card needed for training

## Python Packages Requirements
Next, we need to get all the python packages. For that, we have prepared two requirements.txt file.
### For CPU Only
In order to run the program in CPU alone system. Use the following command to 
install all the packages
```bash
pip install -r requirements-only-cpu.txt
```
### For Nvidia GPU
In order to run the program in GPU included system.Use the following command to 
install all the packages
```bash
pip install -r requirements-gpu.txt
```

# How to execute
## Step 1: Get the Dataset
 - As mentioned, Get the dataset from the official website from [here](https://stanfordmlgroup.github.io/competitions/chexpert/)
 - pull this git repo and extract the cheXpert dataset in a new folder called
 `dataset`. The final extracted path should be like `./dataset/CheXpert-v1.0-small/`

## Step 2: Clean the Dataset
- Use the clean_data.py file to clean the dataset's meta data i.e. the train.csv and
valid.csv file.
- To do this give a python call in your terminal like
```bash
python clean_data.py -i "./dataset/train.csv" -o "./dataset/train_clean.csv" -n 0
```
- Do the same for valid.csv file too. This is done like
```bash
python clean_data.py -i "./dataset/valid.csv" -o "./dataset/valid_clean.csv" -n 0
```

## Step 3: Training Model
- To train the model, we need to configure the config.yaml file. 
Set your own configuration in it. Set your batch_size and all there.
- Once set. You have 2 options.
### Option 1: Fresh Train.
- This will train from scratch. To do this. Open your terminal and give
a python call to train_model.py like
```bash
python train_model.py -C config.yaml
```
- This will automatically train and save the weights in "./models/" folder
We can then use the weights for the testing
### Option 2: Re-train Model / Resume Training
- This will take the models weights and retrain it from a particular epoch.
To do this, give a python call in your terminal like
```bash
python train_model.py -L <load model file path> -R <resume epoch> -W 1 -C <config yaml file path>
```
- This will give the new weights in the ```./models/saved_model_<resume_epoch>/``` folder

## Step 4: Testing Model
Coming Soon.

# Results:
Our Previous results with NIH dataset are as follows:


| Model Name |	Dataset	|	Methodology	| Train Accuracy (20 Epochs) | Test Accuracy (20 Epochs) | Remarks |
| :----------- | :--------- | :--------------------------------------- | :--------- | :--------- | :-------------------------------- |
| DenseNet - 121 | NIH DATASET | MultiLabel | 83% (Average) | 82% (Average) | Heavy class imbalance towards “No Findings” |
| SE-DenseNet-121 | NIH DATASET | Multilabel |	85%(Average) | 81.3%(Average) |	Heavy class imbalance towards “No Findings” |
| SE-DenseNet-121 with Focal Loss |	NIH DATASET	| MultiLabel | 89%(Average) | 82%(Average) | Heavy class imbalance towards “No Findings”. Even Focal Loss can’t fix this problem |
| DenseNet – 121 | NIH DATASET | Binary Classification ;0 => not infected; 1 => infected | 69% | 68% | Problem is with dataset. Better dataset needed |
| SE-DenseNet – 121 | NIH DATASET | Binary Classification; 0 => not infected;1 => infected	| 70% | 68.467% | Problem is with dataset. Better dataset needed |
| NasNet Mobile – 5M params | NIH DATASET | Binary Classification; 0 => not infected;1 => infected	| 73% |	72.7%	| Problem is with dataset. Better dataset needed. Small improvement due to better model |
| NasNet Large – 88M params | NIH DATASET | Binary Classification; 0 => not infected ;1 => infected	| 78%	| 75.65% | Overfitting Problem Noted. Implies nothing can be done. Problem Classified as A.I. Hard |

# LICENSE
BSD 3-Clause License

Copyright (c) 2019, Joel Raymann, Harikrishnan V. S
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# THANK YOU
Wait for more updates. Further Updates in progress
 - Introduction of GRAD-CAM Heat Map 
 - Portability to Conversion to .tflite for all platform portabality
 - Andriod App that runs this model
