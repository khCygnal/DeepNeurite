# DeepNeurite
A deep learning model to identify neurite structure in fluorescence images developed by [Cygnal therapeutics](https://cygnaltx.com/).
The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

## Overview
DeepNeurite utilizes a U-Net structure to tackle the fundamental issue of non-specific labeling of fluorescence probes. The model successfully suppresses the fluorescent signal contributed from the cancer cells in the co-culture conditions, and at the same time, does not sacrifice sensitivity such that even dim neurites are detectable. 

## Dependencies
Python 3.7, Tensorflow-gpu 1.15.0, Keras 2.3.1

## Installation
1. Clone this repository
2. Create a virtual environment
```
conda create -n Deepneurite python=3.7
conda activate Deepneurite
```
3. Install dependencies
```
conda install --file requirements.txt
```
4. run main.py  
```
python main.py
```

## Results
Use the trained model to do segmentation on test images, the result is statisfactory.

![xonaopt3-PC3-2-78-resize](https://user-images.githubusercontent.com/59890910/131156412-9da35493-d148-41e7-960a-1e5dea8cdeab.jpg)
![xonaopt3-PC3-2-78_predict](https://user-images.githubusercontent.com/59890910/131156249-8d993455-a739-4f8c-bbfc-d93d42bb3147.png)


