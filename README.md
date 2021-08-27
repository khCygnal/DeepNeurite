# DeepNeurite
A deep learning model to identify neurite structure in fluorescence images.
The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

## Overview
DeepNeurite utilizes a U-Net structure to tackle the fundamental issue of non-specific labeling of fluorescence probes. The model successfully suppresses the fluorescent signal contributed from the cancer cells in the co-culture conditions, and at the same time, does not sacrifice sensitivity such that even dim neurites are detectable. 

## Dependencies
Python 3.7, Tensorflow, Keras

## Installation
1. Clone this repository
2. Install dependencies
```
pip3 install -r requirements.txt
```
3. run main.py  

## Results
Use the trained model to do segmentation on test images, the result is statisfactory.

![xonaopt3-PC3-2-78-resize](https://user-images.githubusercontent.com/59890910/131156412-9da35493-d148-41e7-960a-1e5dea8cdeab.jpg)
![xonaopt3-PC3-2-78_predict](https://user-images.githubusercontent.com/59890910/131156249-8d993455-a739-4f8c-bbfc-d93d42bb3147.png)


