#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:26:47 2020

@author: khuang
"""
from model import *
from data import *
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping, ReduceLROnPlateau, TensorBoard

NUM_EPOCHS=50
BATCH_SIZE = 2
NUM_TRAIN_IMAGES=120
NUM_VAL_IMAGES=28
NUM_TEST_IMAGES=12

# data augmentation
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    rescale = 1./255,
                    preprocessing_function = add_noise)

trainGene = trainGenerator(BATCH_SIZE,'data/PC3/train','image','label',data_gen_args,save_to_dir = None)
valGene = valGenerator(BATCH_SIZE,'data/PC3/val','image','label',data_gen_args)

# Unet deep learning model
model=UNet(img_shape=(256,256,1), out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
           dropout=0.3, batchnorm=True, maxpool=True, upconv=True, residual=False)
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

# setup callback function
early_stopping = EarlyStopping(patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
model_checkpoint = ModelCheckpoint('model.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir)

# Train the model
H=model.fit_generator(trainGene,steps_per_epoch=NUM_TRAIN_IMAGES/BATCH_SIZE,epochs=NUM_EPOCHS,
                      callbacks=[ model_checkpoint,tensorboard_callback,early_stopping,reduce_lr],
                      validation_data=valGene,
                      validation_steps=NUM_VAL_IMAGES/BATCH_SIZE)

# Predict the test data
test_path="data/PC3/test/img"
file_list=os.listdir(test_path)   
testGene = testGenerator(test_path,file_list)
results = model.predict_generator(testGene,NUM_TEST_IMAGES,verbose=1)
saveResult("data/PC3/test/predict",results,file_list)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H.epoch, H.history["loss"], label="train_loss")
plt.plot(H.epoch, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("loss_curve.png")