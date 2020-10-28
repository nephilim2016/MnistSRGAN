#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:35:11 2020

@author: nephilim
"""

import MnistSRGAN
import numpy as np
from matplotlib import pyplot,cm
import keras
import os 
import math
import skimage.transform
from numba import jit


def plot_images(generator,LR_image_test,show=False,step=0,model_name="gan"):
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict(LR_image_test)
    pyplot.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(LR_image_test.shape[0]))
    for i in range(num_images):
        pyplot.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        pyplot.imshow(image, cmap='gray')
        pyplot.axis('off')
    pyplot.savefig(filename)
    if show:
        pyplot.show()
    else:
        pyplot.close('all')

@jit(nopython=True)
def Load_Data(HR_image,LR_image,scale_factor):
    for idx in range(HR_image.shape[0]):
        LR_image[idx]=HR_image[idx,::scale_factor,::scale_factor]
    return LR_image
    
if __name__=='__main__':
    np.random.seed(10)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    (HR_image,_),(_,_)=keras.datasets.mnist.load_data()
    scale_factor=4
    LR_image=np.zeros((HR_image.shape[0],7,7))
    LR_image=Load_Data(HR_image,LR_image,scale_factor)
    
    HR_image_size=HR_image.shape[1]
    HR_image=np.reshape(HR_image,(-1,HR_image_size,HR_image_size,1))
    LR_image_size=LR_image.shape[1]
    LR_image=np.reshape(LR_image,(-1,LR_image_size,LR_image_size,1))
    
    HR_image=HR_image.astype('float')/255
    LR_image=LR_image.astype('float')/255
    LR_shape=(7,7,1)
    HR_shape=(28,28,1)
    batch_size=64
    epochs=500
    filepath='./DCGAN'
    MnistSRGAN_=MnistSRGAN.MnistSRGAN(LR_shape,HR_shape,scale_factor)
    generator,discriminator,adversarial=MnistSRGAN_.Train_GAN(LR_image,HR_image,filepath,epochs,batch_size=64)
    LR_image_test=LR_image[16:32]
    plot_images(generator,LR_image_test,show=True,step=0,model_name="gan")
    pyplot.figure()
    pyplot.imshow(LR_image_test[0,:,:,0],cmap=cm.gray)