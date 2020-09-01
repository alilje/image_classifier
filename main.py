# Overview
# NETREX - Created by Anne Lilje and Karina Munoz
# Copyright (c) 2020 Sandia National Laboratories, All rights reserved

"""This code contains code that has been modified and merged from the following github
   sites:

   https://github.com/gsurma/image_classifier
   MIT License -See licenses/LICENSE_GSURMA.txt
   https://github.com/rishabhjainps/Facial-Expression-Recognition
   MIT License -See licenses/LICENSE_RISHABHJAINPS.txt
   https://github.com/gabrielkirsten/cnn_keras
   MIT License -See licenses/LICENSE_GABRIALKIRSTEN.txt

from a couple of github.com sites as well as
   codes written by Anne Lilje (Org 8724) and Karina Munoz (Org 8722)

   Please see the following sites from which code was contributed. Both sites
   are under MIT licensing, and as required the LICENCE.md files are included:
"""
#--------------------------------BEGIN NETREX--------------------------#
from skimage import io
import numpy as np
import tensorflow as tf
from tensorflow import keras

import gov.sandia.netRex.models.ImageClassifierCNN as classfy
import numpy as np
from optparse import OptionParser # Import OptionParser
import os.path
from matplotlib import pyplot as plt
from skimage.util import img_as_ubyte
import gov.sandia.netRex.control.hyperparameters as hyPrms

def image_intensity_hist(img):
    (head,tail) = os.path.splitext(os.path.realpath(img))
    (head,filename) = os.path.split(head)                        #filename = img[:-4]
    imageAsIs = io.imread(img, as_gray = True)
    image = img_as_ubyte(imageAsIs)
    plt.figure()
    plt.hist(image.flatten(), bins=256, range=(0, image.max()))
    plt.title("Grayscale Histogram  (" +str(img)+')')
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.savefig("Hist_"+str(filename)+".png", format ='png')
    image_rows, image_columns = image.shape
    max_intensity = image.max()
    min_intensity = image.min()
    mean_intensity = image.mean()
    intensity_standardDeviation = np.ndarray.std(image)
    intensity_variance = intensity_standardDeviation**2
    image_dtype = image.dtype

    return image_dtype, image_rows, image_columns, min_intensity, \
           max_intensity, mean_intensity, intensity_variance

def main(trainer=None):
    parser = OptionParser()  # Instantiate OptionParser
    parser.description = "Defines switches for accepting the name of the input image at the commandline"
    #parser.add_option("-d", "--directory",
                 # dest="imageDir",
                 # help="This is a directory of images")
    parser.add_option("-n", "--numclasses",dest="numClasses",help="Number of Classes")
    parser.add_option("-e", "--epochs",dest="epochs",help="Number of Epochs")
    parser.add_option("-b", "--batchSize",dest="batchSize",help="The preferred batch size for fitting the network")
    parser.add_option("-s", "--imageSize",dest="imgSz",help="The side dimension of the images. In this case rows must equal columns")
    parser.add_option("-a", "--bands", dest="bnds", help="This is the number of bands in the image")
    parser.add_option("-t", "--testSize",dest="testSize",help="Size of the test")
    (options, args) = parser.parse_args()

    #self,anImageSize, anEpoch, aBatchSize, aTestSize, numBands
    #run = hyPrms.InitialParams(options.imageDimension, options.epochs, options.batchSz, options.testSize, options.bnds)
    run = classfy.Classifiers(500,5, 1, 30, 1)
    theModel = run.models()
    run.trainerGen(theModel)





if __name__ == "__main__":
    main()


