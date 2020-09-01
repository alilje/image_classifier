
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras.preprocessing.image as procimg
from livelossplot import plot_losses
from livelossplot import PlotLosses
import tensorflow.keras.models as mods
import keras.optimizers as opmz
import keras.layers as lyrs
import keras.engine
import keras.layers.convolutional as conv
import os
#from tensorflow.keras.preprocessing import kerasProc
#from keras.layers.normalization import BatchNormalization
#import keras.layers.core.Activation
#import keras.layers.convolutional
#import keras.layers.pooling.GlobalAveragePooling2D
#import keras.layers.core.Reshape
#import keras.layers.merge.Multiply
#import efficientnet.model.get_dropout
#import keras.layers.merger.Add

import keras.callbacks as clb
import random as rand
#ToDo: Please move the argument in this init to a more reasonable form
class Classifiers:

    def __init__(self,anImageSize, anEpoch, aBatchSize, aTestSize, numBands):
        #TODO: You n3eed to add these to a configuration file.
        # Recommendation - add directory config and use config module.

        self.trainglogsFile = "training_logs.csv"
        self.modelSummaryFile = "model_summary.txt"
        self.modelFile = "normal-vs-faulty.h5"

        # Data
        self.path = "data/"
        self.training_data_dir = self.path + "training"
        self.validation_data_dir = self.path + "validation"
        self.test_data_dir = self.path + "test"

        # Hyperparams
        self.imageSize = anImageSize
        self.imageWidth, self.imageHeight = self.imageSize, self.imageSize
        self.epochs = anEpoch
        self.batchSize = aBatchSize
        self.testSize = aTestSize
        self.numBands = numBands
        self.inputShape = (self.imageSize, self.imageSize, self.numBands)

    def models(self):

        model = keras.Sequential()

        model.add(conv.Conv2D(32, 3, 3, padding='same', input_shape=self.inputShape, activation='relu'))
        model.add(conv.Conv2D(32, 3, 3, padding='same', activation='relu'))
        model.add(lyrs.pooling.MaxPooling2D(pool_size=(2, 2)))

        model.add(conv.Conv2D(64, 3, 3, padding='same', activation='relu'))
        model.add(conv.Conv2D(64, 3, 3, padding='same', activation='relu'))
        model.add(lyrs.pooling.MaxPooling2D(pool_size=(2, 2)))

        #model.add(conv.Conv2D(128, 3, 3, padding='same', activation='relu'))
        #model.add(conv.Conv2D(128, 3, 3, padding='same', activation='relu'))
        #model.add(lyrs.pooling.MaxPooling2D(pool_size=(2, 2)))

        #model.add(conv.Conv2D(256, 3, 3, padding='same', activation='relu'))
        #model.add(conv.Conv2D(256, 3, 3, padding='same', activation='relu'))
        #model.add(lyrs.pooling.MaxPooling2D(pool_size=(2, 2)))


        model.add(lyrs.Flatten())
        model.add(lyrs.Dense(256, activation='relu'))
        model.add(lyrs.Dropout(0.5))

        model.add(lyrs.Dense(256, activation='relu'))
        model.add(lyrs.Dropout(0.5))

        model.add(lyrs.Dense(1))
        model.add(lyrs.Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                  optimizer=opmz.RMSprop(lr=0.0001),
                  metrics=['accuracy'])

        with open(self.modelSummaryFile,"w") as fh:
            model.summary(print_fn=lambda line: fh.write(line + "\n"))

        return model

    def trainerGen(self,aModel):
        print("Creating Image Data Generator------------------------------------------------")
        print(os.getcwd())
        train_datagen = procimg.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False, # TODO: Note this gives warning on overriding
                                                 #  featurewise_center? Interesting you need to
                                                 #  look at this...
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=(1.0,20.0),
            shear_range=0.0,
            zoom_range=[1-0.3, 1+0.3],
            channel_shift_range=0.0,
            fill_mode="nearest",
            cval=0.0,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0,
            dtype=None,
        )
        print("Train DataGen--------------------)")
        print(str(train_datagen))
        test_datagen = procimg.ImageDataGenerator(rescale=1./255)

        print("Test Data Gen-------------------")
        print(str(test_datagen))

        train_generator = train_datagen.flow_from_directory(
            'data/training',
            target_size=(500, 500),
            batch_size=32,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(500, 500),
            batch_size=32,
            class_mode='binary')
        print("Validation Generator ------------------------------------------")
        print(str(validation_generator))

        aModel.fit(
            train_generator,
            steps_per_epoch=20,
            epochs=5,
            validation_data=validation_generator,
            validation_steps=10)


'''


# tf.keras.preprocessing.image_dataset_from_directory(
#     directory,
#     labels="inferred",
#     label_mode="int",
#     class_names=None,
#     color_mode="rgb",
#     batch_size=32,
#     image_size=(256, 256),
#     shuffle=True,
#     seed=None,
#     validation_split=None,
#     subset=None,
#     interpolation="bilinear",
#     follow_links=False,
# )






 #   print("Training Data Generator---------------------------------------------")
 #   print(training_data_generator)
#         #validation_data_generator = procimg.ImageDataGenerator(rescale=1./255)
#         #test_data_generator = procimg.ImageDataGenerator(rescale=1./255)
#
#         training_generator = training_data_generator.flow_from_directory(
#             self.training_data_dir,
#             target_size=(self.imageWidth, self.imageHeight),
#             batch_size=self.batchSize,
#             class_mode="binary")
#
#         #validation_generator = validation_data_generator.flow_from_directory(
#         #    self.validation_data_dir,
#          #   target_size=(self.imageWidth, self.imageHeight),
#         #    batch_size=self.batchSize,
#         #    class_mode="binary")
#
#         #test_generator = test_data_generator.flow_from_directory(
#          #   self.test_data_dir,
#          #   target_size=(self.imageWidth, self.imageHeight),
#          #   batch_size=1,
#          #   class_mode="binary",
#          #   shuffle=False)
#
#         aModel.fit_generator(
#             training_generator,
#             steps_per_epoch=len(training_generator.filenames) // self.batchSize,
#             epochs=self.epochs,
#             validation_data=validation_generator,
#             validation_steps=len(validation_generator.filenames) // self.batchSize,
#             callbacks=[clb.CSVLogger(self.trainglogsFile ,
#                                                    append=False,
#                                                    separator=";")],
#             verbose=1)
#         aModel.save_weights(self.modelFile)
#
#         return training_generator,test_generator
#
#     def probabilities(self,aModel,aTestGenerator):
#         probabilities = aModel.predict_generator(aTestGenerator, self.testSize)
#         for index, probability in enumerate(probabilities):
#             filerand = rand.random()
#             image_path = self.test_data_dir + "/" +aTestGenerator.filenames[index]
#             img = mpimg.imread(image_path)
#             plt.plot(img)
#             if probability > 0.5:
#                 plt.title("%.2f" % (probability[0]*100) + "% normal")
#             else:
#                 plt.title("%.2f" % ((1-probability[0])*100) + "% abnormal")
#
#           # plt.savefig("output/figure" + str(filerand) + ".png")
'''
