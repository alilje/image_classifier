
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras.preprocessing.image as procimg
import keras.models as mods
import keras.optimizers as opmz
import keras.layers as lyr
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
        self.path = "../datasets/normal_vs_faulty/"
        self.training_data_dir = self.path + "training"
        self.validation_data_dir = self.path + "validation"
        self.test_data_dir = self.path + "test"



        # Hyperparams
        self.imageSize = anImageSize
        self.imageWidth, self.imageHeight = self.imageDimension, self.imageDimension
        self.epochs = anEpoch
        self.batchSize = aBatchSize
        self.testSize = aTestSize
        self.numBands = numBands
        self.inputShape = (self.imageSize, self.imageSize, self.numBands)

    def models(self):

        model = mods.Sequential()

        model.add(mods.Conv2D(32, 3, 3, border_mode='same', input_shape=self.InputShape, activation='relu'))
        model.add(mods.Conv2D(32, 3, 3, border_mode='same', activation='relu'))
        model.add(mods.MaxPooling2D(pool_size=(2, 2)))

        model.add(mods.Conv2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(mods.Conv2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(mods.MaxPooling2D(pool_size=(2, 2)))

        model.add(mods.Conv2D(128, 3, 3, border_mode='same', activation='relu'))
        model.add(mods.Conv2D(128, 3, 3, border_mode='same', activation='relu'))
        model.add(mods.MaxPooling2D(pool_size=(2, 2)))

        model.add(mods.Conv2D(256, 3, 3, border_mode='same', activation='relu'))
        model.add(mods.Conv2D(256, 3, 3, border_mode='same', activation='relu'))
        model.add(mods.MaxPooling2D(pool_size=(2, 2)))

        model.add(lyr.Flatten())
        model.add(lyr.Dense(256, activation='relu'))
        model.add(lyr.Dropout(0.5))

        model.add(mods.Dense(256, activation='relu'))
        model.add(lyr.Dropout(0.5))

        model.add(lyr.Dense(1))
        model.add(lyr.Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                  optimizer=opmz.RMSprop(lr=0.0001),
                  metrics=['accuracy'])

        with open(self.modelSummaryFile,"w") as fh:
            model.summary(print_fn=lambda line: fh.write(line + "\n"))

        return model

    def trainerGen(self,aModel):
        training_data_generator = procimg.ImageDataGenerator(
            rescale=1./255,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            height_shift_range=0.2,
            width_shift_range=0.2)
        validation_data_generator = procimg.ImageDataGenerator(rescale=1./255)
        test_data_generator = procimg.ImageDataGenerator(rescale=1./255)

        training_generator = training_data_generator.flow_from_directory(
            self.training_data_dir,
            target_size=(self.imageWidth, self.imageHeight),
            batch_size=self.batchSize,
            class_mode="binary")

        validation_generator = validation_data_generator.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.imageWidth, self.imageHeight),
            batch_size=self.batchSize,
            class_mode="binary")

        test_generator = test_data_generator.flow_from_directory(
            self.test_data_dir,
            target_size=(self.imageWidth, self.imageHeight),
            batch_size=1,
            class_mode="binary",
            shuffle=False)

        aModel.fit_generator(
            training_generator,
            steps_per_epoch=len(training_generator.filenames) // self.batchSize,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator.filenames) // self.batchSize,
            callbacks=[clb.PlotLossesCallback(), clb.CSVLogger(self.trainglogsFile ,
                                                   append=False,
                                                   separator=";")],
            verbose=1)
        aModel.save_weights(self.modelFile)

        return training_generator,test_generator

    def probabilities(self,aModel,aTestGenerator):
        probabilities = aModel.predict_generator(aTestGenerator, self.testSize)
        for index, probability in enumerate(probabilities):
            filerand = rand.random()
            image_path = self.test_data_dir + "/" +aTestGenerator.filenames[index]
            img = mpimg.imread(image_path)
            plt.plot(img)
            if probability > 0.5:
                plt.title("%.2f" % (probability[0]*100) + "% normal")
            else:
                plt.title("%.2f" % ((1-probability[0])*100) + "% abnormal")

            plt.savefig("output/figure" + str(filerand) + ".png")