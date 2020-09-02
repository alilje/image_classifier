
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras.preprocessing.image as procimg
import keras.optimizers as opmz
import keras.layers as lyrs
import keras.engine
import keras.layers.convolutional as conv
import random as rand

#ToDo: Please move the argument in this init to a more reasonable form
class Classifiers:

    def __init__(self, anImageSize, anEpoch, aBatchSize, aTestSize, numBands):
        #TODO: You n3eed to add these to a configuration file.
        # Recommendation - add directory config and use config module.
        self.model = keras.Sequential()
        self.train_datagen = None
        self.test_datagen = None
        self.train_generator = None
        self.validation_generator = None
        self.aTestGenerator = None
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

        #self.model = keras.Sequential()
        self.model.add(conv.Conv2D(32, (3, 3), padding='same', input_shape=self.inputShape, activation='relu'))
        self.model.add(conv.Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(lyrs.pooling.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(conv.Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(conv.Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(lyrs.pooling.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(lyrs.Flatten())
        self.model.add(lyrs.Dense(256, activation='relu'))
        self.model.add(lyrs.Dropout(0.5))
        self.model.add(lyrs.Dense(256, activation='relu'))
        self.model.add(lyrs.Dropout(0.5))
        self.model.add(lyrs.Dense(1))
        self.model.add(lyrs.Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
            optimizer=opmz.RMSprop(lr=0.0001),
            metrics=['accuracy'])

        with open(self.modelSummaryFile, "w") as fh:
            self.model.summary(print_fn=lambda line: fh.write(line + "\n"))

    def trainerGen(self):
        print("Creating Image Data Generator------------------------------------------------")
        self.train_datagen = procimg.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,  # TODO: Note this gives warning on overriding
                                                  #  featurewise_center? Interesting you need to
                                                  #  look at this...
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=(1.0, 20.0),
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
        print(str(self.train_datagen))

        self.test_datagen = procimg.ImageDataGenerator(rescale=1./255)

        print("Test Data Gen-------------------")
        print(str(self.test_datagen))

        self.train_generator = self.train_datagen.flow_from_directory(
            '/ascldap/users/alilje/W80-1K/data/training',
            target_size=(500, 500),
            batch_size=32,
            class_mode='binary')

        self.validation_generator = self.test_datagen.flow_from_directory(
            '/ascldap/users/alilje/W80-1K/data/validation',
            target_size=(500, 500),
            batch_size=32,
            class_mode='binary')

        print("Validation Generator ------------------------------------------")
        print(str(self.validation_generator))

        self.model.fit(
            self.train_generator,
            steps_per_epoch=20,
            epochs=5,
            validation_data=self.validation_generator,
            validation_steps=10)

    def probabilities(self):
        probabilities = self.model.predict_generator(self.test_datagen, self.testSize)
        for index, probability in enumerate(probabilities):
            filerand = rand.random()
            image_path = self.test_data_dir + "/" +self.test_datagen.filenames[index]
            img = mpimg.imread(image_path)
            plt.plot(img)
            if probability > 0.5:
                plt.title("%.2f" % (probability[0]*100) + "% normal")
            else:
                plt.title("%.2f" % ((1-probability[0])*100) + "% abnormal")
                plt.savefig("output/figure" + str(filerand) + ".png")

