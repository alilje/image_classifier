import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras.preprocessing.image as procimg #ImageDataGenerator
import keras.models as mods
import keras.optimizers as opmz
import keras.layers as lyr
import keras.callbacks as clb
import livelossplot.keras.PlotLossesCallback as pltcall
import efficientnet.keras as efn
import random as rand

#ToDo: All of these need to be in a configuration file ASAP
TRAINING_LOGS_FILE = "training_logs.csv"
MODEL_SUMMARY_FILE = "model_summary.txt"
MODEL_FILE = "normal-vs-faulty.h5"

# Data
path = "../datasets/normal_vs_faulty"
training_data_dir = path + "training"
validation_data_dir = path + "validation"
test_data_dir = path + "test" # 12 500

#%%

# Hyperparams
IMAGE_SIZE = 200
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 30

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

#%%

# CNN EfficientNet (https://arxiv.org/abs/1905.11946)

model = mods.Sequential()
efficient_net = efn.EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
#efficient_net.trainable = False
for index, layer in enumerate(efficient_net.layers):
    if index < 761:
        layer.trainable = False

    print(index)
    print(layer)
model.add(efficient_net)
#model.add(GlobalMaxPooling2D())
model.add(lyr.Dense(1024, activation='relu'))
model.add(lyr.Flatten())
# if dropout_rate > 0:
#     model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(lyr.Dense(1, activation='sigmoid')) #, name="output"
model.compile(loss='binary_crossentropy',
              optimizer=opmz.RMSprop(lr=0.0001),
              metrics=['accuracy'])

with open(MODEL_SUMMARY_FILE,"w") as fh:
    model.summary(print_fn=lambda line: fh.write(line + "\n"))

#%%

# Data augmentation
training_data_generator = procimg.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)
validation_data_generator = procimg.ImageDataGenerator(rescale=1./255)
test_data_generator = procimg.ImageDataGenerator(rescale=1./255)

#%%

# Data preparation
training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary")
validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary")
test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=1,
    class_mode="binary",
    shuffle=False)

#%%

# Training
model.fit_generator(
    training_generator,
    steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames) // BATCH_SIZE,
    callbacks=[clb.PlotLossesCallback(), clb.CSVLogger(TRAINING_LOGS_FILE,
                                               append=False,
                                               separator=";")],
    verbose=1)
model.save_weights(MODEL_FILE)


# Testing
probabilities = model.predict_generator(test_generator, TEST_SIZE)
for index, probability in enumerate(probabilities):
    image_path = test_data_dir + "/" +test_generator.filenames[index]
    img = mpimg.imread(image_path)
    plt.plot(img)
    if probability > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% normal")
    else:
        plt.title("%.2f" % ((1-probability[0])*100) + "% abnormal")
    plt.show()
