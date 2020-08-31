import keras.models as mods
import keras.layers as lyr


model = mods.Sequential()

model.add(mods.Conv2D(32, (3, 3), input_shape=input_shape))
model.add(lyr.Activation("relu"))
model.add(lyr.MaxPooling2D(pool_size=(2, 2)))

model.add(mods.Conv2D(32, (3, 3)))
model.add(lyr.Activation("relu"))
model.add(lyr.MaxPooling2D(pool_size=(2, 2)))

model.add(lyr.Flatten())
model.add(mods.Dense(16))
model.add(lyr.Activation("relu"))
model.add(lyr.Dropout(0.5))
model.add(mods.Dense(1))
model.add(lyr.Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])