import keras
from keras import layers
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from matplotlib import pyplot as plt

def define_autoencoder():
    input_img = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def get_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    return x_train, x_test

def save_autoencoder(autoencoder, epoch):
    model_json = autoencoder.to_json()
    with open("models/model_tex_" + str(epoch) + ".json", "w") as json_file:
        json_file.write(model_json)

    autoencoder.save_weights("models/model_tex_" + str(epoch) + ".h5")
    print("model saved.")

def load_autoencoder(epoch):
    json_path = "models/model_tex_"+ str(epoch) +".json"
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder = keras.models.model_from_json(loaded_model_json)
    print("json-file loaded")
    # load weights into new model
    autoencoder.load_weights("models/model_tex_"+ str(epoch) +".h5")
    print("model loaded from disk")
    return autoencoder

def write_decoded_images(decoded_images, image_count):
    image_count = 10
    for i in range(image_count):
        fig = plt.figure(frameon=False)  # figure without frame
        ax = plt.Axes(fig, [0., 0., 1., 1.])  # make the image fill out the entire figure
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(decoded_images[i], cmap=plt.cm.bone, interpolation='nearest', aspect='auto')  # decoded_images[i] to be reshaped to (256, 256) ?
        fig.savefig("decoded/image_"+'{0:03d}'.format(i))

def main():
    train = False
    predict = False
    epoch = 1  # to be modified for actual results
    x_train, x_test = get_data()
    if train:
        autoencoder = define_autoencoder()
        history = autoencoder.fit(x_train, x_train,
                                    epochs=epoch,
                                    batch_size=128,
                                    shuffle=True,
                                    validation_data=(x_test, x_test),
                                    callbacks=[TensorBoard(log_dir="/tmp/autoencoder")])
        save_autoencoder(autoencoder, epoch)
        print("plotting loss")
        plt.plot(history.history["loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()
    else: # already trained
        autoencoder = load_autoencoder(epoch)
        print("checking loaded Model...")
        autoencoder.compile(optimizer="adam", loss="mse")
        evaluation = autoencoder.evaluate(x_test, x_test)
    if predict:
        decoded_images = autoencoder.predict(x_test)
        image_count = 10
        write_decoded_images(decoded_images, 10)

if __name__ == "__main__":
    main()
