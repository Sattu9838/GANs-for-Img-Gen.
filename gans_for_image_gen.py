import requests
import zipfile # This is a built-in module, no need to install
import os
import glob
from PIL import Image
import numpy as np
import tensorflow as tf
#rest of the code 
from tensorflow.keras.layers import Dense, Reshape, LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to download and extract CelebA dataset
def download_celeba_dataset(url, save_path):
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Download the dataset zip file
    r = requests.get(url, stream=True)
    with open(os.path.join(save_path, 'celeba_dataset.zip'), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    # Extract the dataset zip file
    with zipfile.ZipFile(os.path.join(save_path, 'celeba_dataset.zip'), 'r') as zip_ref:
        zip_ref.extractall(save_path)

# **Corrected URL to download CelebA dataset (Img_align_celeba.zip)**
celeba_url = 'https://www.dropbox.com/s/d1kjpkqklf0uw77/img_align_celeba.zip?dl=1'

# Path to save the dataset
dataset_save_path = './celeba_dataset/'  # Adjust as per your preference

# Download and extract CelebA dataset
download_celeba_dataset(celeba_url, dataset_save_path)

# Function to load and preprocess images
def load_and_preprocess_images(img_dir, img_size=(64, 64)):
    images = []
    for img_path in glob.glob(img_dir + '/*.jpg'):
        img = Image.open(img_path).resize(img_size)
        img = np.array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
        images.append(img)
    return np.array(images)

# Load and preprocess CelebA dataset
celeba_dataset_path = os.path.join(dataset_save_path, 'img_align_celeba/')
data = load_and_preprocess_images(celeba_dataset_path)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Generator
def build_generator():
    model = Sequential()
    model.add(Dense(256 * 8 * 8, activation="relu", input_dim=100))
    model.add(Reshape((8, 8, 256)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'))
    return model

# Discriminator
def build_discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Example usage:
generator = build_generator()
discriminator = build_discriminator((64, 64, 3))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Function to build GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(100,))
    img = generator(gan_input)
    gan_output = discriminator(img)
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan

# Example usage:
gan = build_gan(generator, discriminator)

# Function to train GAN model
def train_gan(gan, generator, discriminator, data, epochs, batch_size=128, save_interval=50):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Check if data is empty and handle it appropriately
        if data.size == 0:
            print("Error: Data array is empty. Please load the dataset before training.")
            return  # Exit the training function

        # Train Discriminator
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_imgs = data[idx]
        # ... rest of your code ...

import numpy as np
import matplotlib.pyplot as plt


# Function to generate new images
def generate_images(generator, num_images=25, img_shape=(64, 64, 3)):
    noise = np.random.normal(0, 1, (25, 100))

    # noise = np.random.normal(0, 1, (num_images, 100))  # Generate random noise as input
    gen_imgs = generator.predict(noise)               # Generate images from noise
    gen_imgs = 0.5 * gen_imgs + 0.5                    # Rescale images to [0, 1] range

    # Plot generated images
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    count = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[count, :, :, :])
            axs[i, j].axis('off')
            count += 1
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming `generator` is your trained generator model
generate_images(generator, num_images=25)
