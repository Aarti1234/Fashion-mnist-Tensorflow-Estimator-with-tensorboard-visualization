import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = './experiments/'
NAME_TO_VISUALISE_VARIABLE = "fmnistembedding"
TO_EMBED_COUNT = 500


path_for_mnist_sprites =  os.path.join(LOG_DIR,'fmnist__sprite.png')
path_for_mnist_metadata =  os.path.join(LOG_DIR,'metadata.tsv')

SOURCE_URL="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"


fmnist = input_data.read_data_sets("FMNIST_data/", one_hot=False, source_url=SOURCE_URL)
batch_xs, batch_ys = fmnist.train.next_batch(TO_EMBED_COUNT)

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage

def vector_to_matrix_mnist(fmnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(fmnist_digits,(-1,28,28))

def invert_grayscale(fmnist_digits):
    """ Makes black white, and white black """
    return 1-fmnist_digits

to_visualise = batch_xs
to_visualise = vector_to_matrix_mnist(to_visualise)
to_visualise = invert_grayscale(to_visualise)

sprite_image = create_sprite_image(to_visualise)

plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray')
plt.imshow(sprite_image,cmap='gray')