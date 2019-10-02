"""Create the input data pipeline using """

import tensorflow as tf
import model.fmnist_dataset as fmnist_dataset

INPUT_FEATURE = 'image'
def train_input_fn(data_dir, params):
    """Train input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = fmnist_dataset.train(data_dir)
    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def test_input_fn(data_dir, params):
    
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = fmnist_dataset.test(data_dir)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset



def serving_input_receiver_fn():


  image = tf.placeholder(tf.float32, [None, 28, 28])
  input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({'image': image,})

  return input_fn
