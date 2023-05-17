"""Input pipeline"""
import tensorflow as tf
import tensorflow_datasets as tfds


def random_crop(tensor, crop_size):
    """Function to randomly crop the tensors.Crops the tensor along its first dimension randomly given a crop size."""
    # We assume that the tensor's shape in the first dimension is larger than `crop_size`
    start = tf.random.uniform(shape=(), maxval=tf.shape(tensor)[0] - crop_size + 1, dtype=tf.int32)
    return tensor[start: start + crop_size]




