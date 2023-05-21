"""
Input pipeline
In early experiments, I am randomly cropping proteins to 256 amino acids and filtering out those that are below this
length. If my model trains with this, then I can move onto the more complicated regime where the number of residues
is variable (quite difficult to make that work in JAX, so I might switch to PyTorch.)
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from polymer import rg_confined_covariance, compute_b, diffuse
import numpy as np


def crop_proteins(ds, crop_size):
    """Given a dataset and crop size, returns a dataset that randomly crops the proteins."""

    def random_crop(tensor):
        """Function to randomly crop the tensors.Crops the tensor along its first dimension randomly given a crop size.
        (unit-tested)"""
        # We assume that the tensor's shape in the first dimension is larger than `crop_size`
        start = tf.random.uniform(shape=(), maxval=tf.shape(tensor)[0] - crop_size + 1, dtype=tf.int32)
        return tensor[start: start + crop_size]

    def crop_tensors(tensor):
        """Implements cropping given the crop size."""
        tensor_length = tf.shape(tensor)[0]
        # Crop if tensor_length > crop_size, else return tensor
        tensor = tf.cond(tensor_length > crop_size, lambda: random_crop(tensor), lambda: tensor)

        # Slice tensor to only get the backbone atoms
        tensor = tensor[:, :4, :]
        return tensor

    # Crop dataset
    ds = ds.map(crop_tensors)
    return ds


def filter_shorts(ds, crop_size):
    """Given a dataset, filters out the proteins that are shorter than the crop size."""

    def check_valid(t):
        return tf.shape(t)[0] > crop_size

    # Filter out invalid tensors
    ds = ds.filter(check_valid)
    return ds


def remove_enumeration(index, element):
    # TODO: (I don't know why there is so much enumeration). This is likely a
    #  bug during my data processing.
    return element[1][1][1][1]


def create_protein_dataset(crop_size: int,
                           batch_size: int = 128) -> tf.data.Dataset:
    """Given a crop size and a batch size, constructs a high-throughput data pipeline
    for TPU training. (unit-tested)"""
    # Load the dataset from GCS
    gcs_path = "gs://chroma-af/thermo-ds"  # TODO: load this from configs file
    ds = tf.data.Dataset.load(gcs_path)

    # Remove enumeration
    ds = ds.map(remove_enumeration)

    # Filter short proteins
    ds = filter_shorts(ds, crop_size)

    # Repeat dataset forever
    ds = ds.repeat()

    # Crop proteins
    ds = crop_proteins(ds, crop_size)

    # Add R and R^-1 to training data
    a = 2.1939
    R, R_inverse = rg_confined_covariance(n_atoms=crop_size * 4,
                                          a=a,
                                          b=compute_b(crop_size, 4, a))
    ds = ds.map(lambda xyz: (xyz, R, R_inverse))

    # Batch and prefetch
    ds = ds.batch(batch_size, drop_remainder=True)  # drop_remainder for TPU training
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def convert2iterator(ds):
    return iter(tfds.as_numpy(ds))


def create_denoising_datasets(crop_size: int, batch_size: int, scale: float = 2.0):
    ds = create_protein_dataset(crop_size, batch_size)
    ds = ds.unbatch()  # unbatch

    # Noise protein
    def make_denoising_example(xyz, R, R_inverse):
        N_at, *_ = R.shape
        x0 = tf.reshape(xyz, (N_at, 3))
        noise = tf.random.normal(shape=x0.shape)
        noised_xyz = tf.reshape((x0 + noise * scale), shape=(N_at // 4, 4, 3))
        return noised_xyz, xyz

    ds = ds.map(make_denoising_example)

    # Split to train and take
    train_ds = ds.take(10_000)
    test_ds = ds.skip(10_000).take(2_000)

    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    return train_ds, test_ds
