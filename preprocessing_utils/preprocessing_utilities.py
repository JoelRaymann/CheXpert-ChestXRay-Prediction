# For python 2 support
from __future__ import absolute_import, print_function, unicode_literals, division

# import necessary packages
import tensorflow as tf
import pandas as pd
import numpy as np


# NOTE: PRIVATE Functions
# TODO ==> Insert any private functions here


# NOTE: PUBLIC Functions
def clean_data(dataset_path: str, output_save_path: str, nan_value: int):
    
    data = pd.read_csv(dataset_path)
    data = data.fillna(nan_value)
    data = data[data["Frontal/Lateral"] == "Frontal"]
    data = pd.concat([data.iloc[:, 0], data.iloc[:, 5:]], axis = 1)
    
    # set uncertain to 2
    for label in data.columns[1:]:
        data.loc[data[label] == -1, label] = 2
    data.to_csv(output_save_path, index = False)
    return True
    
@tf.function
def min_max_scaling(img, old_range:tuple, new_range:tuple):
    """
    Function to min max scaling for the given image from
    the old range to the new range given as tuple
    
    Arguments:
        img {tf.Tensor} -- The image to rescale
        old_range {tuple} -- The old range of scale values eg. (0., 255.)
        new_range {tuple} -- The new range of scale values to rescale eg. (0., 1.)
    
    Returns:
        tf.Tensor -- The output rescaled image
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    img = tf.add(tf.multiply(tf.divide(tf.subtract(img, old_min), tf.subtract(old_max, old_min)), tf.subtract(new_max, new_min)), new_min)
    return img

@tf.function
def xray_load_image(xray_path: str, output_size: tuple):
    """
    Function to load the XRay image from the path
    given and resize it and normalize it
    
    Arguments:
        xray_path {str} -- The path of the xray
    """
    img = tf.io.read_file(xray_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    
    # Resize the image
    d_w, d_h = output_size
    img = tf.image.resize(img, size = (d_w, d_h), method = tf.image.ResizeMethod.AREA)
    
    # Normalize the image
    img = min_max_scaling(img, (tf.reduce_min(img), tf.reduce_max(img)), (0.0, 1.0))

    return img

@tf.function
def preprocess_xray_data(sample_path, *labels) -> tuple:
    """
    Function to preprocess the xray image from the path given
    using the csv meta data provided
    
    Arguments:
        sample_path {[type]} -- The xray image data path
        labels {tf.float32} -- The labels data
    
    Returns:
        tuple -- The X, y output in which 
        X.shape == [224, 224, 3]
        y.shape == 14 * [1, 3]
    """
    # Read images
    img = xray_load_image("./dataset/" + sample_path, output_size = (224, 224))
    # labels
    y = tuple([tf.one_hot(tf.cast(label, dtype = tf.int32), depth = 3, dtype = tf.float32) for label in labels])

    return img, y

def load_tf_dataset_generator(dataset_meta_path: str, batch_size = 32, shuffle = True, shuffle_buffer = 1000):
    """
    Function to return a tf.data.Dataset for the
    given dataset_metadata prepared from the 
    CheXpert dataset
    
    Arguments:
        dataset_meta_path {str} -- The path to the csv file
    
    Keyword Arguments:
        batch_size {int} -- The batch_size needed (default: {32})
        shuffle {bool} -- The shuffle status flag
        shuffle_buffer {int} -- The shuffle buffer for shuffling (default: {1000})
    
    Returns:
        tf.data.Dataset -- The generator for the dataset
    """

    # Get the dataset generator
    dataset_gen = tf.data.experimental.CsvDataset(dataset_meta_path, [tf.string] + [tf.float32] * 14, header =  True)

    # Map the preprocessor
    dataset_gen = dataset_gen.map(preprocess_xray_data, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    # Shuffle and repeat
    if shuffle:
        dataset_gen = dataset_gen.shuffle(shuffle_buffer)
    
    dataset_gen = dataset_gen.repeat()
    
    # Batch it
    dataset_gen = dataset_gen.batch(batch_size)

    # prefetch enabling
    dataset_gen = dataset_gen.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset_gen
    



