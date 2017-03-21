import os
import urllib
import numpy as np
from scipy.io import wavfile
import tensorflow as tf

def download(path):
    """Use urllib to download a file.

    Parameters
    ----------
    path : str
        Url to download

    Returns
    -------
    path : str
        Location of downloaded file.
    """

    #from six.moves import urllib

    fname = path.split('/')[-1]
    if os.path.exists(fname):
        return fname

    print('Downloading ' + path)

    def progress(count, block_size, total_size):
        if count % 20 == 0:
            print('Downloaded %02.02f/%02.02f MB' % (
                count * block_size / 1024.0 / 1024.0,
                total_size / 1024.0 / 1024.0), end='\r')

    filepath, _ = urllib.request.urlretrieve(
        path, filename=fname, reporthook=progress)
    return filepath


def download_and_extract_tar(path, dst):
    """Download and extract a tar file.

    Parameters
    ----------
    path : str
        Url to tar file to download.
    dst : str
        Location to save tar file contents.
    """
    import tarfile
    filepath = download(path)
    if not os.path.exists(dst):
        os.makedirs(dst)
        tarfile.open(filepath, 'r:gz').extractall(dst)


def load_audio(filename, b_normalize=True):
    """Load the audiofile at the provided filename using scipy.io.wavfile.

    Optionally normalizes the audio to the maximum value.

    Parameters
    ----------
    filename : str
        File to load.
    b_normalize : bool, optional
        Normalize to the maximum value.
    """
    sr, s = wavfile.read(filename)
    if b_normalize:
        s = s.astype(np.float32)
        s = (s / np.max(np.abs(s)))
        s -= np.mean(s)
    return s



def conv2d(x, n_output,
           k_h=5, k_w=5, d_h=2, d_w=2,
           padding='SAME', name='conv2d', reuse=None):
    """Helper for creating a 2d convolution operation.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to convolve.
    n_output : int
        Number of filters.
    k_h : int, optional
        Kernel height
    k_w : int, optional
        Kernel width
    d_h : int, optional
        Height stride
    d_w : int, optional
        Width stride
    padding : str, optional
        Padding type: "SAME" or "VALID"
    name : str, optional
        Variable scope

    Returns
    -------
    op : tf.Tensor
        Output of convolution
    """
    with tf.variable_scope(name or 'conv2d', reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[k_h, k_w, x.get_shape()[-1], n_output],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(
            name='conv',
            input=x,
            filter=W,
            strides=[1, d_h, d_w, 1],
            padding=padding)

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=conv,
            bias=b)

    return h, W


def linear(x, n_output, name=None, activation=None, reuse=None):
    """Fully connected layer.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply

    Returns
    -------
    h, W : tf.Tensor, tf.Tensor
        Output of fully connected layer and the weight matrix
    """
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        return h, W


def flatten(x, name=None, reuse=None):
    """Flatten Tensor to 2-dimensions.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to flatten.
    name : None, optional
        Variable scope for flatten operations

    Returns
    -------
    flattened : tf.Tensor
        Flattened tensor.
    """
    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(
                x,
                shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2 or 4.  Found:',
                             len(dims))

        return flattened



def binary_cross_entropy(z, x):
    """Binary Cross Entropy measures cross entropy of a binary variable.

    loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Parameters
    ----------
    z : tf.Tensor
        A `Tensor` of the same type and shape as `x`.
    x : tf.Tensor
        A `Tensor` of type `float32` or `float64`.
    """
    eps = 1e-12
    return (-(x * tf.log(z + eps) +
              (1. - x) * tf.log(1. - z + eps)))