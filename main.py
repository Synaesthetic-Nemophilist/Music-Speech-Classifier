# First check the Python version
import sys

# Now get necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Import Tensorflow
import tensorflow as tf


from libs import utils, dataset_utils, dft


# First download the GTZAN music and speech dataset
dst = 'gtzan_music_speech'
if not os.path.exists(dst):
    dataset_utils.gtzan_music_speech_download(dst)

# Get the full path to the directory
music_dir = os.path.join(os.path.join(dst, 'music_speech'), 'music_wav')

# Now use list comprehension to combine the path of the directory with any wave files
music = [os.path.join(music_dir, file_i)
         for file_i in os.listdir(music_dir)
         if file_i.endswith('.wav')]

# Similarly, for the speech folder:
speech_dir = os.path.join(os.path.join(dst, 'music_speech'), 'speech_wav')
speech = [os.path.join(speech_dir, file_i)
          for file_i in os.listdir(speech_dir)
          if file_i.endswith('.wav')]

# Plot the first .wav music file
file_i = music[0]
s = utils.load_audio(file_i)
plt.plot(s)
plt.show()

# Parameters for DFT
fft_size = 512
hop_size = 256

# This will return our signal as real and imaginary
# components, a polar complex value representation which we will
# convert to a cartesian representation capable of
# saying what magnitudes and phases are in our signal
re, im = dft.dft_np(s, hop_size=256, fft_size=512)
mag, phs = dft.ztoc(re, im)
plt.imshow(mag)


# Calc logarithm of the magnitudes converting it to a psuedo-decibel scale
plt.figure(figsize=(10, 4))
plt.imshow(np.log(mag.T))
plt.xlabel('Time')
plt.ylabel('Frequency Bin')
plt.show()


# Create a sliding window for the data
# The sample rate from our audio is 22050 Hz.
sr = 22050

# We can calculate how many hops there are in a second
# which will tell us how many frames of magnitudes
# we have per second
n_frames_per_second = sr // hop_size

# We want 500 milliseconds of audio in our window
n_frames = n_frames_per_second // 2

# And we'll move our window by 250 ms at a time
frame_hops = n_frames_per_second // 4

# We'll therefore have this many sliding windows:
n_hops = (len(mag) - n_frames) // frame_hops

# Collect all sliding windows into Xs
# label them as 0 or 1 (music/speech) into ys
Xs = []
ys = []
for hop_i in range(n_hops):
    # Creating our sliding window
    frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]

    # Store them with a new 3rd axis and as a logarithmic scale
    # We'll ensure that we aren't taking a log of 0 just by adding
    # a small value, also known as epsilon.
    Xs.append(np.log(np.abs(frames[..., np.newaxis]) + 1e-10))

    # And then store the label
    ys.append(0)


# Store every magnitude frame and its label of being music: 0 or speech: 1
Xs, ys = [], []

# Let's start with the music files
for i in music:
    # Load the ith file:
    s = utils.load_audio(i)

    # Now take the dft of it
    re, im = dft.dft_np(s, fft_size=fft_size, hop_size=hop_size)

    # And convert the complex representation to magnitudes/phases
    mag, phs = dft.ztoc(re, im)

    # This is how many sliding windows we have:
    n_hops = (len(mag) - n_frames) // frame_hops

    # Let's extract them all:
    for hop_i in range(n_hops):
        # Get the current sliding window
        frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]

        # We'll take the log magnitudes, as this is a nicer representation:
        this_X = np.log(np.abs(frames[..., np.newaxis]) + 1e-10)

        # And store it:
        Xs.append(this_X)

        # And be sure that we store the correct label of this observation:
        ys.append(0)

# Now do the same thing with speech
for i in speech:

    # Load the ith file:
    s = utils.load_audio(i)

    # Now take the dft of it
    re, im = dft.dft_np(s, fft_size=fft_size, hop_size=hop_size)

    # And convert the complex representation to magnitudes/phases
    mag, phs = dft.ztoc(re, im)

    # This is how many sliding windows we have:
    n_hops = (len(mag) - n_frames) // frame_hops

    # Let's extract them all:
    for hop_i in range(n_hops):
        # Get the current sliding window
        frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]

        # We'll take the log magnitudes, as this is a nicer representation:
        this_X = np.log(np.abs(frames[..., np.newaxis]) + 1e-10)

        # And store it:
        Xs.append(this_X)

        # Make sure we use the right label
        ys.append(1)

# Convert them to an array:
Xs = np.array(Xs)
ys = np.array(ys)

print(Xs.shape, ys.shape)

assert (Xs.shape == (15360, 43, 256, 1) and ys.shape == (15360,))


# Describe the shape of our input to the network
n_observations, n_height, n_width, n_channels = Xs.shape

# This will accept the Xs, ys, a list defining our dataset split into training,
#  validation, and testing proportions, and a parameter one_hot stating whether
#  we want our ys to be converted to a one hot vector or not
ds = dataset_utils.Dataset(Xs=Xs, ys=ys, split=[0.8, 0.1, 0.1], one_hot=True)

# train split gives us a batch generator of batch_size=100
Xs_i, ys_i = next(ds.train.next_batch())

# Notice the shape this returns.  This will become the shape of our input and output of the network:
print(Xs_i.shape, ys_i.shape)

assert(ys_i.shape == (100, 2))



# Creation of NN
tf.reset_default_graph()

# Create the input to the network.  This is a 4-dimensional tensor!
# Don't forget that we should use None as a shape for the first dimension
# Recall that we are using sliding windows of our magnitudes
X = tf.placeholder(name='X', shape=[None, 43, 256, 1], dtype=tf.float32)

# Create the output to the network.  This is our one hot encoding of 2 possible values
Y = tf.placeholder(name='Y', shape=[None, 2], dtype=tf.float32)

# TODO:  Explore different numbers of layers, and sizes of the network
n_filters = [16, 16, 16, 16]

# Now let's loop over our n_filters and create the deep convolutional neural network
H = X
for layer_i, n_filters_i in enumerate(n_filters):
    # Let's use the helper function to create our connection to the next layer:
    # TODO: explore changing the parameters here:
    H, W = utils.conv2d(
        H, n_filters_i, k_h=16, k_w=16, d_h=2, d_w=2,
        name=str(layer_i))

    # And use a nonlinearity
    # TODO: explore changing the activation here:
    H = tf.nn.relu(H)

    # Just to check what's happening:
    print(H.get_shape().as_list())

# Connect the last convolutional layer to a fully connected network
fc, W = utils.linear(H, 100, name="fc1", activation=tf.nn.relu)

# And another fully connected layer, now with just 2 outputs, the number of outputs that our
# one hot encoding has
Y_pred, W = utils.linear(fc, 2, name="fc2", activation=tf.nn.relu)


#Cost calculation using binary cross entropy measure
loss = utils.binary_cross_entropy(Y_pred, Y)
cost = tf.reduce_mean(tf.reduce_sum(loss, 1))

# Create a measure of accuracy by finding the prediction of our network
predicted_y = tf.argmax(Y_pred, 1)
actual_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(predicted_y, actual_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Create an optimizer
learning_rate = 0.00001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# ...And train the network!
# Explore these parameters: (TODO)
n_epochs = 10
batch_size = 200

# Create a session and init!
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Now iterate over our dataset n_epoch times
for epoch_i in range(n_epochs):
    print('Epoch: ', epoch_i)

    # Train
    this_accuracy = 0
    its = 0

    # Do our mini batches:
    for Xs_i, ys_i in ds.train.next_batch(batch_size):
        # Note here: we are running the optimizer so
        # that the network parameters train!
        this_accuracy += sess.run([accuracy, optimizer], feed_dict={
            X: Xs_i, Y: ys_i})[0]
        its += 1
        print(this_accuracy / its)
    print('Training accuracy: ', this_accuracy / its)

    # Validation (see how the network does on unseen data).
    this_accuracy = 0
    its = 0

    # Do our mini batches:
    for Xs_i, ys_i in ds.valid.next_batch(batch_size):
        # Note here: we are NOT running the optimizer!
        # we only measure the accuracy!
        this_accuracy += sess.run(accuracy, feed_dict={
            X: Xs_i, Y: ys_i})
        its += 1
    print('Validation accuracy: ', this_accuracy / its)