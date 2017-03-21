import numpy as np
from .utils import download_and_extract_tar



def gtzan_music_speech_download(dst='gtzan_music_speech'):
    """Download the GTZAN music and speech dataset.

    Parameters
    ----------
    dst : str, optional
        Location to put the GTZAN music and speech datset.
    """
    path = 'http://opihi.cs.uvic.ca/sound/music_speech.tar.gz'
    download_and_extract_tar(path, dst)


def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors.

    Parameters
    ----------
    labels : array
        Input labels to convert to one-hot representation.
    n_classes : int, optional
        Number of possible one-hot.

    Returns
    -------
    one_hot : array
        One hot representation of input.
    """
    return np.eye(n_classes).astype(np.float32)[labels]




class DatasetSplit(object):
    """Utility class for batching data and handling multiple splits.

    Attributes
    ----------
    current_batch_idx : int
        Description
    images : np.ndarray
        Xs of the dataset.  Not necessarily images.
    labels : np.ndarray
        ys of the dataset.
    n_labels : int
        Number of possible labels
    num_examples : int
        Number of total observations
    """

    def __init__(self, images, labels):
        """Initialize a DatasetSplit object.

        Parameters
        ----------
        images : np.ndarray
            Xs/inputs
        labels : np.ndarray
            ys/outputs
        """
        self.images = np.array(images).astype(np.float32)
        if labels is not None:
            self.labels = np.array(labels).astype(np.int32)
            self.n_labels = len(np.unique(labels))
        else:
            self.labels = None
        self.num_examples = len(self.images)

    def next_batch(self, batch_size=100):
        """Batch generator with randomization.

        Parameters
        ----------
        batch_size : int, optional
            Size of each minibatch.

        Returns
        -------
        Xs, ys : np.ndarray, np.ndarray
            Next batch of inputs and labels (if no labels, then None).
        """
        # Shuffle each epoch
        current_permutation = np.random.permutation(range(len(self.images)))
        epoch_images = self.images[current_permutation, ...]
        if self.labels is not None:
            epoch_labels = self.labels[current_permutation, ...]

        # Then iterate over the epoch
        self.current_batch_idx = 0
        while self.current_batch_idx < len(self.images):
            end_idx = min(
                self.current_batch_idx + batch_size, len(self.images))
            this_batch = {
                'images': epoch_images[self.current_batch_idx:end_idx],
                'labels': epoch_labels[self.current_batch_idx:end_idx]
                if self.labels is not None else None
            }
            self.current_batch_idx += batch_size
            yield this_batch['images'], this_batch['labels']





class Dataset(object):
    """Create a dataset from data and their labels.

    Allows easy use of train/valid/test splits; Batch generator.

    Attributes
    ----------
    all_idxs : list
        All indexes across all splits.
    all_inputs : list
        All inputs across all splits.
    all_labels : list
        All labels across all splits.
    n_labels : int
        Number of labels.
    split : list
        Percentage split of train, valid, test sets.
    test_idxs : list
        Indexes of the test split.
    train_idxs : list
        Indexes of the train split.
    valid_idxs : list
        Indexes of the valid split.
    """

    def __init__(self, Xs, ys=None, split=[1.0, 0.0, 0.0], one_hot=False):
        """Initialize a Dataset object.

        Parameters
        ----------
        Xs : np.ndarray
            Images/inputs to a network
        ys : np.ndarray
            Labels/outputs to a network
        split : list, optional
            Percentage of train, valid, and test sets.
        one_hot : bool, optional
            Whether or not to use one-hot encoding of labels (ys).
        """
        self.all_idxs = []
        self.all_labels = []
        self.all_inputs = []
        self.train_idxs = []
        self.valid_idxs = []
        self.test_idxs = []
        self.n_labels = 0
        self.split = split

        # Now mix all the labels that are currently stored as blocks
        self.all_inputs = Xs
        n_idxs = len(self.all_inputs)
        idxs = range(n_idxs)
        rand_idxs = np.random.permutation(idxs)
        self.all_inputs = self.all_inputs[rand_idxs, ...]
        if ys is not None:
            self.all_labels = ys if not one_hot else dense_to_one_hot(ys)
            self.all_labels = self.all_labels[rand_idxs, ...]
        else:
            self.all_labels = None

        # Get splits
        self.train_idxs = idxs[:round(split[0] * n_idxs)]
        self.valid_idxs = idxs[len(self.train_idxs):
                               len(self.train_idxs) + round(split[1] * n_idxs)]
        self.test_idxs = idxs[
            (len(self.valid_idxs) + len(self.train_idxs)):
            (len(self.valid_idxs) + len(self.train_idxs)) +
             round(split[2] * n_idxs)]

    @property
    def X(self):
        """Inputs/Xs/Images.

        Returns
        -------
        all_inputs : np.ndarray
            Original Inputs/Xs.
        """
        return self.all_inputs

    @property
    def Y(self):
        """Outputs/ys/Labels.

        Returns
        -------
        all_labels : np.ndarray
            Original Outputs/ys.
        """
        return self.all_labels

    @property
    def train(self):
        """Train split.

        Returns
        -------
        split : DatasetSplit
            Split of the train dataset.
        """
        if len(self.train_idxs):
            inputs = self.all_inputs[self.train_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.train_idxs, ...]
            else:
                labels = None
        else:
            inputs, labels = [], []
        return DatasetSplit(inputs, labels)

    @property
    def valid(self):
        """Validation split.

        Returns
        -------
        split : DatasetSplit
            Split of the validation dataset.
        """
        if len(self.valid_idxs):
            inputs = self.all_inputs[self.valid_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.valid_idxs, ...]
            else:
                labels = None
        else:
            inputs, labels = [], []
        return DatasetSplit(inputs, labels)

    @property
    def test(self):
        """Test split.

        Returns
        -------
        split : DatasetSplit
            Split of the test dataset.
        """
        if len(self.test_idxs):
            inputs = self.all_inputs[self.test_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.test_idxs, ...]
            else:
                labels = None
        else:
            inputs, labels = [], []
        return DatasetSplit(inputs, labels)

    def mean(self):
        """Mean of the inputs/Xs.

        Returns
        -------
        mean : np.ndarray
            Calculates mean across 0th (batch) dimension.
        """
        return np.mean(self.all_inputs, axis=0)

    def std(self):
        """Standard deviation of the inputs/Xs.

        Returns
        -------
        std : np.ndarray
            Calculates std across 0th (batch) dimension.
        """
        return np.std(self.all_inputs, axis=0)

