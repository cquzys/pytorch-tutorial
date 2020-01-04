import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils import data
from scipy.io import loadmat


batch_size = 37


def RandomSample(data):
    N = len(data)
    return data[np.random.randint(N)]


def bootstrap(data, labels, samplesPerClass=None):
    Nsamples = samplesPerClass
    classes = np.unique(labels)

    maxSamples = np.max(np.bincount(labels))

    if samplesPerClass is None or samplesPerClass < maxSamples:
        Nsamples = maxSamples

    bootstrapSamples = []
    bootstrapClasses = []

    for i, c in enumerate(classes):
        classLabel = c
        classData = data[labels == c]

        nBootstrap = Nsamples

        for n in range(nBootstrap):
            sample = RandomSample(classData)

            bootstrapSamples.append(sample)
            bootstrapClasses.append(c)

    bootstrapSamples = np.asarray(bootstrapSamples)
    bootstrapClasses = np.asarray(bootstrapClasses)

    bootstrapData = np.hstack((bootstrapSamples, np.array([bootstrapClasses]).T))
    np.random.shuffle(bootstrapData)

    return (bootstrapData[:, :-1], bootstrapData[:, -1])


def load_input_data(path):
    np_matrix = np.load(path)

    return np_matrix


def get_samples(path, upper_k=10, down_k=10):
    np_matrix = load_input_data(path)
    col_shape = np_matrix.shape[1]
    row_shape = np_matrix.shape[0]
    all_samples = []
    for index in range(0, row_shape):
        key_martix = np_matrix[index]
        upper_padding = 0
        down_padding = 0

        if index < upper_k:
            upper_padding = upper_k - index
            upper_zero_matrix = np.zeros((upper_padding, col_shape), dtype=np.float32)

        if index + down_k >= row_shape:
            down_padding = down_k + index - row_shape + 1
            down_zero_matrix = np.zeros((down_padding, col_shape), dtype=np.float32)

        if upper_padding == 0 and down_padding == 0:
            upper_matrix = np_matrix[index-upper_k:index]
            down_matrix = np_matrix[index + 1:index+down_k + 1]
            single_sample = np.vstack((upper_matrix, key_martix, down_matrix))

        elif upper_padding > 0 and down_padding == 0:
            down_matrix = np_matrix[index + 1:index + down_k + 1]
            if upper_k == upper_padding:
                single_sample = np.vstack((upper_zero_matrix, key_martix, down_matrix))
            else:
                upper_matrix = np_matrix[index - upper_k + upper_padding:index]
                single_sample = np.vstack((upper_zero_matrix, upper_matrix, key_martix, down_matrix))

        elif down_padding > 0 and upper_padding == 0:
            upper_matrix = np_matrix[index - upper_k:index]
            if down_k == down_padding:
                single_sample = np.vstack((upper_matrix, key_martix, down_zero_matrix))
            else:
                down_matrix = np_matrix[index + 1:index + down_k - down_padding + 1]
                single_sample = np.vstack((upper_matrix, key_martix, down_matrix, down_zero_matrix))

        else:
            raise Exception("data is smaller \n")

        all_samples.append(single_sample)
    # need convert to numpy
    all_samples = np.concatenate([item[np.newaxis, :] for item in all_samples], axis=0)

    all_samples = all_samples[:, np.newaxis]
    # convert to [sample, w, h, 1]
    # all_samples = all_samples.transpose((0, 2, 3, 1))

    return all_samples


def get_labels(path):
    labels = loadmat(path)
    odd_labels = labels['rows'].flatten()
    even_labels = labels['cols'].flatten()

    return odd_labels, even_labels


def get_train_test(upper_k, down_k, input_path, target_path, data_type, ratio):

    all_samples = get_samples(input_path, upper_k=upper_k, down_k=down_k)
    odd_labels, even_labels = get_labels(target_path)
    # train_index = int(all_samples.shape[0]*ratio)
    train_index = 7000
    if data_type == 'even':
        labels = even_labels
    else:
        labels = odd_labels

    # samples, target = bootstrap(all_samples[:7000], labels[:7000], samplesPerClass=2000)

    train_data = data.TensorDataset(torch.from_numpy(all_samples[:train_index]),
                                    torch.from_numpy(labels[:train_index]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

    test_data = data.TensorDataset(torch.from_numpy(all_samples[train_index:]),
                                   torch.from_numpy(labels[train_index:]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def plot_loss(epochs, x, y):
    plt.plot(x, y)
    # plt.xticks([i * 1 for i in range(0, epochs+3)])
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.title("Train loss in conv model")
    plt.show()
