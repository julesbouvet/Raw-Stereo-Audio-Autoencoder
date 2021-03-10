import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

########################################
#                                      #
#         My Sinus Generator           #
#                                      #
########################################


class SinusSamplesDataset(Dataset):

    def __init__(self, nb_examples, fs, frequency_range, len_sinus):
        self.nb_examples = nb_examples
        self.fs = fs
        self.frequency_range = frequency_range
        self.len_sinus = len_sinus
        self.samples, self.frequencies = self.sinus_generator(self.nb_examples,
                                                              self.fs,
                                                              self.frequency_range,
                                                              self.len_sinus)

    def __len__(self):
        return self.samples.size()[0]

    def __getitem__(self, item):
        return self.samples[item]

    def sinus_generator(self, nb_examples, fs, frequency_range, len_sinus):
        # create samples array
        samples = np.zeros((nb_examples, 2, len_sinus))
        frequencies = np.zeros(nb_examples)

        for num_example in range(nb_examples):
            # define f, gain, phase, celerity of sound and distance btw two mics
            f = np.random.randint(frequency_range[0], frequency_range[1])  # frequency (Hz)
            frequencies[num_example] = f
            gain = np.random.uniform(0.5, 1.0)  # gain
            phase = np.random.random() * np.pi  # phase
            d1 = 2  # distance btw the two microphones (2)
            c = 342  # sound celerity
            H = 1  # height of microphones (m)
            d2 = 2 * np.sqrt(d1 ** 2 + H ** 2)  # distance for echoes

            # define the two sinus
            sinus_1 = gain * np.sin((2 * np.pi * np.arange(len_sinus) * f / fs) + phase)
            sinus_2 = (gain / d1) * np.sin(
                2 * np.pi * f * (np.arange(len_sinus) / fs - d1 / c) + phase) + gain / d2 * np.sin(
                2 * np.pi * f * (np.arange(len_sinus) / fs - d2 / c) + phase)

            # plot
            # plt.plot(np.arange(len_sinus)[:80], sinus_2[:80], color='red')
            # plt.plot(np.arange(len_sinus)[:80], sinus_1[:80], color='blue')
            # plt.show()

            # add to sample
            samples[num_example][0] = sinus_1
            samples[num_example][1] = sinus_2

            # numpy to Tensor
            samples_tensor = torch.from_numpy(samples)

        return samples_tensor, frequencies

#####################################################
#                                                   #
#                 Sinus Generator                   #
#   https://github.com/iCorv/raw_audio_autoencoder  #
#                                                   #
#####################################################

AUDIO_CHUNK_SIZE = 1024
RATE = 44100


def sinus_dataset_generator(num_examples, fs, samples, frequency_range):
    """Builds a dataset of sinus.

    Args:
        num_examples: number of examples to generate (int)
        fs: sample rate of the sinus
        samples: number of samples to generate (int)
        frequency_range: a list of two values defining [lower, upper] frequency range (int)
    Returns:
        A numpy array of sinus examples.
    """
    # first example
    sinus_data = (np.sin((2 * np.pi * np.arange(samples) * 440.0 / fs) + 0.0)).astype(np.float32)
    sinus_data = np.reshape(sinus_data, newshape=(1, 1, samples, 1))

    for idx in range(0, num_examples - 1):
        # random frequency
        f = np.random.randint(frequency_range[0], frequency_range[1])
        # random phase shift
        phase = np.random.random() * np.pi
        # random gain
        gain = np.random.uniform(0.5, 1.0)

        sinus = (np.sin((2 * np.pi * np.arange(samples) * f / fs) + phase) * gain).astype(np.float32)
        # add some noise, mu = 0, sigma = 0.1
        s = np.random.normal(0, 0.1, samples)
        sinus = sinus + s
        # bring it into shape for the model
        sinus = np.reshape(sinus, newshape=(1, 1, samples, 1))
        sinus_data = np.append(sinus_data, sinus, axis=0)

        # scipy signal filt filt
    return sinus_data

def make_sinus_dataset():
    num_examples = 8000
    sinus_data = sinus_dataset_generator(num_examples, RATE, AUDIO_CHUNK_SIZE, [30, 8000])
    print(sinus_data, sinus_data.shape)
    np.savez('samples/sinus_samples', x=sinus_data)


    # see if its actually a sinus...
    fig, ax = plt.subplots(3, 3)
    for ii in range(3):
        for jj in range(3):
            ax[ii, jj].plot(np.squeeze(sinus_data[np.random.randint(0, num_examples - 1)])[:200], color="tab:blue")
            ax[ii, jj].set_xticks([])
            ax[ii, jj].set_yticks([])
    plt.show()

