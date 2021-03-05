import librosa
import scipy
import numpy as np
import torch
from torch.utils.data import Dataset




class DataSetAudio_2D (Dataset) :

    def __init__(self, audio_wavfile, duration_sample):
        self.audio = audio_wavfile
        self.duration = duration_sample
        self.samples, self.len_sample, self.nb_samples = self.segment_audio(self.audio, self.duration)

    def __len__(self):
        return self.samples.size()[0]

    def __getitem__(self, item):
        return self.samples[item]

    def segment_audio(self, audio_wavfile, duration_sample):
        # load signal
        signal, init_sr = librosa.core.load(audio_wavfile, sr=None, mono=False)

        # seperate the two stereo band
        signal_stereo_1 = signal[0]
        signal_stereo_2 = signal[1]

        # resample 44 100 Hz -> 11 000 Hz
        len_signal = signal.shape[1]
        sr = 11000
        nb_points_resample = int((sr / init_sr) * len_signal)

        signal_stereo_1 = scipy.signal.resample(signal_stereo_1, nb_points_resample)
        signal_stereo_2 = scipy.signal.resample(signal_stereo_2, nb_points_resample)

        signal = np.array((signal_stereo_1, signal_stereo_2))

        # convert duration_sample in nb points
        len_sample = duration_sample * sr
        nb_samples = int(signal.shape[1] / len_sample)

        # create samples
        samples = np.zeros((nb_samples, 2, len_sample))
        idx_start = 0
        for num_sample in range(nb_samples):
            samples[num_sample][0] = signal_stereo_1[idx_start: idx_start + len_sample]
            samples[num_sample][1] = signal_stereo_1[idx_start: idx_start + len_sample]
            idx_start += len_sample

        # add extra axis (depth) for CNN
        samples = samples[:, np.newaxis, :, :]

        # convert samples to torch.Tensor
        samples_tensor = torch.from_numpy(samples)

        return samples_tensor, len_sample, nb_samples


class DataSetAudio_1D (Dataset) :

    def __init__(self, audio_wavfile, duration_sample):
        self.audio = audio_wavfile
        self.duration = duration_sample
        self.samples, self.len_sample, self.nb_samples = self.segment_audio(self.audio, self.duration)

    def __len__(self):
        return self.samples.size()[0]

    def __getitem__(self, item):
        return self.samples[item]

    def segment_audio(self, audio_wavfile, duration_sample):
        # load signal
        signal, init_sr = librosa.core.load(audio_wavfile, sr=None, mono=False)

        # seperate the two stereo band
        signal_stereo_1 = signal[0]
        signal_stereo_2 = signal[1]

        # resample 44 100 Hz -> 11 000 Hz
        len_signal = signal.shape[1]
        sr = 11000
        nb_points_resample = int((sr / init_sr) * len_signal)

        signal_stereo_1 = scipy.signal.resample(signal_stereo_1, nb_points_resample)
        signal_stereo_2 = scipy.signal.resample(signal_stereo_2, nb_points_resample)

        signal = np.array((signal_stereo_1, signal_stereo_2))

        # convert duration_sample in nb points
        len_sample = duration_sample * sr
        nb_samples = int(signal.shape[1] / len_sample)

        # create samples
        samples = np.zeros((nb_samples, 2, len_sample))
        idx_start = 0
        for num_sample in range(nb_samples):
            samples[num_sample][0] = signal_stereo_1[idx_start: idx_start + len_sample]
            samples[num_sample][1] = signal_stereo_1[idx_start: idx_start + len_sample]
            idx_start += len_sample

        # convert samples to torch.Tensor
        samples_tensor = torch.from_numpy(samples)

        return samples_tensor, len_sample, nb_samples
