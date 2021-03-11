import torch
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np
import scipy.fft as fft
from raw_audio_autoencoder import AutoEncoder_1D


def generate_data(model, nb_samples):
    # generate random input
    input = torch.rand(1, 2, 688)  # shape (1, 2, 688)

    # input goes through the decoder
    output = model.decoder(input)  # shape (1, 2, 11000)

    # remove the 1st dimension -> (2, 11000)
    output = torch.squeeze(output)
    output = output.detach().numpy()
    print(output.shape)

    plt.plot(output[0][:200], color="tab:red")
    plt.plot(output[1][:200], color="tab:blue")
    plt.show()
    return output


def generate_sinus(model):

    latent_space_values = [10, 20, 30, 40, 50, 60]

    fig, ax = plt.subplots(6, 1)
    fig_spec, ax_spec = plt.subplots(6, 1)
    for ii in range(6):
        # generate random input
        input = torch.normal(mean=latent_space_values[ii], std=1, size=(1, 2, 688))  # shape (1, 2, 688)

        # input goes through the decoder
        output = model.decoder(input)  # shape (1, 2, 11000)

        # remove the 1st dimension -> (2, 11000)
        output = torch.squeeze(output)
        output = output.detach().numpy()

        # plot generated sinus
        ax[ii].plot(output[0][:200], color="tab:red")
        ax[ii].set_title(f"Latent space mean {latent_space_values[ii]}")
        ax[ii].set_xticks([])
        ax[ii].set_yticks([])

        # plot spectrum
        yf = make_spectrum(output[0])
        ax_spec[ii].plot(yf, color="tab:red")
        ax_spec[ii].set_title(f"Spectrum {latent_space_values[ii]}")
        ax_spec[ii].set_yticks([])
    plt.show()
    return output


def make_spectrum(y):
    nb_points = y.shape[0]
    yf = fft.fft(y)
    yf = 2.0/nb_points * np.abs(yf[0:nb_points//2])
    return yf


def from_array_to_wav(sound_array, fs, name):
    sound_array_0, sound_array_1 = sound_array[0], sound_array[1]
    wavfile.write('audio_generated/' + name + 'stereo_0.wav', rate=fs, data=sound_array_0)
    wavfile.write('audio_generated/' + name + 'stereo_1.wav', rate=fs, data=sound_array_1)
    pass


if __name__ == '__main__':
    # load trained model and sinus data
    model_name = 'raw_audio_encoder_1D_sinus_200_1000Hz_kernel13_epoch25.pt'
    model = torch.load('../models/'+model_name)
    model.eval()
    output = generate_sinus(model)
