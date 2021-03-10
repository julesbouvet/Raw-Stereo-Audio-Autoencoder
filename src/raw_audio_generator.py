import torch
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
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

    fig, ax = plt.subplots(2, 3)
    for ii in range(2):
        for jj in range(3):
            # generate random input
            input = torch.normal(mean=latent_space_values[3*ii+jj], std=1, size=(1, 2, 688))  # shape (1, 2, 688)

            # input goes through the decoder
            output = model.decoder(input)  # shape (1, 2, 11000)

            # remove the 1st dimension -> (2, 11000)
            output = torch.squeeze(output)
            output = output.detach().numpy()

            ax[ii, jj].plot(output[0][:200], color="tab:red")
            ax[ii, jj].set_title(f"Latent space {latent_space_values[3*ii+jj]}")
            ax[ii, jj].set_xticks([])
            ax[ii, jj].set_yticks([])
    plt.show()
    return output


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


