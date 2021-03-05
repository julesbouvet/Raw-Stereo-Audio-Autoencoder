import torch
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile


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


def from_array_to_wav(sound_array, fs, name):
    sound_array_0, sound_array_1 = sound_array[0], sound_array[1]
    wavfile.write('audio_generated/' + name + 'stereo_0.wav', rate=fs, data=sound_array_0)
    wavfile.write('audio_generated/' + name + 'stereo_1.wav', rate=fs, data=sound_array_1)
    pass


if __name__ == '__main__':
    # load trained model and sinus data
    model = torch.load('saved_models/raw_audio_encoder_1D')
    model.eval()
    output = generate_data(model)
    from_array_to_wav(sound_array=output, fs= 11000, name='test1')


