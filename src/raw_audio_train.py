import numpy as np
import numpy.matlib
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from raw_audio_autoencoder import AutoEncoder_1D
from raw_audio_dataset import DataSetAudio_1D
from sinus_dataset_generator import SinusSamplesDataset
import matplotlib.pyplot as plt


# DATA GENERATION
def audio_data_generation(audio_wav, duration_sample, batch_size):

    # dataset loading
    dataset = DataSetAudio_1D(audio_wavfile=audio_wav, duration_sample= duration_sample)
    samples = dataset.samples

    # train test split
    # split = int(load_data.shape[0] * 0.7)
    # print('split', split)

    # dataloader creation
    train_dataloader = DataLoader(dataset, batch_size)
    test_dataloader = DataLoader(dataset, batch_size)

    return train_dataloader, test_dataloader, samples


def sinus_data_generation(nb_examples, fs , freq_range, len_sample, batch_size):

    # dataset loading
    dataset = SinusSamplesDataset(nb_examples=nb_examples, fs=fs, frequency_range=freq_range, len_sinus=len_sample)
    frequency = dataset.frequencies
    samples = dataset.samples

    dataloader = DataLoader(dataset, batch_size)

    return dataloader, frequency, samples


def training(dataloader, testloader, model, lr, epochs):

    print('Starting the training')

    # make list for the plot
    loss_list = []
    val_loss_list = []
    epochs_list = []

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model = model.float()

    for epoch in range(epochs):
        loss = 0
        val_loss = 0
        for i, batch_features in enumerate(dataloader, 0):

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            # print(batch_features.shape)
            # batch_features = Variable(batch_features, requires_grad=True)
            batch_features = batch_features.float()
            batch_features = batch_features.to(device)

            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(dataloader)

        # compute the validation loss
        with torch.no_grad():
            for i, test_features in enumerate(testloader, 0):
                test_features = test_features.float()
                test_features = test_features.to(device)
                outputs = model(test_features)
                test_loss = criterion(outputs, test_features)
                val_loss += test_loss.item()

        # compute the epoch validation loss
        val_loss = val_loss/len(testloader)

        # update lists
        loss_list.append(loss)
        val_loss_list.append(val_loss)
        epochs_list.append(epoch)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}, validation loss = {:.6f}".format(epoch + 1, epochs, loss, val_loss))

    plt.plot(epochs_list, loss_list, label='loss')
    plt.plot(epochs_list, val_loss_list, label='validation loss')
    plt.legend()
    plt.show()

    print("Trained Model!")

    return model


def test_prediction(data, model):

    fig, ax = plt.subplots(3, 3)
    for ii in range(3):
        for jj in range(3):
            # data shape = (X, 2, 11000)

            data_sample = data[ii * 100 + jj + 50]  # (2, 11000)
            data_sample = torch.unsqueeze(data_sample, dim=0)  # (1, 2, 11000)
            model = model.float()
            data_sample = data_sample.to(device)
            prediction = model(data_sample.float())  # (1, 2, 11000)

            # remove 1st fim -> (2, 11000)
            prediction = torch.squeeze(prediction)
            data_sample = torch.squeeze(data_sample)

            # torch.Tensor -> numpy.array
            prediction = prediction.cpu()
            prediction = prediction.detach().numpy()
            data_sample = data_sample.cpu()
            data_sample = data_sample.numpy()

            ax[ii, jj].plot(data_sample[1][:200], color="tab:blue")
            ax[ii, jj].plot(prediction[1][:200], color="tab:red")
            ax[ii, jj].set_xticks([])
            ax[ii, jj].set_yticks([])
    plt.show()
    pass


def visualization_latent_space(model, test_loader, frequencies):

    # get the number of examples
    N = frequencies.shape[0]

    # create array with output encoder
    test_encoded = np.zeros((N, 2, 688))

    with torch.no_grad():
        for i, test_features in enumerate(test_loader, 0):

            # set the input to go to encoder
            test_features = test_features.float()
            test_features = test_features.to(device)

            # pass the input in the encoder
            outputs_encoder = model.encoder(test_features)

            # transform the output in np.array
            outputs_encoder = outputs_encoder.cpu()
            outputs_encoder = outputs_encoder.detach().numpy()

            # save it
            test_encoded[i] = outputs_encoder[0]

    test_freqs_t = np.matlib.repmat(frequencies, 688, 1)
    plt.scatter(test_encoded[:, 0, :], test_encoded[:, 1, :], c=test_freqs_t)
    plt.show()

    pass


def run(model, data_type, data_parameter, batch_size, nb_epochs, train, save, savename, load, loadname, latent_space):

    if data_type == 'audio':
        train_loader, test_loader, samples = audio_data_generation(audio_wav=data_parameter[0],
                                                                   duration_sample=data_parameter[1],
                                                                   batch_size=batch_size)

    if data_type == 'sinus':
        train_loader, train_frequencies, train_samples = sinus_data_generation(nb_examples=data_parameter[0],
                                                                   fs=data_parameter[1],
                                                                   freq_range=data_parameter[2],
                                                                   len_sample=data_parameter[3],
                                                                   batch_size=batch_size)
        print("Train loader ready!")

        test_loader, test_frequencies, test_samples = sinus_data_generation(nb_examples=data_parameter[4],
                                                                   fs=data_parameter[1],
                                                                   freq_range=data_parameter[2],
                                                                   len_sample=data_parameter[3],
                                                                   batch_size=1)

        print("Test loader ready!")

    if train == True:
        trained_model = training(train_loader, test_loader, model, lr=0.001, epochs=nb_epochs)

    if save == True:
        torch.save(trained_model, '../models/'+savename)
        print('Model saved!')

    if load == True:
        model = torch.load('../models/'+loadname)
        model.eval()

        if latent_space == True:
            visualization_latent_space(model, test_loader=test_loader, frequencies=test_frequencies)

    test_prediction(test_samples, model)
    pass


if __name__ == '__main__':

    # run on cpu or gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    kernel_size = 5
    padding = int(kernel_size/2)
    model = AutoEncoder_1D(kernel_size=kernel_size,
                           padding=padding)
    model = model.to(device)

    data_type = 'sinus'

    if data_type == 'audio':
        audio = 'audio_data/caroleeg_perf1.wav'
        duration_sample = 1
        data_param = [audio, duration_sample]

    if data_type == 'sinus':
        nb_examples = 1000
        fs = 11000
        freq_range = [200, 1000]
        len_sample = 11000
        nb_test = int(0.3*1000)
        data_param = [nb_examples, fs, freq_range, len_sample, nb_test]

    batch_size = 16
    nb_epochs = 2

    train = False

    save = False
    savename = 'raw_audio_encoder_1D_sinus_200_1000Hz_21kernel.pt'

    load = True
    loadname = 'raw_audio_encoder_1D_sinus_200_1000Hz_kernel13_epoch25.pt'

    latent_space = True

    run(model, data_type=data_type, data_parameter=data_param, batch_size=batch_size, nb_epochs=nb_epochs,
        train=train, save=save, savename=savename, load=load, loadname=loadname, latent_space=latent_space)