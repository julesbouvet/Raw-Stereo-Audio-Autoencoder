import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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
    train_dataset = SinusSamplesDataset(nb_examples=nb_examples, fs=fs, frequency_range=freq_range, len_sinus=len_sample)
    samples = train_dataset.samples
    print('Sinus train dataset done!')

    # train test split
    split = int(nb_examples * 0.3)
    # print('split', split)
    test_dataset = SinusSamplesDataset(nb_examples=split, fs=fs, frequency_range=freq_range,
                                        len_sinus=len_sample)
    print('Sinus test dataset done!')

    # dataloader creation
    train_dataloader = DataLoader(train_dataset, batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size)

    return train_dataloader, test_dataloader, samples


def training(dataloader, testloader, model, lr, epochs):

    print('Starting the training')

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
                outputs = model(test_features)
                test_loss = criterion(outputs, test_features)
                val_loss += test_loss.item()

        # compute the epoch validation loss
        val_loss = val_loss/len(testloader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}, validation loss = {:.6f}".format(epoch + 1, epochs, loss, val_loss))

    print("Trained Model!")

    return model


def test_prediction(data, model):

    fig, ax = plt.subplots(3, 3)
    for ii in range(3):
        for jj in range(3):
            print(data.shape)
            # data shape = (X, 2, 11000)

            data_sample = data[ii * 100 + jj + 50]  # (2, 11000)
            print(ii * 3 + jj + 1)
            print(data_sample.shape)
            data_sample = torch.unsqueeze(data_sample, dim=0)  # (1, 2, 11000)
            print(data_sample.shape)
            model = model.float()
            prediction = model(data_sample.float())  # (1, 2, 11000)
            print(prediction.shape)

            # remove 1st fim -> (2, 11000)
            prediction = torch.squeeze(prediction)
            data_sample = torch.squeeze(data_sample)

            # torch.Tensor -> numpy.array
            prediction = prediction.detach().numpy()
            data_sample = data_sample.numpy()
            print(prediction.shape)

            ax[ii, jj].plot(data_sample[0][:200], color="tab:blue")
            ax[ii, jj].plot(prediction[0][:200], color="tab:red")
            ax[ii, jj].set_xticks([])
            ax[ii, jj].set_yticks([])
    plt.show()
    pass


def run(model, data_type, data_parameter, batch_size, train, save, savename, load, loadname):

    if data_type == 'audio':
        train_loader, test_loader, samples = audio_data_generation(audio_wav=data_parameter[0],
                                                                   duration_sample=data_parameter[1],
                                                                   batch_size=batch_size)

    if data_type == 'sinus':
        train_loader, test_loader, samples = sinus_data_generation(nb_examples=data_parameter[0],
                                                                   fs=data_parameter[1],
                                                                   freq_range=data_parameter[2],
                                                                   len_sample=data_parameter[3],
                                                                   batch_size=batch_size)

    if train == True:
        trained_model = training(train_loader, test_loader, model, lr=0.001, epochs=50)

    if save==True:
        torch.save(trained_model, 'models/'+savename)

    if load==True:
        model = torch.load('models/'+loadname)
        model.eval()

    test_prediction(samples, model)
    pass


if __name__ == '__main__':

    model = AutoEncoder_1D()

    data_type = 'sinus'

    if data_type == 'audio':
        audio = 'audio_data/caroleeg_perf1.wav'
        duration_sample = 1
        data_param = [audio, duration_sample]

    if data_type == 'sinus':
        nb_examples = 1000
        fs = 11000
        freq_range = [200, 201]
        len_sample = 11000
        data_param = [nb_examples, fs, freq_range, len_sample]

    train = True

    save = False
    savename = 'raw_audio_encoder_1D.pt'

    load = False
    loadname = 'raw_audio_encoder_1D.pt'

    run(model, data_type=data_type, data_parameter=data_param, batch_size=16,
        train=train, save=save, savename=savename, load=load, loadname=loadname)
