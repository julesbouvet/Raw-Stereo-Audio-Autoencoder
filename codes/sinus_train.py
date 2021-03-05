import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sinus_autoencoder import AutoEncoder, VAE
import matplotlib.pyplot as plt


# DATA GENERATION
def data_generation(data_npzfile, batch_size):

    # load data
    load_data = np.load(data_npzfile)['x']
    print('data load shape', load_data.shape)

    # reshape data
    data = load_data.reshape(load_data.shape[0], load_data.shape[3],
                             load_data.shape[1], load_data.shape[2])

    print('data new shape', data.shape)

    # train test split
    split = int(load_data.shape[0] * 0.7)
    print('split', split)

    # dataset creation
    train_data = torch.Tensor(data[:split])
    test_data = torch.Tensor(data[split:])

    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)

    # dataloader creation
    train_dataloader = DataLoader(train_dataset, batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size)

    return train_dataloader, test_dataloader, load_data


# defining loss for VAE
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x)
    # BCE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def training(dataloader, model, lr, epochs, VAE=False):

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if VAE==False:
        criterion = nn.MSELoss()

    for epoch in range(epochs):
        loss = 0
        for i, batch_features in enumerate(dataloader, 0):

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            batch_features = torch.Tensor(batch_features[0])
            batch_features = Variable(batch_features, requires_grad=True)

            if VAE==False:
                outputs = model(batch_features)

                # compute training reconstruction loss
                train_loss = criterion(outputs, batch_features)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

            if VAE==True:

                outputs, mu, logvar = model(batch_features)

                train_loss = loss_fn(outputs, batch_features, mu, logvar)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(dataloader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    print("Trained Model!")

    return model


def test_prediction(sinus_data, model, VAE):

    fig, ax = plt.subplots(3, 3)
    for ii in range(3):
        for jj in range(3):
            sinus = sinus_data[ii * 3 + jj + 1]
            # print(sinus.shape)
            sinus_process = sinus.reshape(sinus.shape[2], sinus.shape[0],
                             sinus.shape[1])
            sinus_process = np.expand_dims(sinus_process, axis=0)

            if VAE==True:
                prediction, _, _ = model(torch.Tensor(sinus_process))
            else:
                prediction = model(torch.Tensor(sinus_process))


            prediction_reshape = prediction.reshape(prediction.shape[0],
                                                    prediction.shape[2],
                                                    prediction.shape[3],
                                                    prediction.shape[1])
            prediction_reshape = prediction_reshape.detach().numpy()
            ax[ii, jj].plot(np.squeeze(sinus)[:200], color="tab:blue")
            ax[ii, jj].plot(np.squeeze(prediction_reshape)[:200], color="tab:red")
            ax[ii, jj].set_xticks([])
            ax[ii, jj].set_yticks([])
    plt.show()
    pass


def run(model,VAE,train,save=False, savename='X', load=False, loadname='X'):
    data_npz = 'samples/sinus_samples.npz'
    train_loader, test_loader, sinus_data = data_generation(data_npz, batch_size=1)

    if train == True:
        trained_model = training(train_loader, model, VAE=VAE, lr=0.001, epochs=15)

    if save==True:
        torch.save(trained_model, 'saved_models/'+savename)

    if load==True:
        model = torch.load('saved_models/'+loadname)
        model.eval()

    print('ok3 \n \n \n')
    test_prediction(sinus_data, model, VAE)
    pass

model=VAE()
run(model, train=True, VAE=True, save=False, savename='XXX', load=True, loadname='saved_vae_model')