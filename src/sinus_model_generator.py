import torch
import numpy as np
from sinus_train import test_prediction
import matplotlib.pyplot as plt

# load trained model and sinus data
model = torch.load('models/save_trained_model')
model.eval()

# sinus_data = np.load('samples/sinus_samples.npz')['x']

# test
# test_prediction(sinus_data, model)


# generate data
def generate_data(model):
    input = torch.rand((1, 2, 1, 64))  # shape (2, 1, 64)
    output = model.decoder(input)

    output = output.reshape(output.shape[0],
                            output.shape[2],
                            output.shape[3],
                            output.shape[1])
    output = output.detach().numpy()
    plt.plot(np.squeeze(output)[:200], color="tab:red")
    plt.show()
    pass


generate_data(model)