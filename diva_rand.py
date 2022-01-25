import torch
import torch.nn as nn
import numpy as np
import diva_vocaltract as vt
import rbfn_test
import torch_rbf as rbf


class Babbler():
    def __init__(self):
        pass

    # Generates 1000 random articulator inputs
    # and returns the resulting auditory and somatosensory output of the vocal tract
    def generate_movement(self):
        vocaltract = vt.VocalTract()
        conv_h = vocaltract.hanning(51) / sum(vocaltract.hanning(51))
        x = np.random.randn(13, 1000)
        x = np.reshape(x, x.size)
        x = np.convolve(x, conv_h, mode='same') * 1.25
        x = np.reshape(x, (13, 1000))
        x = x + np.flip(x)
        x[10, :] = 0
        x[11, :] = 0.5
        x[12, :] = x[12, :] + 0.5
        fs = 11025
        x = torch.from_numpy(x)
        # vocaltract.dosound(x, True)
        aud, som, outline, af, filt = vocaltract.diva_synth(x, 'audsom')
        y = aud
        z = som
        return x, y, z

    def train_model(self):
        samples = 1000

        steps = 100
        epochs = 100

        # Modified by: Sean Kinahan, 7/28/2021
        # Instantiating and training an RBF network with the Gaussian basis function
        # This network receives a 10-dimensional input, transforms it into a 32-dimensional
        # hidden representation with an RBF layer and then transforms that into a
        # 4-dimensional output/prediction with a linear layer

        # To add more layers, change the layer_widths and layer_centres lists

        layer_widths = [10, 3]
        layer_centres = [352]
        basis_func = rbf.inverse_multiquadric

        rbfnet = rbfn_test.Network(layer_widths, layer_centres, basis_func)
        cuda0 = torch.device('cuda:0')
        for i in range(4000):
            print("Sample: " + str(i))
            x, y, z = self.generate_movement()
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            rbfnet.fit(x[0:10, :].to(torch.float), y[1:, :].to(torch.float), epochs, 1, 0.01, nn.MSELoss())

        rbfnet.eval()

        # Plotting the ideal and learned decision boundaries

        with torch.no_grad():
            preds = rbfnet.forward(x[0:10, :].unsqueeze(0).to(torch.float)).data.numpy()
            preds = preds.transpose(0, 1)
            return preds
