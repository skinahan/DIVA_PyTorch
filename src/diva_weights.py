import torch
import diva_sigblock as sb
import diva_utils


class WeightBlock(sb.SignalBlock):
    """
    Pytorch implementation of the basic weighted mask block used in the Matlab DIVA implementation
    """

    def __init__(self):
        sb.SignalBlock.__init__(self)
        self.set_inputs(1)
        self.set_outputs(1)
        self.filename = None

        self.M_Idx = 0
        self.Weights = None
        self.Bias = None

        self.weight = None
        self.bias = None

    def set_weight(self, filename):
        self.filename = filename
        self.weight = diva_utils.read_weight_file(filename)
        self.set_bias(filename)

    def set_bias(self, filename):
        self.bias = diva_utils.read_bias_file(filename)

    def get_weights(self, opt):
        """
        if opt == 0:  # load filenames and return weight matrix size
            if self.Weights is None:
                self.Weights = [None] * 5
                self.Bias = [None] * 5
                idx = 0
                ctr = 0
                # load files
                for file in self.Filenames:
                    W, W0 = diva_utils.read_file(file)
                    self.Weights[ctr] = W
                    if W0 is None:
                        f = W.size()
                        W0 = torch.tensor([0] * f[1])
                    self.Bias[ctr] = W0
                    ctr += 1
            else:
                idx = self.M_Idx
                self.weight = self.Weights[idx]
                self.bias = self.Bias[idx]
            return self.weight  # return the size of of weight tensor
        else:
        """
        if opt == 1:  # return index to weight matrix
            return self.M_Idx
        else:  # return weight matrix
            return self.weight.clone(), self.bias.clone()

    def output(self):
        W, W0 = self.get_weights(2)
        x = self.InputPorts[0]
        if len(x.size()) == 1:
            x = x.unsqueeze(1)

        W = W.to(torch.float64)
        masked = W0.to(torch.float64)
        masked = masked[None, :]

        if (torch.max(torch.abs(x))) < 1e-10:  # bias term (if no input)
            masked = W0.to(torch.float64)
        else:
            torch.matmul(torch.transpose(x, 0, 1), W, out=masked)
        return masked

    def reset(self):
        if self.filename is not None:
            self.set_weight(self.filename)
