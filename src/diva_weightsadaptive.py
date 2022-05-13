import torch
import diva_sigblock as sb
import diva_utils


class AdaptiveWeightBlock(sb.SignalBlock):
    """
        Pytorch implementation of the adaptive weighted mask block used in the Matlab DIVA implementation
    """

    def __init__(self):
        sb.SignalBlock.__init__(self)
        self.Filename = ''
        self.set_inputs(3)
        self.set_outputs(1)

        self.Weights = None
        self.Bias = None
        self.Changes = None

        self.filename = None
        self.weight = None
        self.bias = None
        self.ctr = 0
        self.weight_dims = None

        self.learn_inputs = []

    def set_filename(self, filename):
        self.filename = filename
        self.Filename = filename
        self.get_set_weights(0, None)

    def set_weight(self, filename):
        self.weight = diva_utils.read_weight_file(filename)

    def set_bias(self, filename):
        self.bias = diva_utils.read_bias_file(filename)

    def get_set_weights(self, opt, Wnew):
        if opt == 0:  # load filename and return weight matrix size
            if self.Weights is None:
                W, W0 = diva_utils.read_file(self.Filename)
                self.Weights = [W]
                self.Bias = [W0]
                self.Changes = [[torch.zeros(W.size())]]
                idx = 0
                self.M_Idx = idx
                self.weight = self.Weights[idx]
                self.bias = self.Bias[idx]
                self.weight_dims = self.weight.shape
            else:
                idx = self.M_Idx
                self.weight = self.Weights[idx]
                self.bias = self.Bias[idx]
            return len(self.weight)
        if opt == 1:  # return index to weight matrix
            return self.M_Idx
        if opt == 2:  # return weight matrix
            return self.weight
        if opt == 3:  # set weight matrix
            self.weight = Wnew
            self.Weights[0] = Wnew
        if opt == 4:  # return weight matrix from filename
            W, W0 = diva_utils.read_file(self.Filename)
            self.Weights = [W]
            self.Bias = [W0]
            self.Changes = [[0 * len(W)]]
            idx = 0
            self.M_Idx = idx
            self.weight = self.Weights[idx]
            self.bias = self.Bias[idx]
            return self.weight


    def reset(self):
        pass

    def output(self):
        self.ctr = self.ctr + 1
        W = self.get_set_weights(2, None)
        input1 = self.InputPorts[0] # SSM
        input2 = self.InputPorts[1] # Cerebellum
        input3 = self.InputPorts[2] # Learning Signal (from feedback)
        if len(input1.size()) == 1:
            input1 = input1.unsqueeze(1)
        W = W.to(torch.float64)

        num_nonzero1 = torch.nonzero(input1).size(0)
        num_nonzero2 = torch.nonzero(input2).size(0)
        num_nonzero3 = torch.nonzero(input3).size(0)

        if num_nonzero1 > 0:
            ssmHasData  = 1

        if num_nonzero2 > 0:
            cebHasData = 1

        if num_nonzero3 > 0:
            learnHasData = 1

        transp = torch.transpose(input1, 0, 1)
        t = torch.matmul(transp, W)
        self.OutputPorts[0] = t

        self.learn_inputs.append(input2.tolist())

        if input2 is not None:
            EPS = float(0.5)
            if len(input2.size()) == 1:
                input2 = input2.unsqueeze(1)
            if input3 is not None:
                if torch.is_tensor(input3):
                    if num_nonzero2 > 0 and num_nonzero3 > 0:
                        u = diva_utils.tensor_reshape(input2)
                        #u = input2.reshape(W.shape[0], [], order='F').copy()
                        #u = torch.reshape(input2, W.shape[0])
                        u = torch.squeeze(u)
                        v = diva_utils.tensor_reshape(input3)
                        #v = input3.reshape(W.shape[1], [], order='F').copy()
                        #v = torch.reshape(input3, W.shape[1])
                        v = torch.squeeze(v)
                        # dW = EPS * u * v'
                        transp2 = torch.transpose(v, 0, 1)
                        dW = torch.mm(u, transp2)
                        dW = dW * EPS
                        W = W + dW
                        self.get_set_weights(3, W)
