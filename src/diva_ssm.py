import math
import numpy
import time
import torch
import diva_utils
import diva_sigblock as sb
import diva_delayblock as ddb


def print_time(threadName, delay):
    count = 0
    while count < 5:
        time.sleep(delay)
        count += 1
        print(threadName, time.ctime(time.time()))


class SSMSequencer(sb.SignalBlock):

    def __init__(self):
        sb.SignalBlock.__init__(self)
        self.set_inputs(2)
        self.set_outputs(1)
        self.ctr = 0

    def output(self):
        self.ctr += 1
        eps = .01
        N = self.OutputPortDimensions[0]
        n = self.OutputPortDimensions[0]
        out = torch.zeros((N, 1), dtype=torch.float64)

        for i in self.InputPorts[0]:  # loads input sequentially
            if i > eps:
                self.DWork1 = torch.tensor(numpy.arange(0, n))
                self.DWork2 = 1
                break
        if self.InputPorts[1] > 0 and self.DWork2 > 0:  # outputs input sequentially
            idx = self.DWork1
            n = self.DWork2
            if n == 0:
                n = n + self.InputPorts[1]  # add sub-sample jitter
            n0 = math.floor(n)
            n1 = n - n0
            if n0 + 1 >= N:  # end of sequence
                self.DWork2 = 0
            else:  # new sample point
                if n0 <= 1:
                    out[idx[1]] = 1
                else:
                    out[idx[n0]] = 1 - n1
                    out[idx[n0 + 1]] = n1
                self.DWork2 = n + self.InputPorts[1]
        self.OutputPorts[0] = out


# INPUT TO SPEECH SOUND MAP: PRODUCTION UNIT SIGNAL
# OUTPUT OF SPEECH SOUND MAP: TIME-VARYING 16-DIMENSIONAL INPUT
class SpeechSoundMap(sb.SignalBlock):
    """
        PyTorch implementation of the DIVA Speech Sound Map module
    """

    def __init__(self, steps):
        sb.SignalBlock.__init__(self)
        self.set_inputs(2)
        self.set_outputs(4)

        self.speed_signal = 1
        self.Files = []
        self.Filenames = ["diva_weights_SSM2amax.mat", "diva_weights_SSM2amin.mat", "diva_weights_SSM2smax.mat",
                          "diva_weights_SSM2smin.mat", "diva_weights_SSM.mat"]

        self.sequencer = SSMSequencer()

        self.M_Idx = 0
        self.Weights = None
        self.Bias = None

        # Learned delay - Auditory
        self.LDA = ddb.DelayBlock(11)
        self.LDA.setinputdims(steps)
        # Learned delay - Somatosensory
        self.LDS = ddb.DelayBlock(5)
        self.LDS.setinputdims(steps)


        self.weight = None
        self.bias = None
        #default_out = torch.zeros(101, 1, dtype=torch.float64)
        self.dataBuffer = []

        self.get_weights(0)

    def set_weight_idx(self, idx):
        self.M_Idx = idx
        self.weight = self.Weights[idx]
        self.bias = self.Bias[idx]

    def set_weight(self, filename):
        self.weight = diva_utils.read_weight_file(filename)

    def set_bias(self, filename):
        self.bias = diva_utils.read_bias_file(filename)

    def reset(self):
        self.Weights = None
        self.Bias = None
        self.get_weights(0)

    def get_weights(self, opt):
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
                        W0 = torch.tensor([0] * f[1], dtype=torch.float64)
                    self.Bias[ctr] = W0
                    ctr += 1
            else:
                idx = self.M_Idx
                self.weight = self.Weights[idx].clone().detach()
                self.bias = self.Bias[idx].clone().detach()
            return self.weight  # return the size of of weight tensor
        else:
            if opt == 1:  # return index to weight matrix
                return self.M_Idx
            else:  # return weight matrix
                idx = self.M_Idx
                self.weight = self.Weights[idx]
                self.bias = self.Bias[idx]
                return self.weight, self.bias

    def weight_output(self):
        # take production input signal
        # mask it against diva_weights_SSM (standard weight element -> fixed weight, matrix multiplication)
        # this code corresponds to diva_weights.m in the simulink model
        # (W, W0) = self.ssm2amax

        self.set_weight_idx(4)
        W, W0 = self.get_weights(2)
        x = self.InputPorts[0];

        # W = self.weight
        # W0 = self.bias
        W = W.to(torch.float64)
        masked = W0.to(torch.float64)
        masked = masked[None, :]
        
        if (torch.max(torch.abs(x))) < 1e-10:  # bias term (if no input)
            # print("no input")
            masked = W0
        else:
            torch.matmul(x, W, out=masked)
        return masked

    def output(self):
        speed_signal = self.InputPorts[1]
        if speed_signal > 0:
            self.set_weight_idx(4)
            maskedVal = self.weight_output()
            f = maskedVal.size()
            dims = f[0]
            # output from temporal representation neurons goes to diva_sequencer
            # diva_sequencer generates a "sweep of activation" through the last set of activated neurons
            self.sequencer.setoutputdims(dims, 0)
            # Pass input to the sequencer block
            self.sequencer.InputPorts[0] = maskedVal
            self.sequencer.InputPorts[1] = speed_signal

            self.sequencer.output()
            self.sequencer.InputPorts[0] = [0]
            self.dataBuffer.append(self.sequencer.OutputPorts[0])
            self.OutputPorts[0] = self.dataBuffer.pop(0)

            #self.LDA.input(self.OutputPorts[0])
            #self.LDS.input(self.OutputPorts[0])
            #self.LDA.output()
            #self.LDS.output()

            self.OutputPorts[1] = self.OutputPorts[0]
            self.OutputPorts[2] = self.LDA.OutputPorts[0]
            self.OutputPorts[3] = self.LDS.OutputPorts[0]

    def output_seq(self):
        no_out = torch.tensor(0, dtype=torch.float64)
        curr_seq_out = self.sequencer.OutputPorts[0]
        if torch.is_tensor(curr_seq_out):
            self.dataBuffer.append(curr_seq_out)
            self.OutputPorts[0] = self.dataBuffer.pop(0)

            self.LDA.input(self.OutputPorts[0])
            self.LDS.input(self.OutputPorts[0])
            self.LDA.output()
            self.LDS.output()

            self.OutputPorts[1] = self.OutputPorts[0]
            self.OutputPorts[2] = self.LDA.OutputPorts[0]
            self.OutputPorts[3] = self.LDS.OutputPorts[0]
            self.sequencer.output()
        else:
            if self.sequencer.OutputPorts[0] == 0:
                self.sequencer.output()


def test_weights(fileName, outFileName, weightFile, biasFile, actualFileName):
    production = diva_utils.read_input_file(fileName)
    ssm = SpeechSoundMap()
    ssm.set_weight(weightFile)
    ssm.set_bias(biasFile)
    output = ssm.run(torch.FloatTensor(production.float()))
    expectedOut = diva_utils.read_output_file(outFileName)
    torch.save(output, actualFileName)
    testPass = torch.equal(output, expectedOut.float())
    if (testPass):
        print("Unit Test Successful!")
    else:
        print("Unit Test Failed!")
        print("Expected:")
        print(expectedOut)
        print("Actual:")
        print(output)
    print(fileName)
    print(outFileName)


def runTests():
    for i in range(0, 11):
        test_weights("PyTorch_DIVA_Testing/i/" + str(i) + "/SSM_input.mat",
                     "PyTorch_DIVA_Testing/i/" + str(i) + "/SSM_output.mat",
                     "PyTorch_DIVA_Testing/i/" + str(i) + "/SSM_W.mat",
                     "PyTorch_DIVA_Testing/i/" + str(i) + "/SSM_W0.mat",
                     "PyTorch_DIVA_Testing/i/" + str(i) + "/SSM_ACTUAL.pt")
