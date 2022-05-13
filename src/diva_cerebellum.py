import torch
import diva_firblock as fb
import diva_sigblock as sb
import diva_utils


# DIVA Cerebellum Module
# Applies a smoothing delay to the auditory and somatosensory signal from SSM
# This delay is implemented using Discrete FIR Blocks
class Cerebellum(sb.SignalBlock):

    def __init__(self):
        sb.SignalBlock.__init__(self)
        self.set_inputs(1)
        self.set_outputs(1)
        aud_coeff = torch.flatten(diva_utils.read_file_parameter("expected_hann_11_pt5_pt95.mat", 'h'))#self.diva_hanning(11, 0.5, 0.95)
        #aud_coeff = torch.flatten(self.diva_hanning(11, 0.5, 0.95))
        aud_N = 11 + 1 + 100

        ss_coeff = torch.flatten(diva_utils.read_file_parameter("expected_hann_5_pt5_pt95.mat", 'h'))#self.diva_hanning(5, 0.5, 0.95)
        #ss_coeff = torch.flatten(self.diva_hanning(5, 0.5, 0.95))
        ss_N = 5 + 1 + 100

        self.LDA = fb.FIRBlock(aud_N, aud_coeff, 110)
        self.LDS = fb.FIRBlock(ss_N, ss_coeff, 104)

        self.dataBuffer = []
        initSig = torch.zeros(101).to(torch.float64)
        initSig = torch.unsqueeze(initSig, 1)
        self.dataBuffer.append(initSig)
        self.initialSig = True

    def input(self, data):
        self.InputPorts[0] = data
        self.dataBuffer.append(data)

    def output(self):
        if len(self.dataBuffer) == 0:
            inpData = torch.tensor([0], dtype=torch.float64)
        else:
            inpData = self.dataBuffer.pop(0)
        self.LDA.input(inpData)
        self.LDS.input(inpData)
        self.LDA.output()
        self.LDS.output()
        lda_out = self.LDA.OutputPorts[0]
        lds_out = self.LDS.OutputPorts[0]
        output = torch.cat((self.LDA.OutputPorts[0], self.LDS.OutputPorts[0])).to(torch.float64)
        output = torch.unsqueeze(output, 1)
        self.OutputPorts[0] = output

    def reset(self):
        self.LDS.reset()
        self.LDA.reset()

    def diva_hanning(self, N1, w, factor):
        p1 = [0.0 for i in range(1, N1 + 1)]
        p2 = [1.0]
        p3 = [(float(w) * float(1.0 - factor) * (float(factor) ** float(i))) for i in range(1, 101)]
        h = p1 + p2 + p3
        hSum = sum(h)
        h = [float(x / hSum) for x in h]
        return torch.tensor(h, dtype=torch.float64)
