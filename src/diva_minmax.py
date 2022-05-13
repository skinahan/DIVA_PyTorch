import torch
import diva_sigblock as sb


class MinMax(sb.SignalBlock):

    def __init__(self, mode):
        sb.SignalBlock.__init__(self)
        self.set_inputs(2)
        self.set_outputs(1)
        if mode == 'min':
            self.min = True
        else:
            self.min = False

    def output(self):
        in1 = self.InputPorts[0]
        in2 = self.InputPorts[1]

        if in1.size()[0] != in2.size()[0]:
            in2 = torch.transpose(in2, 0, 1)

        if self.min:
            self.OutputPorts[0] = torch.minimum(in1, in2)
        else:
            self.OutputPorts[0] = torch.maximum(in1, in2)