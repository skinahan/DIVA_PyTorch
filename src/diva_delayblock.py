import torch
import diva_sigblock as sb


class DelayBlock(sb.SignalBlock):

    def __init__(self, delay_steps):
        sb.SignalBlock.__init__(self)
        self.delay_steps = delay_steps
        self.currDelay = delay_steps
        self.dataBuffer = [torch.zeros((self.InputPortDimensions[0], 1), dtype=torch.float64)] * self.delay_steps
        self.set_inputs(1)
        self.set_outputs(1)

    def input(self, data):
        self.InputPorts[0] = data
        if len(data.size()) > 1:
            data_sz = max(data.size()[0], data.size()[1])
        else:
            data_sz = data.size()[0]
        if data_sz != self.InputPortDimensions[0]:
            self.setinputdims(data_sz)
        self.dataBuffer.append(data)

    def output(self):
        if len(self.dataBuffer) == 0:
            self.OutputPorts[0] = torch.zeros((self.InputPortDimensions[0], 1), dtype=torch.float64)
        else:
            self.OutputPorts[0] = self.dataBuffer.pop(0)

    # Warning: Clears input buffer!
    def setinputdims(self, dm):
        sb.SignalBlock.setinputdims(self, dm, 0)
        self.dataBuffer = [torch.zeros((self.InputPortDimensions[0], 1), dtype=torch.float64)] * self.delay_steps

    def reset(self):
        self.dataBuffer = [torch.zeros((self.InputPortDimensions[0], 1), dtype=torch.float64)] * self.delay_steps
        self.OutputPorts[0] = torch.zeros((self.OutputPortDimensions[0], 1), dtype=torch.float64)


