import torch


class SignalBlock:

    def __init__(self):
        self.NumInputPorts = 1
        self.NumOutputPorts = 1

        self.OutputPortDimensions = [1] * self.NumOutputPorts
        self.InputPortDimensions = [1] * self.NumInputPorts

        self.InputPorts = []
        for i in range(self.NumInputPorts):
            self.InputPorts.append(torch.zeros(self.InputPortDimensions[i], dtype=torch.float64))

        self.OutputPorts = []
        for i in range(self.NumOutputPorts):
            self.OutputPorts.append(torch.zeros(self.OutputPortDimensions[i], dtype=torch.float64))

        self.DWork1 = None
        self.DWork2 = None

    def set_inputs(self, ports):
        self.NumInputPorts = ports
        self.InputPortDimensions = [1] * self.NumInputPorts
        self.InputPorts = []
        for i in range(self.NumInputPorts):
            self.InputPorts.append(torch.zeros(self.InputPortDimensions[i], dtype=torch.float64))

    def set_outputs(self, ports):
        self.NumOutputPorts = ports
        self.OutputPortDimensions = [1] * self.NumOutputPorts
        self.OutputPorts = []
        for i in range(self.NumOutputPorts):
            self.OutputPorts.append(torch.zeros(self.OutputPortDimensions[i], dtype=torch.float64))

    def setinputdims(self, dm, port_no):
        self.InputPortDimensions[port_no] = dm
        self.InputPorts[port_no] = torch.zeros(dm)

    def setoutputdims(self, dm, port_no):
        self.OutputPortDimensions[port_no] = dm
        self.OutputPorts[port_no] = torch.zeros(dm)

    def output(self):
        # default signal block - passthrough
        for i in self.NumInputPorts:
            self.OutputPorts[i] = self.InputPorts[i]

    def reset(self):
        pass
