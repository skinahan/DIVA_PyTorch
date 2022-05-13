import diva_sigblock as sb


class SumBlock(sb.SignalBlock):

    def __init__(self, mode):
        sb.SignalBlock.__init__(self)
        self.set_inputs(2)
        self.set_outputs(1)
        self.sum = False
        self.diff_first = False
        self.diff_second = False

        if mode == '++':
            self.sum = True
        if mode == '+-':
            self.diff_second = True
        if mode == '-+':
            self.diff_first = True

    def output(self):
        if self.sum:
            self.OutputPorts[0] = self.InputPorts[0] + self.InputPorts[1]
        if self.diff_second:
            self.OutputPorts[0] = self.InputPorts[0] - self.InputPorts[1]
        if self.diff_first:
            self.OutputPorts[0] = self.InputPorts[1] - self.InputPorts[0]
