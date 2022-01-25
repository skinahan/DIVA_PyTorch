import torch
import diva_sigblock as sb
import diva_delayblock as db
import diva_sumblock as sumb


class Vel2Pos(sb.SignalBlock):

    def __init__(self, factor):
        sb.SignalBlock.__init__(self)
        self.set_inputs(1)
        self.set_outputs(1)
        # Integrator factor
        self.factor = factor
        self.learned_delay_aud = db.DelayBlock(1)
        self.learned_delay_aud.setinputdims(13)

        self.sum_block = sumb.SumBlock('++')

    def reset(self):
        self.learned_delay_aud.reset()

    def output(self):
        in1 = self.InputPorts[0]
        # Advance delay block...
        self.learned_delay_aud.output()
        in2 = self.learned_delay_aud.OutputPorts[0]
        if len(in2.size()) == 1:
            in2 = in2.unsqueeze(1)
        # Apply gain factor
        in2 = self.factor * in2

        # Sum arguments
        self.sum_block.InputPorts[0] = in1
        self.sum_block.InputPorts[1] = in2
        self.sum_block.output()
        sum_out = self.sum_block.OutputPorts[0]
        self.learned_delay_aud.input(sum_out)

        # Apply second gain factor
        sum_out = sum_out * (1 - self.factor)
        self.OutputPorts[0] = sum_out
