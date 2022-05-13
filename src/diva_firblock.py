import torch
import numpy as np
from scipy import signal
import diva_sigblock as sb


# Discrete FIR Filter
# Independently filter each channel of the input over time using an FIR filter.
# filter structure : direct form
# input processing: elements as channels (sample based)
class FIRBlock(sb.SignalBlock):

    def __init__(self, N, h, shiftVal, steps):
        sb.SignalBlock.__init__(self)
        self.steps = steps
        self.N = N  # filter length
        self.h = h  # coefficients
        self.x = [torch.tensor([0.] * self.steps)] * N
        self.oldest = 0
        self.set_inputs(1)
        self.set_outputs(1)
        self.last = None
        self.ctr = 0
        self.shiftVal = shiftVal
        self.data = torch.tensor([0])

    def reset(self):
        self.ctr = 0
        self.last = None

    def input(self, data):
        self.data = data
        self.InputPorts[0] = data
        # insert the newest sample into an N-sample circular buffer.
        # the oldest sample in the circular buffer is overwritten
        self.x[self.oldest] = self.data

    # Deprecated, keep around in case...
    def output_backup(self):
        sample = self.data
        # multiply the last N inputs by the appropriate coefficients.
        # their sum is the current output
        y = sample.clone().detach()
        y[y != 0] = 0.
        ctr = 1
        sample_sum = sample.sum()
        if sample_sum == 0:
            b = sample.numel()-1
        else:
            # input index b
            b = (sample == 1).nonzero(as_tuple=True)
            b = b[0].item()
            self.last = sample
        # use the coeff from same idx
        coeff = float(self.h[b].item())
        newval = coeff * sample[b]
        y[ctr] = newval
        ctr += 1
        # iterate backwards in the coefficient indexes (towards 0)
        t = b-1
        decay_coeffs = self.h[:b].tolist()
        decay_coeffs.reverse()
        for coeff in decay_coeffs:
            if ctr == sample.numel():
                break
            for k in range(self.N):
                tens_idx = (self.oldest + k) % self.N
                k_tens = self.x[tens_idx]
                k_sum = k_tens.sum()
                if k_sum > 0:
                    y[ctr] += coeff * k_tens[t]
            t -= 1
            ctr += 1
        self.oldest = (self.oldest + 1) % self.N
        self.OutputPorts[0] = y

    def output(self):
        b = self.h.tolist()
        b.reverse()
        ssm_sig = self.data
        i_sig = torch.flatten(ssm_sig).tolist()
        y = np.array(np.convolve(i_sig, b, mode='full'))
        y = y.tolist()
        # using slicing to left rotate
        shift_amt = self.shiftVal
        y = y[shift_amt:] + y[:shift_amt]
        iSum = sum(i_sig)
        if iSum == 0:
            if self.last is None:
                j = 42 # Do nothing, ignore these zeroes
            else:
                self.last = self.last[-1:] + self.last[:-1]
                y = self.last
        else:
            self.last = y
        y = y[:self.steps]
        y[0] = 0.
        if self.steps > len(y):
            broken=True
        y[self.steps-1] = 0.
        self.ctr += 1
        self.OutputPorts[0] = torch.tensor(y, dtype=torch.float64)
