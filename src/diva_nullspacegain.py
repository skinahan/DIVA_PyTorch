import torch
import diva_sigblock as sb


class NullSpaceGain(sb.SignalBlock):

    def __init__(self, out_type, gain, eps, lmbda, vt):
        sb.SignalBlock.__init__(self)
        self.set_inputs(1)
        self.set_outputs(1)
        # Vocal tract output type
        self.out_type = out_type
        # Gain
        self.gain = gain
        # Jacobian step size
        self.epsilon = eps
        # Jacobian regularization factor
        self.lmbda = lmbda
        self.vt = vt

    def output(self):
        x = self.InputPorts[0]
        if len(x.size()) == 1:
            x = x.unsqueeze(1)

        N = x.size()[0]

        Ix = torch.eye(N, dtype=torch.float64)
        Q = self.vt.diva_vocaltract('base', None)

        DX = self.epsilon * Q

        y = self.vt.diva_vocaltract(self.out_type, x)
        M = y.size()[0]
        Iy = torch.eye(M, dtype=torch.float64)

        DY = torch.zeros((M, N), dtype=torch.float64)
        for ndim in range(N):
            dx_sel = DX[:, ndim]
            dx_sel = dx_sel.unsqueeze(1)
            xt = x + dx_sel
            res = self.vt.diva_vocaltract(self.out_type, xt) - y
            DY[:, ndim] = res[:, 0]
        dy_transp = torch.transpose(DY, 0, 1)
        JJ = torch.matmul(DY, dy_transp)
        # Compute null space projector
        pinv = torch.pinverse(JJ + self.lmbda * self.epsilon ** 2 * Iy)
        N = Ix - torch.matmul(Q, torch.matmul(dy_transp, torch.matmul(pinv, torch.matmul(DY, torch.transpose(Q, 0, 1)))))
        res = self.gain * torch.mm(N, x)
        self.OutputPorts[0] = res
