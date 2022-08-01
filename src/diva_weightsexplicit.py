import torch
import diva_sigblock as sb
import diva_utils


class ExplicitWeightBlock(sb.SignalBlock):
    """
        Pytorch implementation of the explicit weight mask block used in the Matlab DIVA implementation
    """

    def __init__(self, out_type, eps, lmbda, vt):
        sb.SignalBlock.__init__(self)
        self.set_inputs(2)
        self.set_outputs(1)
        # Vocal tract output type
        self.out_type = out_type
        if out_type == 'Somatosensory':
            self.setinputdims(8, 0)
        if out_type == 'Auditory':
            self.setinputdims(4, 0)

        self.setinputdims(13, 1)
        # Jacobian step size
        self.epsilon = eps
        # Jacobian regularization factor
        self.lmbda = lmbda
        self.vt = vt
        self.ctr = 0

    def output(self):
        self.ctr = self.ctr + 1
        # Error Input
        dy = self.InputPorts[0]
        # Current State Input
        x = self.InputPorts[1]

        num_nonzero_x = torch.nonzero(x).size(0)
        if num_nonzero_x > 0:
            beginSeq = True

        if len(x.size()) == 1:
            x = x.unsqueeze(1)

        N = x.size()[0]
        if len(dy.size()) > 1:
            M = max(dy.size()[0], dy.size()[1])
        else:
            M = dy.size()[0]

        Ix = torch.eye(N, dtype=torch.float64)
        Iy = torch.eye(M, dtype=torch.float64)
        Q = self.vt.diva_vocaltract('base', None)
        # Direction of articulatory change

        DX = self.epsilon * Q

        # Direction of auditory / somatosensory change
        DY = torch.zeros((M, N), dtype=torch.float64)

        y = self.vt.diva_vocaltract(self.out_type, x)

        # Computes jacobian
        for ndim in range(N):
            dx_sel = DX[:, ndim]
            dx_sel = dx_sel.unsqueeze(1)
            xt = x + dx_sel
            res = self.vt.diva_vocaltract(self.out_type, xt) - y
            DY[:, ndim] = res[:, 0]

        dy_transp = torch.transpose(DY, 0, 1)
        JJ = torch.matmul(DY, dy_transp)
        #DY * torch.transpose(DY, 0, 1)

        # Computes pseudoinverse
        pinv = torch.pinverse(JJ+self.lmbda*self.epsilon**2*Iy)
        iJ = self.epsilon * torch.matmul(Q, torch.matmul(dy_transp, pinv))
        dx = -1 * torch.matmul(iJ, dy)
        if len(dx.size()) == 1:
            dx = dx.unsqueeze(1)
        self.OutputPorts[0] = dx
