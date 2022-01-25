import math
import cmath
import torch
import diva_sigblock as sb
import vocaltract_args as vta
import diva_utils
import numpy as np
from itertools import product
import numpy as np
from scipy.io.wavfile import write
from scipy.special import erfinv
import sounddevice as sd
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import librosa
import time


# compute vocal tract filter
# returns H, f, Hc
def a2h(a, l, n, fs, closure, mina):
    if mina is None:
        mina = np.minimum(a, [], [1])

    if closure is None:
        closure = 0

    if fs is None:
        fs = 11025

    # c = speed of sound (cm/s)
    c = float(34326)

    NL, ML = 1, 1
    N, M = a.size()

    ceil_n_2 = math.ceil(n / 2)
    m = ceil_n_2 + 1
    t_cn2 = torch.tensor([i for i in range(m)], dtype=torch.float64)
    t_cn2 = t_cn2.unsqueeze(1)

    f = np.divide(np.multiply([fs], t_cn2), n)
    t = l / c

    # reflection at lips (low pass)
    neg_abs_f = np.multiply([-1], np.abs(f))
    abs_f_div = np.divide(neg_abs_f, [4e3])
    f_pow_2 = np.power(abs_f_div, 2)
    f_pow_2 = np.multiply([-1], f_pow_2)
    f_exp = np.exp(f_pow_2)
    Rrad = np.multiply([0.9], f_exp)
    if m < 0:
        broken = True
    H = torch.zeros(m, M)
    Hc = torch.zeros(m, M)

    if mina == 0:
        a = np.maximum([0.5], a)

    for nM in range(M):
        pi_f_t = np.multiply([math.pi], f)
        pi_f_t = np.multiply([t], pi_f_t)
        coswt = np.cos(np.multiply([2], pi_f_t))
        sinwt = np.sin(np.multiply([2], pi_f_t))
        sinwt = np.multiply([complex('1j')], sinwt)

        a2_m = a[1:N, nM]
        a1_m = a[0:N - 1, nM]
        sub_arg = np.subtract(a2_m, a1_m)
        add_arg = np.add(a2_m, a1_m)
        eps = 2 ** (-52)
        max_res = np.maximum([eps], add_arg)
        r_rest = np.divide(sub_arg, max_res)
        R = np.array([0.9])
        R = np.concatenate((R, r_rest))
        U = np.add(coswt, sinwt)
        V = np.subtract(coswt, sinwt)

        h1 = torch.ones(m, 1)
        h2 = torch.zeros(m, 1)
        for nN in range(N - 1):
            RnN = -1 * R[nN]
            u = h1 + RnN * h2
            v = h2 + RnN * h1
            # reflection
            if closure == nN:
                sub_res = np.subtract(u, v)
                Hc = np.subtract(u, v)
            if NL == 1:
                h1 = np.multiply(U, u)
                h2 = np.multiply(V, v)
                # delay
            else:
                h1 = np.multiply(U[:, nN], u)
                h2 = np.multiply(V[:, nN], v)
        # u = np.multiply(Rrad, h2)
        # u = np.subtract(h1, u)
        u = h1 - Rrad * h2
        # reflection
        h = u
        if closure >= N:
            # val = np.multiply(Rrad, h1)
            # val = np.subtract(h2, val)
            # Hc[:, nM] = np.subtract(u, val)
            Hc[:, nM] = u - (h2 - Rrad * h1)

        mult_arg_1 = np.add([1], Rrad)
        mult_arg_2 = np.prod(np.add([1], R))
        div_arg_1 = np.multiply(mult_arg_1, mult_arg_2)
        div_arg_2 = h
        H = np.divide(div_arg_1, div_arg_2)
        # H[:, nM] = (1 + Rrad) * np.prod(np.add([1], R)) / h[:, 0]
        if closure > 0:
            # Hc[:, nM] = (1 + Rrad) * np.prod(np.add([1], R[closure+1:N])) * Hc[:, nM] / h[:, 0]
            mult_arg_2 = np.prod(np.add([1], R[closure + 1:N]))
            mult_arg_3 = Hc
            div_arg_1 = np.multiply(mult_arg_1, mult_arg_2)
            div_arg_1 = np.multiply(div_arg_1, mult_arg_3)
            Hc = np.divide(div_arg_1, div_arg_2)
            bleh = 34

    # torch.conj(H[1+(n-m:-1:1), :])

    """
    concatenate along dimension 1...
    with the complex conjugate of...?
        H(1+(n-m:-1:1),:)
    H=cat(1,H,conj());
    Hc=cat(1,Hc,conj(Hc(1+(n-m:-1:1),:)));
    f=cat(1,f,-f(1+(n-m:-1:1)));
    if mina==0, 
        H=0*H; 
    end
    """
    h_sel = H[1:(n - m) + 1]
    h_sel = torch.flip(h_sel, [0])
    # h_sel = torch.tensor(h_sel, dtype=torch.complex64)

    hc_sel = Hc[1:(n - m) + 1]
    hc_sel = torch.flip(hc_sel, [0])
    # hc_sel = torch.tensor(hc_sel, dtype=torch.complex64)

    f_sel = f[1:(n - m) + 1]
    f_sel = torch.flip(f_sel, [0])
    f_sel = np.multiply([-1], f_sel)

    H = torch.cat((H, torch.conj(h_sel)), dim=0)
    Hc = torch.cat((Hc, torch.conj(hc_sel)), dim=0)
    f = torch.cat((f, f_sel), dim=0)

    if mina == 0:
        H = np.multiply([0], H)

    return H, f, Hc


# helper method for indexing..
def awol(input, other):
    out = [0 for i in range(len(other))]
    in_flat = input.flatten().tolist()
    for idx, val in enumerate(other):
        out[idx] = in_flat[val]
    return out


def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
    """
    An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1)
    a : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array.
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.

    Returns
    -------
    out : ndarray
        The accumulated results.

        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.


    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = accum(accmap, a)
    array([9, 7, 15])
    >>> # A 2D output, from sub-arrays with shapes and positions like this:
    >>> # [ (2,2) (2,1)]
    >>> # [ (1,2) (1,1)]
    >>> accmap = array([
            [[0,0],[0,0],[0,1]],
            [[0,0],[0,0],[0,1]],
            [[1,0],[1,0],[1,1]],
        ])
    >>> # Accumulate using a product.
    >>> accum(accmap, a, func=prod, dtype=float)
    array([[ -8.,  18.],
           [ -8.,   9.]])
    >>> # Same accmap, but create an array of lists of values.
    >>> accum(accmap, a, func=lambda x: x, dtype='O')
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)
    """

    # Check for bad arguments and handle the defaults.
    if accmap.shape[:a.ndim] != a.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = a.dtype
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = []
    # vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals.append([])
    for s in product(*[range(k) for k in a.shape]):
        indx = accmap[s][0]
        # indx = tuple(accmap[s])
        val = a[s]
        # vals[indx] = np.append(vals[indx], val)
        vals[indx].append(val)

    vals = np.array(vals, dtype=object)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s[0]] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s[0]])

    return out


"""
# Liljencrants-Fant glottal model U = (D,T,P)
# d is derivative of flow waveform: must be 0, 1, or 2
# t is in fractions of a cycle
# p has one row per output point
%	p(:,1)=open phase [0.6]
%	p(:,2)=+ve/-ve slope ratio [0.1]
%	p(:,3)=closure time constant/closed phase [0.2]
% Note: this signal has not been low-pass filtered
% and will therefore be aliased
%
% Usage example:	ncyc=5;
%			period=80;
%			t=0:1/period:ncyc;
%			ug=glotlf(0,t);
%			plot(t,ug)
%      Copyright (C) Mike Brookes 1998
%
%      Last modified Thu Apr 30 17:22:00 1998
%
%   VOICEBOX home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def glotlf(d, t, p):
    tt = None
    de = [float(0.6), float(0.1), float(0.2)]
    if t is None:
        if p is None:
            tt = torch.tensor([i for i in range(99)], dtype=torch.float64)
            tt = tt / 100
    else:
        tt = np.subtract(t, np.floor(t))
        # tt = t - math.floor(t)
        if p is None:
            p = de
        else:
            if len(p) < 2:
                p = []
                # p = [p(:); de(length(p)+1:2)]
    u = torch.zeros(len(tt), 1, dtype=torch.float64)
    te = p[0]
    mtc = te - 1
    e0 = 1.0
    wa = float(math.pi / float(te * (1 - p[2])))

    a = -math.log(-p[1] * math.sin(wa * te)) / te
    inta = e0 * ((wa / math.tan(wa * te) - a) / p[1] + wa) / (a ** 2 + wa ** 2)
    rb0 = p[1] * inta
    rb = rb0

    # Use Newton to determine closure time constant so that flow starts and ends at zero
    for i in range(1, 4):
        kk = 1 - math.exp(mtc / rb)
        # ERR should be -5.1645e-05, currently getting ~0.0625
        err = rb + mtc * (1 / kk - 1) - rb0
        derr = 1 - (1 - kk) * (mtc / rb / kk) ** 2
        rb = rb - err / derr
    e1 = float(1 / (p[1] * (1 - math.exp(mtc / rb))))

    ta = [j < te for j in tt]
    # ta = tt < te
    tb = [not j for j in ta]
    # tb = not ta

    if d == 0:
        for i in range(len(ta)):
            if ta[i] == 1:
                u[i] = e0 * (math.exp(a * tt[i] * ta[i]) *
                             ((a * math.sin(wa * tt[i] * ta[i])) - wa * math.cos(wa * tt[i] * ta[i])) + wa) \
                       / float(a ** 2 + wa ** 2)
            if tb[i] == 1:
                u[i] = e1 * (math.exp(mtc / rb) * (tt[i] * tb[i] - 1 - rb) + math.exp((te - tt[i] * tb[i]) / rb) * rb)
    if d == 1:
        for i in range(len(ta)):
            if ta[i] == 1:
                u[i] = e0 * math.exp(a * tt[i] * ta[i]) * math.sin(wa * tt[i] * ta[i])
            if tb[i] == 1:
                u[i] = e1 * (math.exp(mtc / rb) - math.exp((te - tt[i] * tb[i]) / rb))
    if d == 2:
        for i in range(len(ta)):
            if ta[i] == 1:
                u[i] = e0 * math.exp(a * t[i] * ta[i]) * (
                        a * math.sin(wa * tt[i] * ta[i]) + wa * math.cos(wa * tt[i] * ta[i]))
            if tb[i] == 1:
                u[i] = e1 * math.exp((te - tt[i] * tb[i]) / rb) / rb
    if not (d == 0 or d == 1 or d == 2):
        print("ERROR: DERIVATIVE MUST BE 0, 1, or 2")
    return u


def minus(x, y):
    return x - y


class Synth:
    def __init__(self):
        self.fs = 11025
        self.update_fs = 200
        self.f0 = 120
        self.samplesperperiod = math.ceil(self.fs / self.f0)
        stepsize = float(1 / self.samplesperperiod)
        d = [0 + (i * stepsize) for i in range(self.samplesperperiod)]
        self.glottalsource = glotlf(0, torch.tensor(d, dtype=torch.float64), None)
        self.f = [0, 1]
        self.filt = [0, 0]
        self.pressure = 0
        self.voicing = 1
        self.pressurebuildup = 0
        self.pressure0 = 0
        self.sample = torch.zeros(self.samplesperperiod)
        self.sample = self.sample.unsqueeze(1)
        self.k1 = 1
        self.numberofperiods = 1
        self.samplesoutput = 0


class VocalTract(sb.SignalBlock):
    """
        Pytorch implementation of the vocal tract model block used in the Matlab DIVA implementation
    """

    def __init__(self):
        sb.SignalBlock.__init__(self)
        self.vt = None
        self.set_inputs(1)
        self.set_outputs(2)
        self.t = 0
        self.fmfit = None
        self.ab_alpha = None
        self.ab_beta = None
        self.params = vta.VocalTractArgs()
        self.last_prod = None

    def hanning(self, n):
        if n % 2 == 0:  # Even
            varg = torch.tensor([i for i in range(1, n / 2)], dtype=torch.float64)
            varg = varg * (2 * math.pi)

            varg = varg / (n + 1)
            varg = torch.cos(varg)
            varg = 1 - varg
            w = 0.50 * varg
            # w = 0.50 * (1 - math.cos(2*pi*(1:n/2)))
            w = torch.cat(w, torch.flipud(w))
        else:  # Odd
            ar = [i for i in range(1, int((n + 1) / 2) + 1)]
            varg = torch.tensor(ar, dtype=torch.float64)
            varg = varg * (2 * math.pi)
            varg = varg / (n + 1)
            varg = torch.cos(varg)
            varg = 1 - varg
            w = 0.50 * varg
            # w = .5*(1 - cos(2*pi*(1:(n+1)/2)'/(n+1)));
            w = torch.cat((w, torch.flipud(w[:25])), 0)
        return w

    # a, b, sc, af, d = self.xy2ab(outline, None)
    # Computes area function
    def xy2ab(self, x, y):

        if self.ab_alpha is None:
            amax = 220
            alpha = [1, 1, 1, 1, 1, 1, 1]
            beta = [0.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25]
            ab_alpha = [0 for i in range(amax)]
            ab_beta = [0 for i in range(amax)]
            idx = [[i for i in range(0, 60)],
                   [i for i in range(60, 70)],
                   [i for i in range(70, 80)],
                   [i for i in range(80, 120)],
                   [i for i in range(120, 150)],
                   [i for i in range(150, 190)],
                   [i for i in range(190, amax)]]

            for i in range(0, len(idx)):
                idxrow = idx[i]
                for j in idxrow:
                    ab_alpha[j] = alpha[i]
                    ab_beta[j] = beta[i]
            hann_51 = self.hanning(51)
            hann_51_sum = torch.sum(hann_51)
            h = hann_51 / hann_51_sum.item()
            arg_ab_alpha_convn = [float(1) for i in range(270)]
            arg_ab_beta_convn = []
            for i in range(270):
                val = float(0.25)
                if i >= 84:
                    val = float(1.25)
                arg_ab_beta_convn.append(val)
            alpha = arg_ab_alpha_convn  # torch.flatten(arg_ab_alpha_convn).tolist()
            ab_alpha = np.array(np.convolve(alpha, h, mode='valid'))

            beta = arg_ab_beta_convn  # torch.flatten(arg_ab_beta_convn).tolist()
            ab_beta = np.array(np.convolve(beta, h, mode='valid'))

            ab_beta = ab_beta.tolist()
            shift_amt = -1
            # using slicing to left rotate
            ab_beta = ab_beta[shift_amt:] + ab_beta[:shift_amt]
            ab_beta[0] = ab_beta[1]

            self.ab_alpha = ab_alpha
            self.ab_beta = ab_beta

        ab_alpha = self.ab_alpha
        ab_beta = self.ab_beta
        orig_x = x
        if y is None:
            const_arg = cmath.exp(complex('-1j') * complex(math.pi / 12))
            x = np.multiply(x.cpu(), [const_arg])
            # x = cmath.exp() * x
            y = torch.imag(x)
            x = torch.real(x).to(torch.float64)

        x0 = 45
        y0 = -100
        r = 60
        k = math.pi * r / 2
        d = 0.75 / 10

        a = torch.zeros(len(x))
        b = torch.zeros(len(x))

        i1 = y < y0
        i2 = x < x0

        p1 = y >= y0
        p2 = x >= x0
        i3 = torch.logical_and(p1, p2)

        # a,b: "linearized coordinates along vocal tract
        for idx, val in enumerate(i1):
            if val.numel() == 0:
                broken = True
            else:
                if val:
                    a[idx] = y[idx] - y0
                    b[idx] = x[idx] - x0
        for idx, val in enumerate(i2):
            if val.numel() == 0:
                broken = True
            else:
                if val:
                    a[idx] = k + x0 - x[idx]
                    b[idx] = y[idx] - y0
        z = torch.zeros(len(x), dtype=torch.complex64)
        for idx, val in enumerate(i3):
            if val.numel() == 0:
                broken = True
            else:
                if val:
                    z[idx] = x[idx] - x0 + complex('1j') * (y[idx] - y0)
        for idx, val in enumerate(i3):
            if val.numel() == 0:
                broken = True
            else:
                if val:
                    p = math.atan2(z[idx].imag, z[idx].real);
                    a[idx] = r * p
                    b[idx] = abs(z[idx])
        # Tube area
        olips = [i for i in range(29, 45)]
        ilips = [i for i in range(256, 302)]
        owall = [i for i in range(44, 164)]
        iwall = [i for i in range(163, 257)]
        oall = [i for i in range(29, 164)]
        iall = [i for i in range(163, 302)]
        xmin = -20
        ymin = -160
        amin = ymin - y0
        amax = math.ceil((x0 - xmin + k - amin))

        fact = 3
        out = torch.zeros(len(oall), dtype=torch.float64)
        arg_2 = []
        for idx, val in enumerate(oall):
            arg_2.append(b[val])
            out[idx] = (fact * 9) * (a[val] - amin) / amax
        out = torch.ceil(out)
        accmap = out.to(torch.int64)
        arg_3 = [fact * 9, 1]

        accmap = np.array(accmap.numpy())
        arg_2 = np.array(arg_2)
        wallab1 = accum(accmap, arg_2, func=np.min, size=arg_3, fill_value=math.nan, dtype=float)

        out = torch.zeros(len(iwall), dtype=torch.float64)
        arg_2 = []
        for idx, val in enumerate(iwall):
            arg_2.append(b[val])
            out[idx] = (fact * 9) * (a[val] - amin) / amax
        out = torch.ceil(out)
        accmap = out.to(torch.int64)
        accmap = np.array(accmap.numpy())
        arg_2 = np.array(arg_2)

        wallab2 = accum(accmap, arg_2, func=np.max, size=arg_3, fill_value=math.nan, dtype=float)

        lipsab1 = np.min(awol(b, olips))
        lipsab2 = np.max(awol(b, ilips))

        sub_arg_1 = wallab1[fact * 2 + 1:fact * 8 + 1]
        sub_arg_2 = wallab2[fact * 2 + 1:fact * 8 + 1]

        rshp_arg_1 = sub_arg_1 - sub_arg_2
        rshp_arg_1 = torch.from_numpy(rshp_arg_1)
        rshp_arg_2 = [fact, 6]
        rshp_out = np.reshape(rshp_arg_1, rshp_arg_2, order="F")
        mind = np.nanmin(rshp_out, axis=0)

        first_d = torch.from_numpy(mind[0:4])
        first_d = first_d.unsqueeze(0)
        prod_arg_1 = torch.transpose(first_d, 0, 1)
        prod_arg_1 = torch.cat((prod_arg_1, torch.tensor([min(mind[4:6])]).unsqueeze(0)), 0)
        prod_arg_1 = torch.cat((prod_arg_1, torch.tensor([lipsab1 - lipsab2]).unsqueeze(0)), 0)

        sc = d * prod_arg_1

        w = 2

        np_sub_arg = np.subtract(awol(a, oall), [amin])
        np_rnd_arg = np.round(np_sub_arg)
        accmap = np_rnd_arg.astype(dtype=int)
        arg_2 = awol(b, oall)
        arg_2 = np.array(arg_2)
        arg_3 = np.array([amax, 1])

        # ADJUST FOR OFF-BY-ONE ERROR IN AB1
        accmap = np.subtract(accmap, [1])
        ab1 = accum(accmap, arg_2, func=np.nanmin, fill_value=math.nan, size=arg_3, dtype=float)

        np_sub_arg = np.subtract(awol(a, iall), [amin])
        np_rnd_arg = np.round(np_sub_arg)
        accmap = np_rnd_arg.astype(dtype=int)
        arg_2 = awol(b, iall)
        arg_2 = np.array(arg_2)
        # ADJUST FOR OFF-BY-ONE ERROR (indexing diff.)
        accmap = np.subtract(accmap, [1])
        ab2 = accum(accmap, arg_2, func=np.nanmax, size=arg_3, fill_value=math.nan, dtype=float)

        for n1 in range(w):
            i_arg1 = np.array(ab1[1:len(ab1) - 1])
            i_arg2 = np.array(ab1[0:len(ab1) - 2])
            inner_ = np.fmin(i_arg1, i_arg2)
            ab1[1:len(ab1) - 1] = np.fmin(inner_, ab1[2:len(ab1)])

        for n1 in range(w):
            inner_ = np.fmax(ab2[1:len(ab1) - 1], ab2[0:len(ab1) - 2])
            ab2[1:len(ab2) - 1] = np.fmax(inner_, ab2[2:len(ab2)])

        i = ab1 > 0
        i2 = ab2 > 0
        i = np.logical_and(i, i2)
        c = []
        ab_alpha_us = []
        ab_beta_us = []

        for idx, val in enumerate(i):
            if val:
                c.append(ab1[idx][0] - ab2[idx][0])
                ab_alpha_us.append(ab_alpha[idx])
                ab_beta_us.append(ab_beta[idx])
        ab_alpha_us = np.array(ab_alpha_us)
        ab_beta_us = np.array(ab_beta_us)
        c = torch.tensor(c, dtype=torch.float64)
        af = d * c
        # source position
        idx = 0
        for q0 in enumerate(af):
            if q0[1] > 0:
                idx = q0[0]
                break
        # af: area function
        af_bet = np.minimum(af, [0])

        prod_arg_2 = np.power(np.maximum([0], af), ab_beta_us)

        af = np.multiply(ab_alpha_us, prod_arg_2)
        af = af_bet + af
        af = af[idx:len(af)]
        af = af.unsqueeze(1)

        return a, b, sc, af, d

    def diva_vt(self):
        filename = "diva_synth.mat"
        vt = diva_utils.read_file_parameter_alternate(filename, 'vt')
        vt = vt[0, 0]
        vt_dict = {'Average': vt.Average, 'Base': vt.Base, 'Box': vt.Box, 'Scale': vt.Scale,
                   'idx': [i for i in range(1, 10)], 'pressure': 0, 'f0': 120, 'closed': 0, 'closure_time': 0,
                   'closure_position': 0, 'opening_time': 0}
        self.vt = vt_dict

        fmfit = diva_utils.read_file_parameter_alternate(filename, 'fmfit')
        fmfit = fmfit[0, 0]
        fmfit_dict = {'mu': fmfit.mu, 'iSigma': fmfit.iSigma, 'p': fmfit.p, 'beta_fmt': fmfit.beta_fmt,
                      'beta_som': fmfit.beta_som}
        self.fmfit = fmfit_dict

    def diva_vocaltract(self, out_type, x_data):
        if self.vt is None:
            self.diva_vt()

        out_type = out_type.lower()

        if out_type == 'auditory':
            y, z = self.vocal_tract_compute(x_data, None)
            return y

        if out_type == 'somatosensory':
            y, z = self.vocal_tract_compute(x_data, True)
            return z

        if out_type == 'auditory&somatosensory':
            y, z = self.vocal_tract_compute(x_data, True)
            return torch.cat((y, z), 0)

        if out_type == 'output':
            return self.vocal_tract_compute(x_data, False)

        if out_type == 'base':
            return torch.eye(self.params.Input.Dimensions, dtype=torch.float64)

    """
        Computes auditory/somatosensory representations
        Art(1:10) vocaltract shape params
        Art(11:13) F0/P/V params
        Aud(1:4) F0-F3 pitch&formants
        Som(1:6) place of articulation (~ from pharyngeal to labial closure)
        Som(7:8) P/V params (pressure,voicing)
    """

    # Returns Aud, Som, Outline, af, d]
    def diva_synth_sample(self, art, nargout):
        if self.vt is None:
            self.diva_vt()

        if art.size(0) == 0:
            return None, None, None, None, None

        if len(art.size()) == 1:
            art = art.unsqueeze(1)

        d = None
        af = None

        # Compute vocal tract configuration
        cuda0 = torch.device('cuda:0')
        idx = torch.tensor([i for i in range(10)])
        scale_tens = torch.from_numpy(self.vt['Scale']).to(torch.float64)
        scale_tens = scale_tens[:10]
        x = scale_tens * art[:10]
        avg_tens = torch.from_numpy(self.vt['Average']).to(torch.complex64)
        base_tens = torch.from_numpy(self.vt['Base']).to(torch.complex64)
        base_tens = torch.narrow(base_tens, 1, 0, 10)
        base_tens = base_tens.to(torch.complex64).cuda()
        #base_tens = torch.tensor(base_tens, dtype=torch.complex64, device=cuda0)
        x = x.to(torch.complex64).cuda()
        #x = torch.tensor(x, dtype=torch.complex64, device=cuda0)
        avg_tens = avg_tens.to(torch.complex64).cuda()
        #avg_tens = torch.tensor(avg_tens, dtype=torch.complex64, device=cuda0)
        outline = torch.matmul(base_tens, x)
        outline = avg_tens + outline

        # Compute somatosensory output (explicitly from vocal tract configuration)
        som = torch.zeros(8, 1, dtype=torch.float64)
        if nargout > 3:
            a, b, sc, af, d = self.xy2ab(outline, None)
            som[0:6] = np.maximum([-1], np.minimum([1], -1 * np.tanh(sc)))
            som[7:len(som)] = art[len(art) - 1:len(art)]

        # Compute auditory/somatosensory output (through previously computed forward fit)
        aud = torch.zeros(4, 1, dtype=torch.float64)
        if self.fmfit is not None:
            art_idx = art[:10]
            if len(art_idx.size()) == 1:
                art_idx = art_idx.unsqueeze(1)

            art_transp = torch.transpose(art_idx, 0, 1)

            aud[0] = 100 + 50 * art[11, 0]
            mu = torch.from_numpy(self.fmfit['mu']).to(torch.float64)
            dx = torch.sub(art_transp, mu)

            i_sigma = torch.from_numpy(self.fmfit['iSigma']).to(torch.float64)
            sig_adj = torch.matmul(dx, i_sigma)
            p = (-1 * torch.sum(sig_adj * dx, 1)) / 2
            p = p.unsqueeze(1)

            p_minus = torch.sub(p, torch.max(p))
            p_exp = torch.exp(p_minus)
            fmfit_p = torch.tensor(self.fmfit['p']).to(torch.float64)

            p = fmfit_p * p_exp
            p = p / torch.sum(p)
            art_arg = torch.cat((art_transp, torch.tensor([[1]])), 1)
            px = p * art_arg
            beta_fmt = torch.tensor(self.fmfit['beta_fmt']).to(torch.float64)

            # Convert to a column vector
            px_col = px.t().flatten()
            px_col = px_col.unsqueeze(1)

            aud_out = torch.matmul(beta_fmt, px_col)
            aud[1] = aud_out[0].item()
            aud[2] = aud_out[1].item()
            aud[3] = aud_out[2].item()

            aud_list = aud.tolist()

            if 1 < nargout <= 3:
                beta_som = torch.tensor(self.fmfit['beta_som']).to(torch.float64)
                som_out = torch.matmul(beta_som, px_col)
                for i in range(6):
                    som[i] = som_out[i].item()  # *px
                som[6] = art[11, 0]
                som[7] = art[12, 0]

        return aud, som, outline, af, d

    # outputs soundwave associated with sequence of articulatory states
    def diva_synth_sound(self, art):

        fixed_sample_size = False
        debug_no_rand = False

        if debug_no_rand:
            test_randn = diva_utils.read_file_parameter('rand_exports.mat', 'test_randn')
            test_randn_multi = diva_utils.read_file_parameter('rand_exports.mat', 'test_randn_multi')
            test_rand_single = diva_utils.read_file_parameter('rand_exports.mat', 'test_rand_single')

        if self.vt is None:
            self.diva_vt()

        synth = Synth()
        voices = {'F0': (120, 340), 'size': (1, 0.7)}
        opt_voices = 0

        ndata = art.size()[1]
        dt = 0.005
        s = torch.zeros(math.ceil((ndata + 1) * dt * synth.fs), dtype=torch.float64)
        s = s.unsqueeze(1)
        time = 0
        upper_lim = (ndata + 1) * dt
        min_v_sum = np.inf
        max_v_sum = -np.inf

        while time < upper_lim:
            # sample articulatory parameters

            if synth.samplesoutput >= 1104:
                nahMon = 43

            if synth.samplesoutput >= 1196:
                jahMon = 33

            t0 = math.floor(time / dt)
            t1 = (time - t0 * dt) / dt
            range_arg = min(ndata - 1, t0)
            range_arg2 = min(ndata - 1, 1 + t0)
            arg_pre_move = art[:, min(ndata - 1, t0)]
            arg_pre_move = arg_pre_move.unsqueeze(1)
            arg_post_move = art[:, min(ndata - 1, 1 + t0)]
            arg_post_move = arg_post_move.unsqueeze(1)
            oo, pp, ww, af1, d = self.diva_synth_sample(arg_pre_move, 4)
            oo, pp, ww, af2, d = self.diva_synth_sample(arg_post_move, 4)
            if af1 is not None and af2 is not None:
                naf1 = len(af1)
                naf2 = len(af2)

                if naf2 < naf1:
                    app_tens = torch.tensor([af2[af2.size()[0] - 1] for i in range(naf1 - naf2)])
                    app_tens = app_tens.unsqueeze(1)
                    af2 = torch.cat((af2, app_tens), 0)

                if naf1 < naf2:
                    app_tens = torch.tensor([af1[af1.size()[0] - 1] for i in range(naf2 - naf1)])
                    app_tens = app_tens.unsqueeze(1)
                    af1 = torch.cat((af1, app_tens), 0)

                af = af1 * (1 - t1) + af2 * t1

                fpv_max_arg_1 = [-1]
                fpv_min_arg_1 = [1]

                fpv_prod_arg_1 = art[10:13, min(ndata - 1, t0)] * (1 - t1)
                fpv_prod_arg_2 = np.multiply([t1], art[10:13, min(ndata - 1, 1 + t0)])

                fpv_min_arg_2 = np.add(fpv_prod_arg_1, fpv_prod_arg_2)
                fpv_max_arg_2 = np.minimum(fpv_min_arg_1, fpv_min_arg_2)

                fpv = np.maximum(fpv_max_arg_1, fpv_max_arg_2)

                self.vt['voicing'] = (1 + math.tanh(3 * fpv[2])) / 2
                self.vt['pressure'] = fpv[1]
                self.vt['pressure0'] = self.vt['pressure'] > 0.01
                self.vt['f0'] = 100 + 20 * fpv[0]

                af0 = np.maximum([0], af)
                k = 0.025
                for idx, val in enumerate(af0):
                    if 0 < val < k:
                        af0[idx] = k

                minaf = af.amin(0)
                minaf0 = af0.min(0)[0].item()
                self.vt['af'] = af

                # tracks place of articulation
                if minaf0 == 0:
                    release = 0
                    self.vt['opening_time'] = 0
                    self.vt['closure_time'] = self.vt['closure_time'] + 1

                    af0_rev = af0.flatten().tolist()
                    af0_rev.reverse()
                    index = af0_rev.index(0.0)
                    pos = len(af0) - index - 1

                    self.vt['closure_position'] = pos

                    if not self.vt['closed']:
                        closure = self.vt['closure_position']
                    else:
                        closure = 0

                    self.vt['closed'] = 1
                else:
                    if self.vt['closed']:
                        release = self.vt['closure_position']
                        release_closure_time = self.vt['closure_time']
                    else:
                        release = 0

                    if self.vt['pressure0'] and not synth.pressure0:
                        self.vt['opening_time'] = 0

                    self.vt['opening_time'] = self.vt['opening_time'] + 1
                    self.vt['closure_time'] = 0

                    self.vt['closure_position'] = af.min(0).indices[0].item()

                    closure = 0
                    self.vt['closed'] = 0

                if release > 0:
                    af = np.maximum([k], af)
                    minaf = max(k, minaf)
                    minaf0 = max(k, minaf0)
                    if debug_no_rand:
                        rand = test_rand_single
                        self.vt['f0'] = (0.95 + 0.1 * rand) * voices['F0'][opt_voices]
                    else:
                        rand = np.random.uniform()
                        self.vt['f0'] = (0.95 + 0.1 * rand) * voices['F0'][opt_voices]
                    synth.pressure = 0
                else:
                    if self.vt['pressure0'] and not synth.pressure0:
                        if debug_no_rand:
                            rand = test_rand_single
                            self.vt['f0'] = (0.95 + 0.1 * rand) * voices['F0'][opt_voices]
                        else:
                            rand = np.random.uniform()
                            self.vt['f0'] = (0.95 + 0.1 * rand) * voices['F0'][opt_voices]
                        synth.pressure = self.vt['pressure']
                        synth.f0 = 1.25 * self.vt['f0']
                        synth.pressure = 1
                    else:
                        if not self.vt['pressure0'] and synth.pressure0 and not self.vt['closed']:
                            synth.pressure = synth.pressure / 10

                # compute glottal source
                if fixed_sample_size:
                    synth.samplesperperiod = 92
                else:
                    synth.samplesperperiod = math.ceil(synth.fs / synth.f0)

                pp = [0.6, 0.2 - 0.1 * synth.voicing, 0.25]

                stepsize = float(1 / synth.samplesperperiod)
                g = [0 + (i * stepsize) for i in range(synth.samplesperperiod)]
                g = torch.tensor(g, dtype=torch.float64)
                g = g.unsqueeze(1)
                synth.glottalsource = 10 * 0.25 * glotlf(0, g, pp) + 10 * 0.025 * synth.k1 * glotlf(1, g, pp)
                numberofperiods = synth.numberofperiods

                # compute vocal tract filter
                if synth.samplesperperiod < 0:
                    broken = True
                [synth.filt, synth.f, synth.filt_closure] = \
                    a2h(af0, d, synth.samplesperperiod, synth.fs, self.vt['closure_position'], minaf0)
                eps = 2 ** (-52)
                synth.filt = 2 * synth.filt / np.maximum(eps, synth.filt[0])
                synth.filt[0] = 0
                synth.filt_closure = 2 * synth.filt_closure / np.maximum(eps, synth.filt_closure[0])
                synth.filt_closure[0] = 0

                # compute sound signal

                w = torch.linspace(0, 1, synth.samplesperperiod)
                w = w.unsqueeze(1)
                if release > 0:
                    # test_randn_again = np.random.normal(size=(synth.samplesperperiod, 1))
                    if debug_no_rand:
                        u = synth.voicing * 1 * 0.010 * \
                            (synth.pressure + 20 * synth.pressurebuildup) * synth.glottalsource + (1 - synth.voicing) * \
                            1 * 0.010 * (synth.pressure + 20 * synth.pressurebuildup) * \
                            test_randn_multi
                    else:
                        u = synth.voicing * 1 * 0.010 * \
                            (synth.pressure + 20 * synth.pressurebuildup) * synth.glottalsource + (1 - synth.voicing) * \
                            1 * 0.010 * (synth.pressure + 20 * synth.pressurebuildup) * \
                            np.random.normal(size=(synth.samplesperperiod, 1))
                    v0 = torch.real(torch.fft.ifft2(torch.fft.fft2(u) * synth.filt_closure))
                    numberofperiods = numberofperiods - 1
                    synth.pressure = synth.pressure / 10
                    vnew = v0[:synth.samplesperperiod]
                    t_numel = synth.sample.size()[0]
                    test_size = (t_numel - 1) * (
                        torch.tensor([i for i in range(synth.samplesperperiod)], dtype=torch.long))
                    test_size = torch.ceil(test_size / synth.samplesperperiod)
                    test_size = test_size.to(torch.long)
                    v0 = (1 - w) * synth.sample[test_size] + w * vnew
                    synth.sample = vnew
                else:
                    v0 = torch.tensor([], dtype=torch.float64)

                if numberofperiods > 0:
                    # vocal tract filter
                    # scal_factor = 0.25 * synth.pressure * synth.glottalsource
                    # arg_2 = (1 + np.multiply([0.1], np.random.normal(size=(synth.samplesperperiod, 1))))
                    # u = np.multiply(scal_factor, arg_2)
                    # u = synth.voicing * u + (1 - synth.voicing) * 0.025 * synth.pressure * \
                    #    np.random.normal(size=(synth.samplesperperiod, 1))

                    if debug_no_rand:
                        u = 0.25 * synth.pressure * synth.glottalsource * \
                            (1 + 0.1 * test_randn_multi)
                        u = synth.voicing * u + (1 - synth.voicing) * 0.025 * synth.pressure * \
                            test_randn_multi
                    else:
                        u = 0.25 * synth.pressure * synth.glottalsource * \
                            (1 + np.multiply([0.1], np.random.normal(size=(synth.samplesperperiod, 1))))
                        u = synth.voicing * u + (1 - synth.voicing) * 0.025 * synth.pressure * \
                            np.random.normal(size=(synth.samplesperperiod, 1))

                    if 0 < minaf0 <= k:
                        u = minaf / k * u + (1 - minaf / k) * 0.02 * synth.pressure * \
                            np.random.normal(size=(synth.samplesperperiod, 1))

                    innermost = torch.fft.fft2(u) * synth.filt
                    ifft_res = torch.fft.ifft2(innermost)
                    real_res = torch.real(ifft_res)
                    v = torch.real(torch.fft.ifft2(torch.fft.fft2(u) * synth.filt))

                    vnew = v[:synth.samplesperperiod]

                    t_numel = synth.sample.size()[0]
                    test_size = (t_numel - 1) * (
                        torch.tensor([i for i in range(synth.samplesperperiod)], dtype=torch.long))
                    test_size = torch.ceil(test_size / synth.samplesperperiod)
                    test_size = test_size.to(torch.long)
                    v = (1 - w) * synth.sample[test_size] + w * vnew

                    synth.sample = vnew
                else:
                    v = torch.tensor([])

                #v0 = torch.tensor(v0, dtype=torch.float64)
                v = torch.cat((v0, v), 0)

                randn_v = torch.randn(v.size()[0])
                randn_v = randn_v.unsqueeze(1)

                if debug_no_rand:
                    v = v + np.multiply([0.0001], test_randn_multi)
                else:
                    v = v + np.multiply([0.0001], randn_v)

                v = torch.divide((1 - torch.exp(-1 * v)), (1 + torch.exp(-1 * v)))
                vsz_0 = v.size()[0]
                start_out_idx = synth.samplesoutput
                end_out_idx = synth.samplesoutput + vsz_0
                if end_out_idx > s.size()[0]:
                    end_out_idx = s.size()[0] - 1
                # put the slice v of samples into the output set s
                select_size = end_out_idx - start_out_idx

                if end_out_idx >= 1197:
                    jahMon = 332

                v_sum = v.sum()
                if v_sum > max_v_sum:
                    max_v_sum = v_sum

                if v_sum < min_v_sum:
                    min_v_sum = v_sum

                if v_sum > 0:
                    sample_greater = True

                s[start_out_idx:end_out_idx, :] = v[:select_size]
                time = time + v.size()[0] / synth.fs
                synth.samplesoutput = synth.samplesoutput + v.size()[0]

                # computes f0/amp/voicing/pressurebuildup modulation
                synth.pressure0 = self.vt['pressure0']
                alpha = min(1, 0.1 * synth.numberofperiods)
                beta = 100 / synth.numberofperiods
                # test_randn = np.random.normal()

                synth.pressure = synth.pressure + alpha * (
                        self.vt['pressure'] * (max(1, 1.5 - self.vt['opening_time'] / beta)) - synth.pressure)
                alpha = min(1, 0.5 * synth.numberofperiods)
                rand = (np.random.normal())
                if debug_no_rand:
                    synth.f0 = synth.f0 + 2 * math.sqrt(alpha) * test_randn + alpha * (
                            self.vt['f0'] * max(1, 1.25 - self.vt['opening_time'] / beta) - synth.f0)
                else:
                    synth.f0 = synth.f0 + 2 * math.sqrt(alpha) * rand + alpha * (
                            self.vt['f0'] * max(1, 1.25 - self.vt['opening_time'] / beta) - synth.f0)

                synth.voicing = max(0, min(1, synth.voicing + 0.5 * (self.vt['voicing'] - synth.voicing)))
                alpha = min(1, 0.1 * synth.numberofperiods)
                synth.pressurebuildup = max(0, min(1, synth.pressurebuildup + alpha * (
                        2 * (self.vt['pressure'] > 0 and minaf < 0))))
                synth.numberofperiods = max(1, numberofperiods)
            else:
                time += 1
        return s

    # Returns Aud,Som,Outline,af,filt
    def diva_synth(self, art, option):
        ndata = art.size()[1]
        if option is None:
            if ndata > 1:
                option = 'sound'
            else:
                option = 'audsom'
        m = torch.nn.Tanh()
        art = m(art)
        aud = None
        filt = None
        # Output soundwave associated with sequence of articulatory states
        if option == 'sound':
            aud = self.diva_synth_sound(art)
            som = None
            outline = None
            af = None
            filt = None
        # Output auditory / somatosensory representation associated with a given articulatory state
        if option == 'audsom':
            if ndata > 1:
                aud_out = []
                som_out = []
                outl_out = []
                af_out = []
                d_out = []
                cuda0 = torch.device('cuda:0')
                aud_out = torch.tensor([], dtype=torch.float64)
                som_out = torch.tensor([], dtype=torch.float64)
                outl_out = torch.tensor([], dtype=torch.float64, device=cuda0)
                af_out = torch.tensor([], dtype=torch.float64)
                for i in range(ndata):
                    if i == 12:
                        jahMon = 22
                    aud_new, som_new, outline_new, af_new, d_new = self.diva_synth_sample(art[:, i], 2)
                    aud_out = torch.cat((aud_out, aud_new), 1)
                    som_out = torch.cat((som_out, som_new), 1)
                    outl_out = torch.cat((outl_out, outline_new), 1)
                return aud_out, som_out, outl_out, af_out, filt
            else:
                aud, som, outline, af, d = self.diva_synth_sample(art, 2)
        return aud, som, outline, af, filt

    def vocal_tract_compute(self, x, audsom):
        if len(x.size()) == 1:
            x = x.unsqueeze(1)
        in_scale = torch.tensor(self.params.Input.Scale, dtype=torch.float64)
        x = np.multiply(x, in_scale)
        if audsom:
            aud, som, outline, af, filt = self.diva_synth(x, 'audsom')
        else:
            aud, som, outline, af, filt = self.diva_synth(x, None)
        y = aud
        z = som
        out_scale = torch.tensor(self.params.Output.AuditoryArgs.Scale, dtype=torch.float64)
        y = y / out_scale
        y = torch.diagonal(y)
        y = y.unsqueeze(1)

        # adjust for minor error

        y[1] += 0.00000000000000014
        y[2] += 0.0000000000000005

        out_scale_som = torch.tensor(self.params.Output.SomatosensoryArgs.Scale, dtype=torch.float64)

        return y, z

    def dosound(self, x, display):
        if display is None:
            display = False

        in_scale = torch.tensor(self.params.Input.Scale, dtype=torch.float64)
        x = np.multiply(x, in_scale)
        aud, som, outline, af, filt = self.diva_synth(x, 'sound')
        y = aud
        z = som
        out_scale = torch.tensor(self.params.Output.AuditoryArgs.Scale, dtype=torch.float64)

        if display:
            plt.figure(figsize=(14, 5))
            librosa.display.waveplot(y.flatten().numpy(), sr=11025)

            X = librosa.stft(y.flatten().numpy())
            Xdb = librosa.amplitude_to_db(abs(X))
            plt.figure(figsize=(14, 5))
            librosa.display.specshow(Xdb, sr=11025, x_axis='time', y_axis='hz')
            plt.colorbar()

        t = time.time()
        t_stamp = int(t)
        file_name = str(t_stamp) + '_DIVA_OUT.wav'

        y_max = torch.max(y)
        y_min = torch.min(y)
        aud = y.flatten().numpy()
        aud = (aud * (2 ** 15 - 1)).astype("<h")
        write(file_name, 11025, aud)

        self.last_prod = y


        sd.play(y.flatten(), 11025)
        sd.wait()

    def PlayLast(self):
        if self.last_prod is not None:
            sd.play(self.last_prod.flatten(), 11025)
            sd.wait()

    def reset(self):
        self.OutputPorts[0] = torch.zeros(self.OutputPortDimensions[0], dtype=torch.float64)
        self.OutputPorts[1] = torch.zeros(self.OutputPortDimensions[1], dtype=torch.float64)

    def output(self):
        self.t = self.t + 1
        input1 = self.InputPorts[0]
        rem = ~(self.t % 2)
        y, z = self.vocal_tract_compute(input1, rem)
        self.OutputPorts[0] = y
        if self.OutputPortDimensions[0] != y.size():
            self.OutputPortDimensions[0] = y.size()
        self.OutputPorts[1] = z
        if self.OutputPortDimensions[1] != z.size():
            self.OutputPortDimensions[1] = z.size()
