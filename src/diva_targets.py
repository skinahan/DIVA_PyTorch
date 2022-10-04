import numpy as np
from scipy.interpolate import interp1d
import math
import diva_utils
import vocaltract_args as vta


class diva_targets:

    def __init__(self):
        self.filename = 'diva.csv'
        self.production_info = {}

    def list(self):
        [production_ids, production_labels] = diva_utils.diva_targets_readcsvfile(self.filename)
        return production_ids, production_labels

    def delete(self, production):
        [production_ids, production_labels] = diva_utils.diva_targets_readcsvfile(self.filename)
        idx = production_labels.index(production)

    def init_struct(self, nrandom=0):
        production_info = self.production_info
        production_info["name"] = ''
        production_info['length'] = 500
        production_info['wrapper'] = 0
        production_info['interpolation'] = 'spline'
        params = vta.VocalTractArgs()
        aud_args = params.Output.AuditoryArgs
        som_args = params.Output.SomatosensoryArgs
        all_args = [aud_args, som_args]
        ctr = 0
        for arg in all_args:
            ctr += 1
            for n1 in range(len(arg.Plots_dim)):
                if len(arg.Plots_dim[n1]) == 1:
                    idx = arg.Plots_dim[n1][0]
                    if arg.DefaultSound is not None:
                        Default = arg.DefaultSound
                    else:
                        Default = arg.Range
                    if n1 >= len(arg.Plots_label):
                        broke = True
                    argLabel = arg.Plots_label[n1]
                    control_label = argLabel + '_control'
                    min_label = argLabel + '_min'
                    max_label = argLabel + '_max'
                    if ctr == 1 and nrandom > 0:
                        production_info[control_label] = np.linspace(0, production_info['length'], nrandom)
                        x = np.sort(np.random.rand(2, nrandom))
                        production_info[min_label] = Default[idx][0] * (1 - x[1, :]) + Default[idx][1] * x[1, :]
                        production_info[max_label] = Default[idx][0] * (1 - x[2, :]) + Default[idx][1] * x[2, :]
                    else:
                        production_info[control_label] = [0]
                        production_info[min_label] = Default[idx][0]
                        production_info[max_label] = Default[idx][1]
        self.production_info = production_info
        return production_info

    def interpolate(self, x0, y0, x, interpolation):
        x_shape = np.shape(x0)
        y_shape = np.shape(y0)
        if len(x_shape) == 1:
            x0 = np.expand_dims(x0, 1)
        # if len(y_shape) == 1:
        #     y0 = np.expand_dims(y0, 1)
        x0 = x0[:]
        y0 = y0[:]
        x = x[:]
        if len(x0) != len(y0):
            y = math.nan(len(x))
            return y
        y0 = np.array(y0)
        idx = np.argsort(x0)
        x0 = np.sort(x0)
        y0 = y0[idx]
        end = len(x0) - 1
        idx = np.nonzero(x0[1:end] == x0[0:end - 1])
        x0[idx] = x0[idx] + np.spacing(1)
        if np.min(x0) > 0:
            x0 = np.vstack((0, x0))
            y0 = np.vstack((y0[0], y0))
        if np.max(x0) < np.max(x):
            x0 = np.vstack((x0, np.max(x)))
            y0 = np.vstack((y0, y0[end]))
        if len(x0) <= 2:
            interpolation = 'linear'
        if len(x0) <= 1:
            interpolation = 'nearest'
        # y = interp1d(x0, y0, x, kind=interpolation)
        y = np.interp(x.flatten(), x0.flatten(), y0.flatten())
        y = np.expand_dims(y, axis=1)
        return y

    def timeseries_convert(self, production_info, params_info):
        DT = 5
        length = int(production_info['length'])
        wrapper = int(production_info['wrapper'])
        Nt = int(1 + np.ceil((length + (2 * wrapper)) / DT))
        Time = np.array(np.arange(0, Nt)).T * DT
        Time = np.expand_dims(Time, axis=1)
        y_min = np.zeros([Nt, params_info.Dimensions])
        y_max = np.zeros([Nt, params_info.Dimensions])
        for n in range(len(params_info.Plots_dim)):
            if len(params_info.Plots_dim[n]) == 1:
                idx = params_info.Plots_dim[n]
                control_label = params_info.Plots_label[n] + '_control'
                min_label = params_info.Plots_label[n] + '_min'
                max_label = params_info.Plots_label[n] + '_max'
                base_x0 = production_info[control_label]
                if isinstance(base_x0, list):
                    base_x0 = int(base_x0[0])
                else:
                    if isinstance(base_x0, str):
                        base_x0 = [float(i) for i in base_x0.split()]
                x0 = base_x0  # + wrapper

                if isinstance(production_info[min_label], str):
                    x1 = [float(i) for i in production_info[min_label].split()]
                # x1 = float(production_info[min_label])
                if isinstance(production_info[max_label], str):
                    x2 = [float(i) for i in production_info[max_label].split()]
                # x2 = float(production_info[max_label])

                if isinstance(x0, int) or isinstance(x0, float):
                    x0 = [x0]
                if isinstance(x1, int) or isinstance(x1, float):
                    x1 = [x1]
                if isinstance(x2, int) or isinstance(x2, float):
                    x2 = [x2]

                y_min[:, idx] = self.interpolate(x0, x1, Time, production_info['interpolation'])
                y_max[:, idx] = self.interpolate(x0, x2, Time, production_info['interpolation'])
        if wrapper > 0:
            mask = Time < wrapper or Time > (wrapper + length)
            y0_min = params_info.DefaultSilence[:, 0]
            y0_max = params_info.DefaultSilence[:, 1]
            y_min[mask, :] = np.tile(y0_min.H, (np.count_nonzero(mask), 1))
            y_max[mask, :] = np.tile(y0_max.H, (np.count_nonzero(mask), 1))
        temp = np.maximum(y_min, y_max)
        y_min = np.minimum(y_min, y_max)
        y_max = temp
        return y_min, y_max, Time

    def timeseries(self, production_info, doheader=False):

        if isinstance(production_info, list):
            temp_prod_info = production_info
            new_prod_info = {}
            for idx, label in enumerate(production_info[0]):
                new_prod_info[label.rstrip("\n")] = production_info[1][idx]
            production_info = new_prod_info
        params = vta.VocalTractArgs()
        [Aud_min, Aud_max, Time] = self.timeseries_convert(production_info, params.Output.AuditoryArgs)
        [Som_min, Som_max, Time] = self.timeseries_convert(production_info, params.Output.SomatosensoryArgs)
        Art = np.tile(np.mean(params.Input.Default, 2).T, (Aud_min.shape[0], 1))
        N_samplesperheader = 0
        if doheader and N_samplesperheader:
            # Disabled/Unreachable code area in Matlab source
            foo = 32

        series = {'Aud_min': Aud_min, 'Aud_max': Aud_max, 'Som_min': Som_min, 'Som_max':
            Som_max, 'Art': Art, 'time': Time}
        return series

    def new(self, ftype, prod_info=None):
        if prod_info is None:
            prod_info = self.init_struct()
        if ftype == 'txt':
            return prod_info
        if ftype == 'mat':
            return self.timeseries(prod_info, True)
