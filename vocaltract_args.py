import torch
import numpy as np


# creates a large matrix B
#  consisting of an M-by-N tiling of copies of A.

def repmat(m, n):
    b = [[m] for i in range(n[0])]
    return b


def bsxfun(fun, a, b):
    c = fun(a, b)
    return c


def eq(a, b):
    if a == b:
        return 1
    else:
        return 0


class InputArgs:

    def __init__(self):
        self.NArt = 13
        self.Name = 'Articulatory'
        self.Dimensions = self.NArt
        self.Range = repmat([-1, 1], [self.NArt, 1])  # [[-1, 1] for item in self.NArt]
        self.Scale = [[1] for i in range(self.NArt)]
        self.Default = repmat([-1, 1], [self.NArt, 1])
        self.DefaultSilence = repmat([-1, 1], [10, 1])
        self.DefaultSilence.append([-1, 1])
        self.DefaultSilence.append([-1, -0.5])
        self.DefaultSilence.append([-1, 1])
        self.DefaultSound = repmat([-1, 1], [10, 1])
        self.DefaultSound.append([-1, 1])
        self.DefaultSound.append([0, 1])
        self.DefaultSound.append([-1, 1])
        ej = [1 * 10, 2, 3, 4]
        tj = torch.tensor(ej)
        tj = torch.transpose(tj, 0, 0)
        ej = np.array(ej)
        tj = np.array(tj.tolist())
        self.BlockDiagonal = (ej == tj)
        self.Plots_dim = [[x for x in range(1, 10)], [x for x in range(1, 13)]]

        self.Plots_label = ['VocalTract']
        for x in range(1, 10):
            self.Plots_label.append('v[t' + str(x))
        self.Plots_label.append('tension')
        self.Plots_label.append('pressure')
        self.Plots_label.append('voicing')


class AuditoryArgs:

    def __init__(self):
        self.Name = 'Auditory'
        self.Dimensions = 4
        self.Range = [[0, 200],
                      [0, 1000],
                      [0, 3000],
                      [0, 4000]]
        self.Scale = [100, 500, 1500, 3000]
        self.Default = [[0, 200],
                        [0, 1000],
                        [0, 3000],
                        [0, 4000]]
        self.DefaultSilence = [[0, 200],
                               [0, 1000],
                               [0, 3000],
                               [0, 4000]]
        self.DefaultSound = [[0, 200],
                             [0, 1000],
                             [500, 3000],
                             [2000, 4000]]
        self.Plots_dim = [[2, 3, 4], 1, 2, 3, 4]
        self.Plots_label = [['Formants'], 'F0', 'F1', 'F2', 'F3']


class SomatosensoryArgs:
    def __init__(self):
        self.Name = 'Somatosensory'
        self.Dimensions = 8
        self.Range = repmat([-1, 1], [8, 1])
        self.Scale = [1 * 8]
        self.Default = repmat([-1, -0.25], [6, 1])
        self.Default.append([0.75, 1])
        self.Default.append([0.75, 1])
        self.DefaultSilence = repmat([-1, 1], [6, 1])
        self.DefaultSilence.append([-1, -0.5])
        self.DefaultSilence.append([-1, -1])
        self.DefaultSound = repmat([-1, -0.25], [6, 1])
        self.DefaultSound.append([0.75, 1])
        self.DefaultSound.append([0.75, 1])
        self.Plots_dim = [x for x in range(1, 6)]
        self.Plots_dim.append(7)
        self.Plots_dim.append(8)
        for x in range(1, 6):
            self.Plots_dim.append(x)
        self.Plots_label = [
            ['PlaceofArt', 'pressure', 'voicing', 'PA_pharyngeal', 'PA_uvular', 'PA_velar', 'PA_palatal',
             'PA_alveolardental', 'PA_labial']]


class OutputArgs:

    def __init__(self):
        self.AuditoryArgs = AuditoryArgs()
        self.SomatosensoryArgs = SomatosensoryArgs()


class VocalTractArgs:

    def __init__(self):
        self.dosound = 1
        self.Input = InputArgs()
        self.Output = OutputArgs()
