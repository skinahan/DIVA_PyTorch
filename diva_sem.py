import diva_errormap as dem


class SomatosensoryErrorMap(dem.ErrorMap):

    def __init__(self):
        dem.ErrorMap.__init__(self, 'diva_weights_SSM2smax.mat', 'diva_weights_SSM2smin.mat')
        self.target_max_gain = 1
        self.target_min_gain = 1
        self.error_gain = 1
