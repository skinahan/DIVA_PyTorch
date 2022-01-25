import diva_errormap as dem


class AuditoryErrorMap(dem.ErrorMap):

    def __init__(self):
        dem.ErrorMap.__init__(self, 'diva_weights_SSM2amax.mat', 'diva_weights_SSM2amin.mat')
        self.target_max_gain = 1
        self.target_min_gain = 1
        self.error_gain = 1
