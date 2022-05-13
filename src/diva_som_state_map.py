import diva_delayblock as db


class SomatosensoryStateMap(db.DelayBlock):

    def __init__(self):
        db.DelayBlock.__init__(self, 4)
        self.set_inputs(1)
        self.set_outputs(1)
        self.scalar_gain = 1

