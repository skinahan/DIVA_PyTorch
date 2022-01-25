import time


class Simulation:

    def __init__(self):
        # global sample rate in milliseconds
        self.sample_rate_ms = 0.005
        # global step counter for the simulation
        self.sim_step = 0
        # global step count limit for the simulation
        self.step_limit = 101
        self.setup_diva_sim()
        self.sig_blocks = []

    def setup_diva_sim(self):
        print("Initializing DIVA...")

    def run(self):
        print("Beginning Simulation...")
        for step in range(0, self.step_limit):
            print("Step: " + str(step))

            for block in self.sig_blocks:
                block.output()

            time.sleep(self.sample_rate_ms)
