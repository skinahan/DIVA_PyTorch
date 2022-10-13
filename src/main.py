import sys
import os
import time
import torch
import diva_targets
import diva_utils
import numpy as np

import diva_ssm
import diva_cerebellum as ceb
import diva_motorcortex as dmc
import diva_vocaltract as dvt
import diva_asm as aud_sm
import diva_som_state_map as som_sm
import diva_aem as aud_em
import diva_sem as som_em


class DIVA:

    def __init__(self):
        self.simCtr = 0
        self.targetCSV = "diva.csv"
        self.targetIndex = 0
        # the simulation fundamental sample time is 5 ms, 200Hz
        self.sample_rate = 5
        self.sim_speed_scale = 1  # scaling factor for simulation time
        # Simulink implementation of DIVA is a discrete-time process.
        [production_ids, production_labels] = diva_utils.diva_targets_readcsvfile(self.targetCSV)
        self.production_ids = production_ids
        self.production_labels = production_labels
        self.production_labels.append("new@default")
        self.production_labels.append("new@random")
        self.production_idx = 2
        self.filetype = "mat"
        self.ssm = None
        self.cerebellum = None
        self.motor_cortex = None
        self.vocal_tract = None
        self.aud_state_map = None
        self.som_state_map = None
        self.aud_err_map = None
        self.som_err_map = None
        self.production_info = None
        self.production_art = None
        self.production_log = {}
        self.ChangeProduction('happy')
        self.PrepareSimulation()
        self.diva_targets = diva_targets.diva_targets()
        torch.manual_seed(0)
        np.random.seed(0)

    def production(self):
        return self.production_labels[self.production_idx]

    def reset(self):
        if self.ssm is not None:
            self.ssm.reset()
        if self.cerebellum is not None:
            self.cerebellum.reset()
        if self.motor_cortex is not None:
            self.motor_cortex.reset()
        if self.vocal_tract is not None:
            self.vocal_tract.reset()
        if self.aud_state_map is not None:
            self.aud_state_map.reset()
        if self.som_state_map is not None:
            self.som_state_map.reset()
        if self.aud_err_map is not None:
            self.aud_err_map.reset()
        if self.som_err_map is not None:
            self.som_err_map.reset()

    def LoadTarget(self, filetype, id):
        if filetype == "txt":
            filename = "diva_00000" + str(id) + ".txt"
            return diva_utils.txt2struct(filename)
        if filetype == "mat":
            filename = "diva_00000" + str(id) + ".mat"
            if not os.path.exists(filename):
                filename_txt = "diva_00000" + str(id) + ".txt"
                prod_info = diva_utils.txt2struct(filename_txt)
            else:
                return diva_utils.read_file_parameter_alternate(filename, 'timeseries')

    def diva_vocaltract(self):
        filename = "diva_vocaltract.mat"
        return diva_utils.read_file_parameter_alternate(filename, 'params')

    def PrepareSimulation(self):
        n_productions = 1

        if self.production_art is None:
            self.production_art = self.diva_targets.timeseries(self.production_info, doheader=True)

        # loading the nested structs from matlab!
        n_samplesperproduction = len(self.production_art['Art'])

        params = self.diva_vocaltract()
        params_obj = params[0, 0]
        params_output = params_obj.Output
        # Auditory Params
        params_output_1_obj = params_output[0, 0]
        auditory_scale = params_output_1_obj.Scale.astype(np.float)
        auditory_scale = torch.from_numpy(auditory_scale)

        # Somatosensory Params
        params_output_2_obj = params_output[0, 1]
        somato_scale = params_output_2_obj.Scale.astype(np.float)
        somato_scale = torch.from_numpy(somato_scale)

        diag_aud = (1.0 / auditory_scale).diagflat()
        diag_som = (1.0 / somato_scale).diagflat()

        Target_production = [0, 1]

        # define simulation weight matrices

        W = [1] * n_samplesperproduction
        wdict = {}
        wdict['W'] = W
        diva_utils.write_file("diva_weights_SSM.mat", wdict)
        # save diva_weights_SSM.mat W

        W = self.production_art['Art']
        wdict['W'] = W
        diva_utils.write_file("diva_weights_SSM2FF.mat", wdict)
        # save diva_weights_SSM2FF.mat W

        range_aud = params_output_1_obj.Range
        range_aud_col1 = range_aud[:, :1]
        range_aud_col2 = range_aud[:, 1:2]

        aud_min_tensor = torch.from_numpy(self.production_art['Aud_min'].astype(np.float))
        W = torch.matmul(aud_min_tensor, diag_aud)
        W0 = range_aud_col1 / auditory_scale
        wdict['W'] = W.numpy()
        wdict['W0'] = W0.numpy()
        diva_utils.write_file("diva_weights_SSM2amin.mat", wdict)
        # save diva_weights_SSM2amin.mat W W0

        aud_max_tensor = torch.from_numpy(self.production_art['Aud_max'].astype(np.float))
        W = torch.matmul(aud_max_tensor, diag_aud)
        W0 = range_aud_col2 / auditory_scale
        wdict['W'] = W.numpy()
        wdict['W0'] = W0.numpy()
        diva_utils.write_file("diva_weights_SSM2amax.mat", wdict)
        # save diva_weights_SSM2amax.mat W W0

        range_som = params_output_2_obj.Range
        range_som_col1 = range_som[:, :1]
        range_som_col2 = range_som[:, 1:2]

        som_min_tensor = torch.from_numpy(self.production_art['Som_min'].astype(np.float))
        W = torch.matmul(som_min_tensor, diag_som)
        W0 = range_som_col1 / somato_scale
        wdict['W'] = W.numpy()
        wdict['W0'] = W0.numpy()
        diva_utils.write_file("diva_weights_SSM2smin.mat", wdict)
        # save diva_weights_SSM2smin.mat W W0

        som_max_tensor = torch.from_numpy(self.production_art['Som_max'].astype(np.float))
        W = torch.matmul(som_max_tensor, diag_som)
        W0 = range_som_col2 / somato_scale
        wdict['W'] = W.numpy()
        wdict['W0'] = W0.numpy()
        diva_utils.write_file("diva_weights_SSM2smax.mat", wdict)
        self.reset()
        # save diva_weights_SSM2smax.mat W W0

    def PromptProduction(self):
        user_in = input("Enter the production label:\n")
        return self.ChangeProduction(user_in)

    def ChangeProductionIdx(self, idx):
        if idx >= 0:
            self.production_idx = idx
            id = self.production_ids[idx]
            self.production_info = None
            self.production_info = self.LoadTarget("txt", id)
            # print(self.production_info)
            self.production_art = None
            self.production_art = vars(self.LoadTarget("mat", id)[0,0])
            # print(self.production_art)
            print("Target production loaded successfully")
            self.targetIndex = idx
            if self.motor_cortex is not None:
                self.motor_cortex = None
            return True
        else:
            print("warning: no entry matching " + self.production() + " in " + self.targetCSV)
            return False

    def ChangeProduction(self, user_in):
        idx = self.production_labels.index(user_in)
        self.targetIndex = idx
        return self.ChangeProductionIdx(idx)

    def ShowMenu(self):
        print("= PyTorch DIVA Implementation Menu =")
        print("Active Production: " + self.production())
        print("Select from the following options:")
        print("l    :   list targets")
        print("n    :   new target")
        print("p    :   change production")
        print("r    :   reset target")
        print("s    :   save target")
        print("sim    :   start simulation")
        print("last    :    playback last output")
        print("log     :    save produced articulator movements")
        print("q    :   quit\n")

    def PlayLast(self):
        if self.vocal_tract is not None:
            self.vocal_tract.PlayLast()

    def StartSim(self, playsound, fname, save_file=False):
        sim_label = 'VT_' + str(self.simCtr)
        self.simCtr = self.simCtr + 1
        length_idx = self.production_info[0].index('length\n')
        total_prod_length = self.production_info[1][length_idx]
        num_steps = int(np.ceil(float(total_prod_length) / self.sample_rate)) + 1
        if self.ssm is None:
            self.ssm = diva_ssm.SpeechSoundMap(num_steps)
        ssm = self.ssm

        if self.cerebellum is None:
            self.cerebellum = ceb.Cerebellum(num_steps)
        cerebellum = self.cerebellum

        if self.vocal_tract is None:
            self.vocal_tract = dvt.VocalTract()
        vocal_tract = self.vocal_tract

        if self.motor_cortex is None:
            self.motor_cortex = dmc.MotorCortex(vocal_tract)
        motor_cortex = self.motor_cortex

        if self.aud_state_map is None:
            self.aud_state_map = aud_sm.AuditoryStateMap()
        aud_state_map = self.aud_state_map
        aud_state_map.setinputdims(4)

        if self.aud_err_map is None:
            self.aud_err_map = aud_em.AuditoryErrorMap()
        aud_err_map = self.aud_err_map

        if self.som_state_map is None:
            self.som_state_map = som_sm.SomatosensoryStateMap()
        som_state_map = self.som_state_map
        som_state_map.setinputdims(8)

        if self.som_err_map is None:
            self.som_err_map = som_em.SomatosensoryErrorMap()
        som_err_map = self.som_err_map

        target_production = [0, 1]
        psum = sum(target_production)
        lookup_table = [[0, 1], [0, 0]]

        is_zero = bool(sum == 0)

        if is_zero:
            idx1 = 1
        else:
            idx1 = 0

        idx2 = 1  # initial condition, unit delay block

        det_rise_from_zero = lookup_table[idx1][idx2]

        product1 = psum * det_rise_from_zero

        ssm.InputPorts[0] = torch.tensor([product1], dtype=torch.float64)
        ssm.InputPorts[1] = 1

        art_accum = None

        ssm.output()
        ssm.output_seq()

        for i in range(num_steps + 10):
            ssm_out_primary = ssm.OutputPorts[0]
            ssm_out_aud = ssm.OutputPorts[2]
            ssm_out_som = ssm.OutputPorts[3]

            # Advance to the next timestep / SSM signal
            ssm.output_seq()

            aud_err_map.InputPorts[0] = ssm_out_aud
            som_err_map.InputPorts[0] = ssm_out_som

            aud_state_map.output()
            som_state_map.output()

            aud_state = aud_state_map.OutputPorts[0]
            som_state = som_state_map.OutputPorts[0]

            aud_err_map.InputPorts[1] = aud_state
            som_err_map.InputPorts[1] = som_state

            aud_err_map.output()
            som_err_map.output()

            aud_err = aud_err_map.OutputPorts[0]
            som_err = som_err_map.OutputPorts[0]

            cerebellum.input(ssm_out_primary)
            cerebellum.output()
            ceb_out = cerebellum.OutputPorts[0]

            motor_cortex.InputPorts[0] = ssm_out_primary
            motor_cortex.InputPorts[1] = ceb_out
            # Somatosensory Error
            motor_cortex.InputPorts[2] = som_err
            # Auditory Error
            motor_cortex.InputPorts[3] = aud_err
            motor_cortex.output()

            cort_out = motor_cortex.OutputPorts[0]
            cort_transpose = torch.transpose(cort_out, 0, 1)

            vocal_tract.InputPorts[0] = cort_out
            vocal_tract.output()

            vt_aud_out = vocal_tract.OutputPorts[0]
            vt_som_out = vocal_tract.OutputPorts[1]

            aud_state_map.input(vt_aud_out)
            som_state_map.input(vt_som_out)

            # Track the sequence of articulatory movements
            if art_accum is None:
                art_accum = cort_out
            else:
                art_accum = torch.cat((art_accum, cort_out), 1)

        self.production_log[sim_label] = art_accum.numpy()
        if playsound or save_file:
            vocal_tract.dosound(art_accum, False,  fname, save_file)
        self.reset()

    def ListTargets(self):
        print("Target List: ")
        for target in self.production_labels:
            print(target)

    # Reset the weight of the forward mapping so it must be relearned over time
    def ResetTarget(self):
        self.production_art['Art'].fill(0.0)
        self.production_art = None
        self.ssm = None
        self.cerebellum = None
        self.motor_cortex = None
        self.aud_state_map = None
        self.som_state_map = None
        self.aud_err_map = None
        self.som_err_map = None
        self.PrepareSimulation()
        print("Target reset.")

    def ProcessInput(self, str_arg):
        if str_arg == "l":
            self.ListTargets()
            return
        if str_arg == "n":
            diva_utils.shownotimplemented()
            return
        if str_arg == "p":
            changed = self.PromptProduction()
            if changed:
                self.PrepareSimulation()
            return
        if str_arg == "r":
            self.ResetTarget()
            return
        if str_arg == "s":
            self.Save()
            return
        if str_arg == "sim":
            with torch.no_grad():
                self.StartSim(True, self.production_labels[self.targetIndex], True)
            return
        if str_arg == "last":
            self.PlayLast()
            return
        if str_arg == "test_runs":
            for i in range(20):
                self.StartSim(False, None, False)
            self.SaveLogNoPrompt()
            print("Exiting...")
            sys.exit(0)
        if str_arg == "runs":
            for prod_idx in range(6, 506):
                prod_changed = self.ChangeProductionIdx(prod_idx)
                self.PrepareSimulation()
                self.ResetTarget()
                if not prod_changed:
                    something_broke = True
                for i in range(4):
                    # imitation phase for new speech target
                    self.StartSim(False, None, False)
                self.StartSim(True, self.production_labels[prod_idx], True)
                # final, production phase for learned speech target
            self.SaveLogNoPrompt()
            print("Exiting...")
            sys.exit(0)
            return
        if str_arg == "q":
            print("Exiting...")
            sys.exit(0)
        if str_arg == "log":
            self.SaveLog()
        print("Error: invalid input. Please try again")

    def SaveLogNoPrompt(self):
        t = time.time()
        t_stamp = int(t)
        foldername = str(t_stamp) + '_DIVA_OUT_' + self.production()
        os.mkdir(foldername)
        for key in self.production_log:
            inner_dict = {}
            inner_dict[key] = self.production_log[key]
            diva_utils.write_file(foldername + '/' + key + '.mat', inner_dict)

    def SaveLog(self):
        foldername = input("Please input a directory name for the articulator movements to be backed up..")
        dest = os.path.join(foldername)
        os.mkdir(dest)
        for key in self.production_log:
            inner_dict = {}
            inner_dict[key] = self.production_log[key]
            diva_utils.write_file(foldername + '/' + key + '.mat', inner_dict)

    def Save(self):
        targetId = self.production_ids[self.targetIndex]
        if self.production_art is None:
            self.production_art = self.diva_targets.timeseries(self.production_info, doheader=True)
        if self.motor_cortex is not None:
            self.production_art['Art'] = self.motor_cortex.LMC.get_set_weights(2, None).numpy()
        t_dict = {'timeseries': self.production_art}
        filename = "diva_00000" + str(targetId) + ".mat"
        diva_utils.write_file(filename, t_dict)

    def MenuLoop(self):
        while True:
            self.ShowMenu()
            arg = input()
            self.ProcessInput(arg)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = DIVA()
    x.MenuLoop()
