from unittest import TestCase
import diva_utils
import diva_sem
import torch


class TestSomatosensoryErrorMap(TestCase):

    def test_output(self):
        sem_ins_cs = diva_utils.read_file_parameter_alternate('som_err_map_ins_cs_111.mat', "runs1")
        sem_ins_targets = diva_utils.read_file_parameter_alternate('som_err_map_ins_target_111.mat', "runs1")
        sem_outs = diva_utils.read_file_parameter_alternate('som_err_map_outs_111.mat', "runs1")
        som_err_map = diva_sem.SomatosensoryErrorMap()
        num_runs = 111
        for i in range(num_runs):
            cs_in = sem_ins_cs[0, i]
            target_in = sem_ins_targets[0, i]
            sem_out_expected = sem_outs[0, i]
            sem_out_expected = torch.from_numpy(sem_out_expected).to(torch.float64)
            cs_in = torch.from_numpy(cs_in).to(torch.float64)
            target_in = torch.from_numpy(target_in).to(torch.float64)
            som_err_map.InputPorts[0] = target_in
            som_err_map.InputPorts[1] = cs_in
            if i == 110:
                endSequence = True
            som_err_map.output()
            sem_output_actual = som_err_map.OutputPorts[0]
            if len(sem_output_actual.size()) == 1:
                sem_output_actual = sem_output_actual.unsqueeze(1)
            out_matches = torch.equal(sem_output_actual, sem_out_expected)
            if not out_matches:
                print("Expected:")
                print(sem_out_expected.tolist())
                print("Actual:")
                print(sem_output_actual.tolist())
            self.assertTrue(out_matches)

