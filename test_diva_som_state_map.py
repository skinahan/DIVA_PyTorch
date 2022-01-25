from unittest import TestCase
import diva_utils
import diva_som_state_map

import torch


class TestSomatosensoryStateMap(TestCase):

    def test_output(self):
        som_state_ins = diva_utils.read_file_parameter_alternate('som_state_map_ins_111.mat', "runs1")
        som_state_outs = diva_utils.read_file_parameter_alternate('som_state_map_outs_111.mat', "runs2")
        som_state_map = diva_som_state_map.SomatosensoryStateMap()
        som_state_map.setinputdims(8)
        num_runs = 111
        for i in range(num_runs):
            som_input = som_state_ins[0, i]
            som_output_expected = som_state_outs[0, i]
            som_output_expected = torch.from_numpy(som_output_expected).to(torch.float64)
            som_input = torch.from_numpy(som_input)
            som_state_map.input(som_input)
            som_state_map.output()
            som_output_actual = som_state_map.OutputPorts[0]
            if len(som_output_actual.size()) == 1:
                som_output_actual = som_output_actual.unsqueeze(1)
            out_matches = torch.equal(som_output_actual, som_output_expected)
            if not out_matches:
                print("Expected:")
                print(som_output_expected.tolist())
                print("Actual:")
                print(som_output_actual.tolist())
            self.assertTrue(out_matches)
