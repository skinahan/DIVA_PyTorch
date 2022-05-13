from unittest import TestCase
import diva_utils
import diva_asm
import torch


class TestAuditoryStateMap(TestCase):

    def test_output(self):
        asm_ins = diva_utils.read_file_parameter_alternate('aud_state_map_ins_111.mat', "runs1")
        asm_outs = diva_utils.read_file_parameter_alternate('aud_state_map_outs_111.mat', "runs2")
        aud_state_map = diva_asm.AuditoryStateMap()
        aud_state_map.setinputdims(4)
        num_runs = 111
        for i in range(num_runs):
            asm_input = asm_ins[0, i]
            asm_output_expected = asm_outs[0, i]
            asm_output_expected = torch.from_numpy(asm_output_expected).to(torch.float64)
            asm_input = torch.from_numpy(asm_input)
            aud_state_map.input(asm_input)
            aud_state_map.output()
            asm_output_actual = aud_state_map.OutputPorts[0]
            if len(asm_output_actual.size()) == 1:
                asm_output_actual = asm_output_actual.unsqueeze(1)
            out_matches = torch.equal(asm_output_actual, asm_output_expected)
            if not out_matches:
                print("Expected:")
                print(asm_output_expected.tolist())
                print("Actual:")
                print(asm_output_actual.tolist())
            self.assertTrue(out_matches)

