from unittest import TestCase
import diva_utils
import diva_aem
import torch


class TestAuditoryErrorMap(TestCase):

    def test_output(self):
        aem_ins_cs = diva_utils.read_file_parameter_alternate('aud_err_map_ins_cs_111.mat', "runs1")
        aem_ins_targets = diva_utils.read_file_parameter_alternate('aud_err_map_ins_target_111.mat', "runs1")
        aem_outs = diva_utils.read_file_parameter_alternate('aud_err_map_outs_111.mat', "runs1")
        aud_err_map = diva_aem.AuditoryErrorMap()
        num_runs = 111
        for i in range(num_runs):
            cs_in = aem_ins_cs[0, i]
            target_in = aem_ins_targets[0, i]
            aem_out_expected = aem_outs[0, i]
            aem_out_expected = torch.from_numpy(aem_out_expected).to(torch.float64)
            cs_in = torch.from_numpy(cs_in).to(torch.float64)
            target_in = torch.from_numpy(target_in).to(torch.float64)
            aud_err_map.InputPorts[0] = target_in
            aud_err_map.InputPorts[1] = cs_in
            if i == 110:
                jah = 32
            aud_err_map.output()
            aem_output_actual = aud_err_map.OutputPorts[0]
            if len(aem_output_actual.size()) == 1:
                aem_output_actual = aem_output_actual.unsqueeze(1)
            out_matches = torch.equal(aem_output_actual, aem_out_expected)
            if not out_matches:
                print("Expected:")
                print(aem_out_expected.tolist())
                print("Actual:")
                print(aem_output_actual.tolist())
            self.assertTrue(out_matches)

