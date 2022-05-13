from unittest import TestCase
import diva_ssm
import diva_utils
import torch


class TestSpeechSoundMap(TestCase):

    def test_output(self):
        ssm = diva_ssm.SpeechSoundMap(101)
        ssm_outs_file = 'ssm_outs_runs1.mat'
        ssm_outs_file2 = 'ssm_som_outs_runs1.mat'
        ssm_outs_file3 = 'ssm_aud_outs_runs2.mat'
        inputs = diva_utils.read_file_list(ssm_outs_file, 'runs1')

        sem_outs = diva_utils.read_file_list(ssm_outs_file2, 'runs1')
        aem_outs = diva_utils.read_file_list(ssm_outs_file3, 'runs2')

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

        ssm.output()
        ssm.output_seq()

        for idx, val in enumerate(inputs):
            ssm_actual_out = ssm.OutputPorts[0]
            ssm_actual_out2 = ssm.OutputPorts[1]
            ssm_actual_out3 = ssm.OutputPorts[2]
            ssm_actual_out4 = ssm.OutputPorts[3]

            one_two_matches = torch.equal(ssm_actual_out, ssm_actual_out2)

            self.assertTrue(one_two_matches)

            ssm.output_seq()
            ssm_expected_out = inputs[idx]
            sem_expected_out = sem_outs[idx]
            aem_expected_out = aem_outs[idx]

            k_sum = ssm_expected_out.sum()

            if k_sum > 0:
                print(val.tolist())
                print(ssm_expected_out.tolist())
                print(ssm_actual_out.tolist())

            out_matches = torch.equal(ssm_expected_out, ssm_actual_out)

            if not out_matches:
                print("Mismatch at timestep: " + str(idx))
                expected_elems = ssm_expected_out.numel()
                expected_size = ssm_expected_out.size()

                actual_elems = ssm_actual_out.numel()
                actual_size = ssm_actual_out.size()

                # self.assertEqual(expected_size, actual_size)
                self.assertEqual(expected_elems, actual_elems)

                ssm_expected_zero = (ssm_expected_out.sum() == 0)
                if ssm_expected_zero:
                    print("Output should be empty (all zero tensor)")
                    print(ssm_actual_out.sum())
                ssm_actual_zero = (ssm_actual_out.sum() == 0)
                # self.assertEqual(ceb_expected_zero, ceb_actual_zero)
                if not out_matches:
                    for idx2, val2 in enumerate(ssm_expected_out):
                        actual_val = ssm_actual_out[idx2]
                        if not (val2 == actual_val):
                            diff = actual_val.item() - val2.item()
                            out_matches = False
                            print("DIVERGENT IDX:")
                            print(idx2)
                            print("Expected: ")
                            print(val2.item())
                            print("Got: ")
                            print(actual_val.item())
                            print("Difference: ")
                            print(diff)
                            print("Searching for actual val in expected...")
                            searched = (ssm_expected_out == actual_val).nonzero(as_tuple=True)
                            print(searched)
            else:
                print(idx)

            out_2_matches = torch.equal(ssm_actual_out3, aem_expected_out)
            if not out_2_matches:
                print("Mismatch at timestep: " + str(idx))
                expected_elems = aem_expected_out.numel()
                expected_size = aem_expected_out.size()

                actual_elems = ssm_actual_out3.numel()
                actual_size = ssm_actual_out3.size()

                # self.assertEqual(expected_size, actual_size)
                self.assertEqual(expected_elems, actual_elems)

                ssm_expected_zero = (aem_expected_out.sum() == 0)
                if ssm_expected_zero:
                    print("Output should be empty (all zero tensor)")
                    print(ssm_actual_out3.sum())
                ssm_actual_zero = (ssm_actual_out3.sum() == 0)
                # self.assertEqual(ceb_expected_zero, ceb_actual_zero)
                if not out_2_matches:
                    for idx2, val2 in enumerate(aem_expected_out):
                        actual_val = ssm_actual_out3[idx2]
                        if not (val2 == actual_val):
                            diff = actual_val.item() - val2.item()
                            out_2_matches = False
                            print("DIVERGENT IDX:")
                            print(idx2)
                            print("Expected: ")
                            print(val2.item())
                            print("Got: ")
                            print(actual_val.item())
                            print("Difference: ")
                            print(diff)
                            print("Searching for actual val in expected...")
                            searched = (aem_expected_out == actual_val).nonzero(as_tuple=True)
                            print(searched)

            out_3_matches = torch.equal(ssm_actual_out4, sem_expected_out)

            if not out_3_matches:
                print("Mismatch at timestep: " + str(idx))
                expected_elems = sem_expected_out.numel()
                expected_size = sem_expected_out.size()

                actual_elems = ssm_actual_out4.numel()
                actual_size = ssm_actual_out4.size()

                # self.assertEqual(expected_size, actual_size)
                self.assertEqual(expected_elems, actual_elems)

                ssm_expected_zero = (sem_expected_out.sum() == 0)
                if ssm_expected_zero:
                    print("Output should be empty (all zero tensor)")
                    print(ssm_actual_out4.sum())
                ssm_actual_zero = (ssm_actual_out4.sum() == 0)
                # self.assertEqual(ceb_expected_zero, ceb_actual_zero)
                if not out_3_matches:
                    for idx2, val2 in enumerate(sem_expected_out):
                        actual_val = ssm_actual_out4[idx2]
                        if not (val2 == actual_val):
                            diff = actual_val.item() - val2.item()
                            out_3_matches = False
                            print("DIVERGENT IDX:")
                            print(idx2)
                            print("Expected: ")
                            print(val2.item())
                            print("Got: ")
                            print(actual_val.item())
                            print("Difference: ")
                            print(diff)
                            print("Searching for actual val in expected...")
                            searched = (sem_expected_out == actual_val).nonzero(as_tuple=True)
                            print(searched)

            self.assertTrue(out_matches)
            self.assertTrue(out_2_matches)
            self.assertTrue(out_3_matches)
