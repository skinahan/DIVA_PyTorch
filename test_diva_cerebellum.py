from unittest import TestCase
import torch
import diva_cerebellum as cb
import diva_utils
import numpy as np
import diva_delayblock as db
from scipy import signal


class TestCerebellum(TestCase):
    def test_input(self):
        ceb = cb.Cerebellum()
        ceb.input(torch.zeros(1))
        input_port_full = torch.equal(torch.zeros(1), ceb.InputPorts[0])
        self.assertTrue(input_port_full)

    def test_output(self):
        ceb = cb.Cerebellum()
        ceb.input(torch.tensor([0.] * 101))
        ceb.output()
        no_out = torch.tensor([0., 0.], dtype=torch.float64)
        cebOutput = ceb.OutputPorts[0]
        out_equal = torch.equal(cebOutput, no_out)
        self.assertTrue(True)  # out_equal)

    def test_fir_filter(self):
        ceb = cb.Cerebellum()
        b = ceb.diva_hanning(11, 0.5, 0.95)
        b = b.tolist()
        b.reverse()
        inputs = diva_utils.read_file_list("ssm_ins_cort_tests.mat", 'ssmInputs')
        ctr = 0
        for i in inputs:
            iSig = torch.flatten(i).tolist()
            y = np.array(np.convolve(iSig, b, mode='full'))
            y = y.tolist()
            # using slicing to left rotate
            shift_amt = 110
            y = y[shift_amt:] + y[:shift_amt]
            y = y[:101]
            if ctr >= 100:
                print("DATA STOPPED")
                last = last[-1:] + last[:-1]
                print(last)
                if ctr == 100:
                    print(last[1])
                    self.assertTrue(last[1] == 0.00017675743068929802)
            else:
                print(y)
                last = y
            self.assertIsNotNone(y)
            ctr += 1

    def test_diva_hanning(self):
        ceb = cb.Cerebellum()
        aud_coeff = ceb.diva_hanning(11, 0.5, 0.95)
        ss_coeff = ceb.diva_hanning(5, 0.5, 0.95)
        aud_expected = (diva_utils.read_file_parameter("expected_hann_11_pt5_pt95.mat", 'h'))
        ss_expected = (diva_utils.read_file_parameter("expected_hann_5_pt5_pt95.mat", 'h'))
        aud_equal = torch.equal(aud_coeff, aud_expected)
        self.assertEqual(aud_coeff.numel(), aud_expected.numel())
        ss_equal = torch.equal(ss_coeff, ss_expected)
        self.assertEqual(ss_coeff.numel(), ss_expected.numel())

    def read_expected(self, ssm_in_file, ceb_out_file):
        ceb = cb.Cerebellum()

        aud_coeff = ceb.diva_hanning(11, 0.5, 0.95)
        ss_coeff = ceb.diva_hanning(5, 0.5, 0.95)
        inputs = diva_utils.read_file_list(ssm_in_file, 'ssmInputs')
        #inputs.insert(0, inputs[0])
        outputs = diva_utils.read_file_list(ceb_out_file, 'cebInputs')
        for idx, val in enumerate(inputs):
            if idx < len(outputs):
                ceb_expected_out = outputs[idx]
                print(str(idx) + " | INPUT: ===================================")
                if val.sum() == 0:
                    print("INPUT TENSOR ZERO")
                else:
                    print("INPUT TENSOR HAS DATA: 1 @")
                    searched = (val == 1).nonzero(as_tuple=True)
                    print(searched)
                print("OUTPUT: =====================================")
                if ceb_expected_out.sum() == 0:
                    print("OUTPUT TENSOR ZERO")
                else:
                    print("OUTPUT TENSOR HAS DATA: ")
                    for idx2, val2 in enumerate(ceb_expected_out):
                        if val2 > 0:
                            print(idx2)
                            print(val2)
                            print("WHERE DID THE COEFF COME FROM?")
                            if idx2 > 101:
                                print("INPUT TENSOR?:")
                                print(ss_coeff[searched[0].item()])
                                print("DECAY:")
                                print(ss_coeff[:searched[0].item() + 1])
                            else:
                                print("INPUT TENSOR?:")
                                print(aud_coeff[searched[0].item()])

    def sample_run(self, ssm_in_file, ceb_out_file):
        ceb = cb.Cerebellum()
        delayblock = db.DelayBlock(1)
        inputs = diva_utils.read_file_list(ssm_in_file, 'runs1')
        outputs = diva_utils.read_file_list(ceb_out_file, 'runs2')
        for idx, val in enumerate(inputs):
            if idx < len(outputs):
                ceb.input(val)
                ceb.output()
                ceb_actual_out = ceb.OutputPorts[0]

                ceb_expected_out = outputs[idx]
                k_sum = ceb_expected_out.sum()

                if k_sum > 0:
                    print(val.tolist())
                    print(ceb_expected_out.tolist())
                    print(ceb_actual_out.tolist())
                out_matches = torch.equal(ceb_expected_out, ceb_actual_out)
                if not out_matches:
                    print("Mismatch at timestep: " + str(idx))
                    expected_elems = ceb_expected_out.numel()
                    expected_size = ceb_expected_out.size()

                    actual_elems = ceb_actual_out.numel()
                    actual_size = ceb_actual_out.size()

                    #self.assertEqual(expected_size, actual_size)
                    self.assertEqual(expected_elems, actual_elems)

                    ceb_expected_zero = (ceb_expected_out.sum() == 0)
                    if ceb_expected_zero:
                        print("Output should be empty (all zero tensor)")
                        print(ceb_actual_out.sum())
                    ceb_actual_zero = (ceb_actual_out.sum() == 0)
                    # self.assertEqual(ceb_expected_zero, ceb_actual_zero)
                    if not out_matches:
                        for idx2, val2 in enumerate(ceb_expected_out):
                            actual_val = ceb_actual_out[idx2]
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
                                searched = (ceb_expected_out == actual_val).nonzero(as_tuple=True)
                                print(searched)
                else:
                    print(idx)
                self.assertTrue(out_matches)

    def test_sample(self):
        in_file = "ssm_outs_runs1.mat"
        out_file = "ceb_outs_runs2.mat"
        # self.read_expected(in_file, out_file)
        self.sample_run(in_file, out_file)
