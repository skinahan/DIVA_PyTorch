from unittest import TestCase

import torch

import diva_weightsadaptive as dwa
import diva_utils


class TestAdaptiveWeightBlock(TestCase):

    def test_set_weight(self):
        WeightBlock = dwa.AdaptiveWeightBlock()
        WeightBlock.set_filename("diva_weights_SSM2FF.mat")
        WeightBlock.set_weight("diva_weights_SSM2FF.mat")

    def test_set_bias(self):
        self.assertTrue(True)

    def test_get_set_weights(self):
        WeightBlock = dwa.AdaptiveWeightBlock()
        WeightBlock.set_filename("diva_weights_SSM2FF.mat")
        self.Weight = WeightBlock
        # load filename and return weight matrix size
        ans = self.Weight.get_set_weights(0, None)
        self.assertIsNotNone(ans)
        # return index to weight matrix
        ans = self.Weight.get_set_weights(1, None)
        self.assertIsNotNone(ans)
        self.assertEqual(ans, 0)
        # return weight matrix
        ans = self.Weight.get_set_weights(2, None)
        self.assertIsNotNone(ans)
        # set weight matrix
        ans = self.Weight.get_set_weights(3, None)
        self.assertIsNone(ans)
        ans = self.Weight.get_set_weights(2, None)
        self.assertIsNone(ans)
        # return weight matrix from filename
        ans = self.Weight.get_set_weights(4, None)
        self.assertIsNotNone(ans)

    def test_single(self):
        ssm_in_file = "single_ssm_cort_tests.mat"
        ceb_in_file = "single_ceb_cort_tests.mat"
        learn_in_file = "single_learn_cort_tests.mat"
        cort_out_file = "single_cort_cort_tests.mat"

        WeightBlock = dwa.AdaptiveWeightBlock()
        WeightBlock.set_filename("diva_weights_SSM2FF.mat")
        self.Weight = WeightBlock
        ssm_in = diva_utils.read_file_parameter(ssm_in_file, 'ssmInput')
        ceb_in = diva_utils.read_file_parameter(ceb_in_file, 'cebInput')
        learn_in = diva_utils.read_file_parameter(learn_in_file, 'learnInput')
        cort_out = diva_utils.read_file_parameter(cort_out_file, 't')

        self.Weight.InputPorts[0] = ssm_in
        self.Weight.InputPorts[1] = ceb_in
        self.Weight.InputPorts[2] = learn_in
        self.Weight.output()

        actual_out = self.Weight.OutputPorts[0]
        out_matches = torch.equal(cort_out, actual_out)
        self.assertTrue(out_matches)

    def test_tensor_reshape(self):
        j = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        out = diva_utils.tensor_reshape(j)
        # print(out)
        # print(out.size())

    def test_output(self):
        ssm_in_file = "ssm_ins_cort_tests.mat"
        ceb_in_file = "ceb_outs_cort_tests.mat"
        cort_out_file = "cort_outs_cort_tests.mat"
        learn_in_file = "learn_ins_cort_tests.mat"

        WeightBlock = dwa.AdaptiveWeightBlock()
        WeightBlock.set_filename("diva_weights_SSM2FF.mat")
        self.Weight = WeightBlock
        inputs = diva_utils.read_file_list(ssm_in_file, 'ssmInputs')
        ceb_inputs = diva_utils.read_file_list(ceb_in_file, 'cebInputs')
        learn_inputs = diva_utils.read_file_list(learn_in_file, 'cortLearnSignals')
        outputs = diva_utils.read_file_list(cort_out_file, 'cortexOutputs')
        for idx, val in enumerate(inputs):
            if idx < len(ceb_inputs) and idx < len(outputs) and idx < len(learn_inputs):
                ceb_in = ceb_inputs[idx]
                cort_expected_out = outputs[idx]
                learn_in = learn_inputs[idx]
                self.Weight.InputPorts[0] = val
                self.Weight.InputPorts[1] = ceb_in
                self.Weight.InputPorts[2] = learn_in
                self.Weight.output()
                cort_actual_out = self.Weight.OutputPorts[0]
                out_matches = torch.equal(cort_expected_out, cort_actual_out)
                if not out_matches:
                    print("Mismatch found at index: " + str(idx))
                    print("Expected:")
                    print(cort_expected_out)
                    print("Actual:")
                    print(cort_actual_out)

                    print("Difference:")
                    dW = cort_expected_out - cort_actual_out
                    print(dW)

                    expected_elems = cort_expected_out.numel()
                    expected_size = cort_expected_out.size()
                    actual_elems = cort_actual_out.numel()
                    actual_size = cort_actual_out.size()

                    self.assertEqual(expected_size, actual_size)
                    self.assertEqual(expected_elems, actual_elems)

                    cort_expected_zero = (cort_expected_out.sum() == 0)
                    if cort_expected_zero:
                        print("Output should be empty (all zero tensor)")
                        print(cort_expected_out.sum())
                    cort_actual_zero = (cort_actual_out.sum() == 0)
                self.assertTrue(out_matches)
