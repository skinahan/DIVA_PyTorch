from unittest import TestCase

import torch

import diva_weightsexplicit as dwe
import diva_vocaltract as dvt
import diva_utils


class TestExplicitWeightBlock(TestCase):
    def test_output(self):
        vocal_tract = dvt.VocalTract()
        aim = dwe.ExplicitWeightBlock('Auditory', 0.05, 0.05, vocal_tract)
        aim_in_file = 'dwe_in_single_tests.mat'
        dy = diva_utils.read_file_parameter(aim_in_file, 'dy')
        EPS = diva_utils.read_file_parameter(aim_in_file, 'EPS')
        LAMBDA = diva_utils.read_file_parameter(aim_in_file, 'LAMBDA')
        x = diva_utils.read_file_parameter(aim_in_file, 'x')

        aim.InputPorts[0] = dy
        aim.InputPorts[1] = x
        aim.output()

        dx = aim.OutputPorts[0]
        expected_dx = diva_utils.read_file_parameter(aim_in_file, 'dx')
        are_eq = torch.equal(dx, expected_dx)

        self.assertTrue(are_eq)





