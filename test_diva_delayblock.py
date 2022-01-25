from unittest import TestCase
import diva_delayblock as db
import torch


class TestDelayBlock(TestCase):
    def test_constructor(self):
        delay_block = db.DelayBlock(10)
        self.assertIsNotNone(delay_block)
        self.assertEqual(delay_block.delay_steps, 10)

    def test_input(self):
        delay_block = db.DelayBlock(10)
        inp = torch.tensor([100])
        delay_block.input(inp)
        self.assertEqual(inp, delay_block.InputPorts[0])

    def test_output(self):
        delay_block = db.DelayBlock(10)
        inp = torch.tensor([100])
        delay_block.input(inp)
        for i in range(10):
            delay_block.output()
            self.assertIsNotNone(delay_block.OutputPorts[0])
        delay_block.output()
        self.assertEqual(inp, delay_block.OutputPorts[0])