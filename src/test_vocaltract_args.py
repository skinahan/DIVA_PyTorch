from unittest import TestCase
import vocaltract_args as vta



class TestVocalTractArgs(TestCase):

    def test_init(self):
        vtargs = vta.VocalTractArgs()
        self.assertIsNotNone(vtargs)


