from unittest import TestCase
import diva_rand
import rbfn_test


class TestBabbler(TestCase):

    def test_generate_movement(self):
        babbler = diva_rand.Babbler()
        x, y, z = babbler.generate_movement()
        self.assertIsNotNone(x)
        self.assertIsNotNone(y)
        self.assertIsNotNone(z)

    def test_model_fit(self):
        babbler = diva_rand.Babbler()
        test_out = babbler.train_model()
        self.assertTrue(True)


