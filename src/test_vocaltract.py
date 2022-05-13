import unittest
import diva_vocaltract as dvt
import torch
import diva_utils
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa


class TestVocalTract(unittest.TestCase):
    def test_glotlf(self):
        samplesperperiod = 92
        stepsize = float(1 / samplesperperiod)
        d = [0 + (i * stepsize) for i in range(samplesperperiod)]
        glottalsource = dvt.glotlf(0, torch.tensor(d, dtype=torch.float64), None)
        expected_gs = diva_utils.read_file_parameter('glotlf_out_synth_tests.mat', 'gsource_expected')
        are_equal = torch.equal(glottalsource, expected_gs)
        if not are_equal:
            gs_list = glottalsource.tolist()
            egs_list = expected_gs.tolist()
            for idx, val in enumerate(egs_list):
                actual = gs_list[idx][0]
                expected = val[0]
                if not actual == expected:
                    diff = (actual - expected)
                    if diff < 0.00000000000001:
                        are_equal = True
                    else:
                        are_equal = False
                        print("Mismatch at idx: " + str(idx))
                        print('Expected:')
                        print(val)
                        print('Got:')
                        print(gs_list[idx])
        self.assertTrue(are_equal)

    def test_dosound(self):
        vocal_tract = dvt.VocalTract()
        vt_in = diva_utils.read_file_parameter('vt_in_sound_tests.mat', 'whole_arg')
        vocal_tract.dosound(vt_in, True)
        vt_expected_out = diva_utils.read_file_parameter('vt_out_sound_tests_no_feedback.mat', 's')
        sd.play(vt_expected_out, 11025)

        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(vt_expected_out.flatten().numpy(), sr=11025)

        X = librosa.stft(vt_expected_out.flatten().numpy())
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=11025, x_axis='time', y_axis='hz')
        plt.colorbar()
        sd.wait()
        self.assertTrue(True)

    def test_vocaltract_compute(self):
        vocal_tract = dvt.VocalTract()
        vt_in = diva_utils.read_file_parameter_alternate('vt_ins_tests.mat', 'vtinputs')
        vt_y_out = diva_utils.read_file_parameter_alternate('vt_youts_tests.mat', 'youts')
        vt_z_out = diva_utils.read_file_parameter_alternate('vt_zouts_tests.mat', 'zouts')
        idx = 0
        for colIdx, column in enumerate(vt_in.T):
            vt_input = np.array(column[0]).astype(dtype=np.float)
            vt_input = torch.from_numpy(vt_input)
            vt_y_actual, vt_z_actual = vocal_tract.vocal_tract_compute(vt_input, None)
            vt_y_expected = torch.from_numpy(vt_y_out.T[idx][0])
            vt_z_expected = torch.from_numpy(vt_z_out.T[idx][0])
            y_equal = torch.equal(vt_y_actual, vt_y_expected)
            z_equal = torch.equal(vt_z_actual, vt_z_expected)

            if not y_equal:
                y_act_list = vt_y_actual.tolist()
                y_exp_list = vt_y_expected.tolist()

                for idx2, aud_pt in enumerate(y_act_list):
                    exp_aud_pt = y_exp_list[idx2]
                    if aud_pt[0] != exp_aud_pt[0]:
                        diff = abs(exp_aud_pt[0] - aud_pt[0])
                        if diff < 0.00000001:
                            y_equal = True
                        else:
                            y_equal = False
                            break

            if not z_equal:
                z_act_list = vt_z_actual.tolist()
                z_exp_list = vt_z_expected.tolist()

                for idx2, som_pt in enumerate(z_act_list):
                    exp_som_pt = z_exp_list[idx2]
                    if som_pt[0] != exp_som_pt[0]:
                        diff = abs(exp_som_pt[0] - som_pt[0])
                        if diff < 0.0000000001:
                            z_equal = True
                        else:
                            z_equal = False
                            break

            #self.assertEqual(y_act_list, y_exp_list)
            #self.assertEqual(z_act_list, z_exp_list)
            self.assertTrue(y_equal)
            self.assertTrue(z_equal)
            idx = idx + 1


if __name__ == '__main__':
    unittest.main()
