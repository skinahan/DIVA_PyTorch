import torch
import diva_delayblock as ddb
import diva_sigblock as sb
import diva_weightsadaptive as dwa
import diva_weightsexplicit as dwe
import diva_nullspacegain as dnsg
import diva_sumblock as dsb
import diva_vel2pos as v2p


class MotorCortex(sb.SignalBlock):
    """
        PyTorch implementation of the DIVA Motor Cortex module
        (Articulatory Velocity and Position Maps)
    """

    def __init__(self, vt):
        sb.SignalBlock.__init__(self)
        self.set_inputs(4)
        self.set_outputs(1)

        self.setinputdims(13, 2)
        self.setinputdims(13, 2)

        # Tunable scalar gains
        self.feedforward_gain = 1
        self.feedback_gain = 1
        self.position_gain = 1

        # Fixed scalar gains
        self.ff_gain_fixed = 1
        self.fb_gain_fixed = 0.5
        self.sfb_gain_fixed = 1
        self.afb_gain_fixed = 1

        # Built-in Delays
        self.ssm_delay = ddb.DelayBlock(1)
        self.ceb_delay = ddb.DelayBlock(1)
        self.sem_input_delay = ddb.DelayBlock(1)
        self.aem_input_delay = ddb.DelayBlock(1)

        self.sem_input_delay.setinputdims(8)
        self.aem_input_delay.setinputdims(13)

        self.sim_cap_delay = ddb.DelayBlock(5)
        self.sim_cap_delay.setinputdims(13)

        self.aim_cap_delay = ddb.DelayBlock(11)
        self.aim_cap_delay.setinputdims(13)

        # LMC - Learned Motor Command
        self.LMC = dwa.AdaptiveWeightBlock()
        self.LMC.set_filename("diva_weights_SSM2FF.mat")

        # SIM - Somatosensory Inverse Mapping
        self.SIM = dwe.ExplicitWeightBlock('Somatosensory', 0.05, 0.05, vt)

        # Somat. NullSpace
        self.somat_nullspace = dnsg.NullSpaceGain('Auditory&Somatosensory', 0.10, 0.05, 0.05, vt)
        self.sfb_sum = dsb.SumBlock('+-')

        # AIM - Auditory Inverse Mapping
        self.AIM = dwe.ExplicitWeightBlock('Auditory', 0.05, 0.05, vt)

        # Audit. NullSpace
        self.aud_nullspace = dnsg.NullSpaceGain('Auditory&Somatosensory', 0.10, 0.05, 0.05, vt)
        self.afb_sum = dsb.SumBlock('+-')

        self.feedback_sum = dsb.SumBlock('++')
        self.combined_sum = dsb.SumBlock('++')
        self.vel2pos = v2p.Vel2Pos(0.95)

        default_out = torch.zeros(1, 13, dtype=torch.float64)
        self.dataBuffer = [default_out]

    def reset(self):
        self.ssm_delay.reset()
        self.ceb_delay.reset()
        self.sem_input_delay.reset()
        self.aem_input_delay.reset()
        self.sim_cap_delay.reset()
        self.aim_cap_delay.reset()
        self.LMC.reset()
        self.SIM.reset()
        self.somat_nullspace.reset()
        self.AIM.reset()
        self.aud_nullspace.reset()
        self.vel2pos.reset()

    def output(self):
        ssm_input = self.InputPorts[0]
        ceb_input = self.InputPorts[1]
        sem_input = self.InputPorts[2]
        aem_input = self.InputPorts[3]

        self.ssm_delay.input(ssm_input)
        self.ceb_delay.input(ceb_input)
        self.sem_input_delay.input(sem_input)
        self.aem_input_delay.input(aem_input)

        # Advance delay blocks...
        self.ssm_delay.output()
        self.ceb_delay.output()
        self.sem_input_delay.output()
        self.aem_input_delay.output()

        # 'Current' Articulatory Position (delayed)
        self.sim_cap_delay.output()
        self.aim_cap_delay.output()

        ssm_input = self.ssm_delay.OutputPorts[0]
        ceb_input = self.ceb_delay.OutputPorts[0]
        sem_input = self.sem_input_delay.OutputPorts[0]
        aem_input = self.aem_input_delay.OutputPorts[0]

        sim_cap = self.sim_cap_delay.OutputPorts[0]
        aim_cap = self.aim_cap_delay.OutputPorts[0]

        # Calculate inverse mappings - somatosensory
        self.SIM.InputPorts[0] = sem_input
        self.SIM.InputPorts[1] = sim_cap
        self.SIM.output()
        sim_out = self.SIM.OutputPorts[0]

        # Somatosensory null-space gain
        self.somat_nullspace.InputPorts[0] = sim_cap
        self.somat_nullspace.output()
        somat_null = self.somat_nullspace.OutputPorts[0]

        # Subtract somat. nullspace gain from the somat. inverse mapping output
        self.sfb_sum.InputPorts[0] = sim_out
        self.sfb_sum.InputPorts[1] = somat_null
        self.sfb_sum.output()
        somat_feedback = self.sfb_sum.OutputPorts[0]

        # Apply somatosensory feedback gain factor (1 by default)
        somat_feedback = somat_feedback * self.sfb_gain_fixed

        # Calculate inverse mappings - auditory
        self.AIM.InputPorts[0] = aem_input
        self.AIM.InputPorts[1] = aim_cap
        self.AIM.output()
        aim_out = self.AIM.OutputPorts[0]

        # Auditory null-space gain
        self.aud_nullspace.InputPorts[0] = aim_cap
        self.aud_nullspace.output()
        aud_null = self.aud_nullspace.OutputPorts[0]

        # Subtract aud. nullspace gain from the aud. inverse mapping output
        self.afb_sum.InputPorts[0] = aim_out
        self.afb_sum.InputPorts[1] = aud_null
        self.afb_sum.output()
        aud_feedback = self.afb_sum.OutputPorts[0]

        # Apply auditory feedback gain factor (1 by default)
        aud_feedback = aud_feedback * self.afb_gain_fixed

        # Create learning signal for learned motor command (adaptive weight block)
        learn_sig = torch.cat((aud_feedback, somat_feedback), 0).to(torch.float64)

        # Combine Auditory, Somatosensory feedback to create the feedback motor command
        self.feedback_sum.InputPorts[0] = somat_feedback
        self.feedback_sum.InputPorts[1] = aud_feedback
        self.feedback_sum.output()
        summed_feedback = self.feedback_sum.OutputPorts[0]

        # vel2pos subsystem seems to smooth the feedback signal..
        self.vel2pos.InputPorts[0] = summed_feedback
        self.vel2pos.output()

        # Apply overall feedback motor command gain factor...
        feedback_command = self.vel2pos.OutputPorts[0]
        feedback_command = feedback_command * self.feedback_gain

        # Apply fixed FB gain...
        feedback_command = feedback_command * self.fb_gain_fixed

        self.LMC.InputPorts[0] = ssm_input
        self.LMC.InputPorts[1] = ceb_input
        self.LMC.InputPorts[2] = learn_sig
        self.LMC.output()

        # Apply overall feedforward motor command gain factor...
        feedforward_command = self.LMC.OutputPorts[0]
        feedforward_command = torch.transpose(feedforward_command, 0, 1)
        feedforward_command = feedforward_command * self.feedforward_gain

        # Apply fixed FF gain...
        feedforward_command = feedforward_command * self.ff_gain_fixed

        # Create final combined motor command...
        self.combined_sum.InputPorts[0] = feedforward_command
        self.combined_sum.InputPorts[1] = feedback_command
        self.combined_sum.output()

        # Apply overall current articulatory position gain factor...
        curr_art_pos = self.combined_sum.OutputPorts[0]
        curr_art_pos = curr_art_pos * self.position_gain

        self.sim_cap_delay.input(curr_art_pos)
        self.aim_cap_delay.input(curr_art_pos)

        self.OutputPorts[0] = curr_art_pos
