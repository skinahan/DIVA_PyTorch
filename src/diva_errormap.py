import diva_sigblock as sb
import diva_minmax as dm
import diva_weights as dw
import diva_sumblock as dsb
import torch


class ErrorMap(sb.SignalBlock):

    def __init__(self, max_file, min_file):
        sb.SignalBlock.__init__(self)
        # input 1: Target
        # Input 2: CurrentState
        self.max_file = max_file
        self.min_file = min_file

        self.set_inputs(2)
        self.set_outputs(1)
        self.weight_max = dw.WeightBlock()
        self.weight_max.set_weight(max_file)
        self.target_max_gain = 1

        self.weight_min = dw.WeightBlock()
        self.weight_min.set_weight(min_file)
        self.target_min_gain = 1

        self.min_block = dm.MinMax('min')
        self.max_block = dm.MinMax('max')
        self.sum_block = dsb.SumBlock('+-')
        self.error_gain = 1

    def reset(self):
        self.InputPorts[0] = torch.zeros(self.InputPortDimensions[0], dtype=torch.float64)
        self.InputPorts[1] = torch.zeros(self.InputPortDimensions[1], dtype=torch.float64)
        self.OutputPorts[0] = torch.zeros(self.OutputPortDimensions[0], dtype=torch.float64)
        self.weight_max.reset()
        self.weight_min.reset()

    def output(self):
        target = self.InputPorts[0]
        current_state = self.InputPorts[1]
        if len(current_state.size()) == 1:
            current_state = current_state.unsqueeze(1)

        num_nonzero = torch.nonzero(current_state).size(0)
        if num_nonzero > 0:
            beginSequence = True

        current_state = torch.transpose(current_state, 0, 1)

        self.weight_max.InputPorts[0] = target
        weight_max_out = self.weight_max.output()

        self.weight_min.InputPorts[0] = target
        weight_min_out = self.weight_min.output()

        # Target_max scaling
        weight_max_out = weight_max_out * self.target_max_gain

        # Target_min scaling
        weight_min_out = weight_min_out * self.target_min_gain

        self.min_block.InputPorts[0] = current_state
        self.min_block.InputPorts[1] = weight_max_out

        self.min_block.output()

        min_block_out = self.min_block.OutputPorts[0]

        self.max_block.InputPorts[0] = min_block_out
        self.max_block.InputPorts[1] = weight_min_out

        self.max_block.output()

        max_block_out = self.max_block.OutputPorts[0]

        self.sum_block.InputPorts[0] = current_state
        self.sum_block.InputPorts[1] = max_block_out

        self.sum_block.output()

        sum_block_out = self.sum_block.OutputPorts[0]

        # Error scaling
        sum_block_out = sum_block_out * self.error_gain
        data_sz = sum_block_out.size()
        if data_sz[0] < data_sz[1]:
            sum_block_out = torch.transpose(sum_block_out, 0, 1)
        if self.OutputPortDimensions[0] != sum_block_out.size():
            self.OutputPortDimensions[0] = sum_block_out.size()
        self.OutputPorts[0] = sum_block_out
