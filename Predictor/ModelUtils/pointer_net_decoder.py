import torch as t
import torch.nn.functional as F



class PointerNetDecoder(t.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PointerNetDecoder, self).__init__()
        self.rnn_cell = t.nn.GRUCell(input_size, hidden_size)
        self.C = hidden_size ** 0.5


    def forward(self, inputs, start_side_info, inputs_mask):
        """

        :param inputs: B, L, H
        :param side_info: B, 1, H
        :return:
        """
        net = t.bmm(inputs, start_side_info.transpose(-1, -2)).squeeze(-1) / self.C
        # B, L
        start = F.softmax(net, -1)

        end_side_info = self.rnn_cell(net, start_side_info)
        # B, L
        net = t.bmm(inputs, end_side_info.unsqueeze(-1)).squeeze(-1)

        end = F.softmax(net, -1)

        return start, end


