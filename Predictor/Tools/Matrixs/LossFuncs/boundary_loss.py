import torch as t
import ipdb
import torch.nn.functional as F


def boundary_loss_func(pre_start_logits, pre_end_logits, tru_start, tru_end):
    """

    :param pre_start_logits: B, L ------need to be log_softmax()
    :param pre_end_logits: B, L ------need to be log_softmax()
    :param tru_start_logits: B
    :param tru_end_logits: B
    :return:
    """
    start_loss = F.nll_loss(pre_start_logits, tru_start)
    end_loss = F.nll_loss(pre_end_logits, tru_end)
    loss = (start_loss + end_loss) / 2.0
    return loss


