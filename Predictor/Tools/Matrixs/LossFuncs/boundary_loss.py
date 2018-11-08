import torch as t



def boundary_loss_func(pre_start_logits, pre_end_logits, tru_start_logits, tru_end_logits):
    """

    :param pre_start_logits: B, L ------need to be log_softmax()
    :param pre_end_logits: B, L ------need to be log_softmax()
    :param tru_start_logits: B
    :param tru_end_logits: B
    :return:
    """

    start_loss = t.nn.functional.nll_loss(pre_start_logits, tru_start_logits)
    end_loss = t.nn.functional.nll_loss(pre_end_logits, tru_end_logits)
    loss = (start_loss + end_loss) / 2.0
    return loss
