import fire
import torch as t
from torch.utils.data import DataLoader
from Predictor import Models
from Trainner import Trainner
import fire
from Loaders import get_dataloader
from Predictor.Tools.Matrixs.ScoreFuncs.RougeScore.score_function import rouge_func
from Predictor.Tools.Matrixs.LossFuncs.boundary_loss import boundary_loss_func
from configs import DefaultConfig
import pickle as pk
import ipdb


def train(**kwargs):
    args = DefaultConfig()
    args.parse(kwargs)
    score_func = rouge_func
    loss_func = boundary_loss_func
    train_loader = get_dataloader('train', args.batch_size, 16)
    dev_loader = get_dataloader('dev', args.batch_size, 16)
    word_vocab = pk.load(open(args.word_vocab, 'rb'))
    char_vocab = pk.load(open(args.char_vocab, 'rb'))
    model = getattr(Models, args.model_name)(hidden_size=args.hidden_size, dropout=args.dropout, num_head=args.num_head,
                                             word_matrix=word_vocab.matrix, char_matrix=char_vocab.matrix)
    print("----Init trainner----")
    trainner = Trainner(args, model, loss_func, score_func, train_loader, dev_loader)
    trainner.init_trainner(word_vocab=word_vocab, char_vocab=char_vocab)
    print("----Train model----")
    trainner.train()

if __name__ == '__main__':
    fire.Fire()