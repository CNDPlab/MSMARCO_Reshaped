from Trainner import BaseTrainner
import torch as t
from tqdm import tqdm
import numpy as np
import ipdb


class Trainner(BaseTrainner):
    def __init__(self, args, model, loss_func, score_func, train_loader, dev_loader, use_multi_gpu=False):
        super(Trainner, self).__init__(args, model, loss_func, score_func, train_loader, dev_loader, use_multi_gpu=True)

    def train(self):
        for epoch in range(self.args.num_epochs):
            self.train_epoch()
            self.global_epoch += 1
            self.reserve_topk_model(5)
        if self.summary_writer:
            self.summary_writer.close()
        print('Trainning Done')

    def train_epoch(self):
        self.model.train()
        for data in tqdm(self.train_loader, desc='train step'):
            train_loss = self.train_inference(data)
            train_loss.backward()

            t.nn.utils.clip_grad_norm_(parameters=[i for i in self.model.parameters() if i.requires_grad is True], max_norm=5.0)
            self.optim.step_and_update_lr()

            if self.summary_writer:
                self.summary_writer.add_scalar('loss/train_loss', train_loss.item(), self.global_step)
                self.summary_writer.add_scalar('lr', self.optim.current_lr, self.global_step)
            self.global_step += 1

            if self.global_step % self.args.eval_every_step == 0:
                eval_score, eval_loss = self.evaluation()


                if self.global_step % self.args.save_every_step == 0:
                    self.save(eval_score, eval_loss)

    def evaluation(self):
        losses = []
        scores = []
        self.model.eval()
        with t.no_grad():
            for data in tqdm(self.dev_loader, desc='eval_step'):
                loss, score = self.eval_inference(data)
                losses.append(loss.item())
                scores.append(score)
        # self.write_sample_result_text(pre, tru)
        eval_loss = np.mean(losses)
        eval_score = np.mean(scores)
        if self.summary_writer:
            self.summary_writer.add_scalar('loss/eval_loss', eval_loss, self.global_step)
            self.summary_writer.add_scalar('score/eval_score', eval_score, self.global_step)
            #self.summary_writer.add_image('dot_attention', self.model.module.get_dot_attention(), self.global_step)
            if self.use_multi_gpu:
                for i, v in self.model.module.named_parameters():
                    self.summary_writer.add_histogram(i.replace('.', '/'), v.clone().cpu().data.numpy(), self.global_step)
            else:
                for i, v in self.model.named_parameters():
                    self.summary_writer.add_histogram(i.replace('.', '/'), v.clone().cpu().data.numpy(), self.global_step)
        self.model.train()
        return eval_loss, eval_score

    def train_inference(self, data):
        self.optim.zero_grad()
        question_word, passage_word, question_char, passage_char, start, end, passage_index = [j.cuda() for j in data]
        print(passage_word.shape)
        print(passage_word.ne(0).sum(-1).max())
        pre_start, pre_end = self.model(question_word, question_char, passage_word, passage_char)
        loss = self.loss_func(pre_start, pre_end, start, end)
        # if (self.global_step==1)&(self.summary_writer is not None):
        #     with t.no_grad():
        #         self.summary_writer.add_graph(self.model, (query, passages))
        return loss

    def eval_inference(self, data):
        pre_answer_list = []
        tar_answer_list = []
        question_word, passage_word, question_char, passage_char, start, end, passage_index = [j.cuda() for j in data]
        fake_batch_size, passage_lenth = passage_word.size()
        if self.use_multi_gpu:
            passage_num = self.model.module.passage_num
        else:
            passage_num = self.model.passage_num
        batch_size = int(fake_batch_size / passage_num)

        pre_start, pre_end = self.model(question_word, question_char, passage_word, passage_char)

        loss = self.loss_func(pre_start, pre_end, start, end)
        start_pos = t.argmax(pre_start, -1)
        end_pos = t.argmax(pre_end, -1)
        for i in zip(start_pos, end_pos, passage_word.view(batch_size, passage_num, passage_lenth).view(batch_size, passage_num * passage_lenth)):
            if i[0] < i[1]:
                answer = i[2][i[0]:i[1]+1].tolist()
            else:
                answer = i[2][i[1]:i[0] + 1].tolist()
            pre_answer_list.append(answer)
        for i in zip(start, end, passage_word.view(batch_size, passage_num, passage_lenth).view(batch_size, passage_num * passage_lenth)):
            answer = i[2][i[0]:i[1]+1].tolist()
            tar_answer_list.append(answer)
        score = self.score_func(pre_answer_list, tar_answer_list)
        return loss, score
