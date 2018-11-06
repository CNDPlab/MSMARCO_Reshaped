from itertools import chain, repeat, islice
from configs import DefaultConfig


args = DefaultConfig()


def padding(instance):
    all_passages = [instance['passages'][str(i)]['text'] for i in range(10)]
    all_answers = [instance['answers'][str(i)]['text'] for i in range(10)]
    padded_p = []
    padded_a = []
    for p in all_passages:
        padded_p.append(list(pad(p, args.max_passage_lenth)))
    for a in all_answers:
        padded_a.append(list(pad(a, args.max_answer_lenth)))
    instance['passages_text'] = padded_p
    instance['answers_text'] = padded_a
    instance['question_text'] = list(pad(instance['question']['text'], args.max_question_lenth))
    return instance


def pad(iterable, size, padding=0):
    return islice(chain(iterable, repeat(padding)), size)