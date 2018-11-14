from Predictor.Tools.Matrixs import Rouge
import numpy as np
import bisect
import time
import ipdb
"""
    processed_structure = {
        'have_answer': None,
        'yes_no': None,
        'answers': {
            index: {
                'text': '', 'char': '', 'pos': '', 'ner': ''
            } for index in range(answer_num)
        },
        'question': {'text': '', 'pos': '', 'ner': ''},
        'passages': {
            index: {
                'is_selected': None, 'text': '', 'char': '', 'pos': '', 'ner': ''
            } for index in range(11)
        },

        'golden_span': {'start': None, 'end': None, 'passage_index': None, 'answer_index': None, 'score': 0}
    }
"""
rouge = Rouge()


def extract_golden_span(instance):
    #start = time.time()

    if not instance['have_answer']:
        instance['golden_span']['start'] = 0
        instance['golden_span']['end'] = 3
        instance['golden_span']['passage_index'] = 10
        instance['golden_span']['score'] = 1
        instance['golden_span']['answer_index'] = 0

    else:

        for answer_index, answer in instance['answers'].items():
            if answer['text'] != '':
                for passage_index, passage in instance['passages'].items():
                    if passage['is_selected'] and passage['text'] != '':
                        pos = ' '.join(passage['text']).find(' '.join(answer['text']))
                        if pos != -1:
                            instance['golden_span']['start'] = bisect.bisect(np.cumsum([1 + len(t) for t in passage['text']]), pos)
                            instance['golden_span']['end'] = instance['golden_span']['start'] + len(answer['text']) - 1
                            instance['golden_span']['passage_index'] = passage_index
                            instance['golden_span']['score'] = 1
                            instance['golden_span']['answer_index'] = answer_index
                        else:
                            local_best_result = find_best_span(answer['text'], passage['text'])
                            if local_best_result['score'] > instance['golden_span']['score']:
                                instance['golden_span']['start'] = local_best_result['start']
                                instance['golden_span']['end'] = local_best_result['end']
                                instance['golden_span']['passage_index'] = passage_index
                                instance['golden_span']['score'] = local_best_result['score']
                                instance['golden_span']['answer_index'] = answer_index
    #end = time.time()
    #print(f'use:{end-start} ansl:{instance["golden_span"]["end"]-instance["golden_span"]["start"]}')
    return instance


def find_best_span(answer, passage):
    """
    :param answer: list of token
    :param passage: list of token
    :return:
    """

    answer_lenth = len(answer)
    best_result = {'span': '0,0', 'score': 0}
    if answer_lenth < 20:
        for start_index, start_token in enumerate(passage):
            if start_token in answer:
                result = {
                    str(start_index) + ',' + str(start_index+offset): rouge.calc_score([passage[start_index: start_index+offset+1]], [answer])
                    for offset, _ in enumerate(passage[start_index: start_index+answer_lenth * 2])
                }
                local_best_span = max(result, key=result.get)
                local_best_score = result[local_best_span]
                if (local_best_score != 0) & (local_best_score > best_result['score']):
                    best_result['span'] = local_best_span
                    best_result['score'] = local_best_score

    elif answer_lenth < 100:
        for start_index, start_token in enumerate(passage):
            if start_token in answer:
                result = {
                    str(start_index) + ',' + str(start_index+offset): rouge.calc_score([passage[start_index: start_index + offset+1]], [answer])
                    if offset > answer_lenth * 0.6 else 0
                    for offset, _ in enumerate(passage[start_index: start_index+int(answer_lenth * 1.4)])
                }
                local_best_span = max(result, key=result.get)
                local_best_score = result[local_best_span]
                if (local_best_score != 0) & (local_best_score > best_result['score']):
                    best_result['span'] = local_best_span
                    best_result['score'] = local_best_score

    else:
        for start_index, start_token in enumerate(passage):
            if start_token in answer:
                result = {str(start_index) + ',' + str(start_index+offset): rouge.calc_score([passage[start_index: start_index+offset+1]], [answer])
                          if offset > answer_lenth * 0.8 else 0
                          for offset, _ in enumerate(passage[start_index: start_index+int(answer_lenth * 1.2)])}
                local_best_span = max(result, key=result.get)
                local_best_score = result[local_best_span]
                if (local_best_score != 0) & (local_best_score > best_result['score']):
                    best_result['span'] = local_best_span
                    best_result['score'] = local_best_score
    return extract_result(best_result)


def extract_result(result):
    start, end = result['span'].split(',')
    return {'start': int(start), 'end': int(end), 'score': result['score']}