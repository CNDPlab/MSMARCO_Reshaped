from Predictor.Utils.ScoreFunc.RougeScore import Rouge
import time
import ipdb
"""
processed_structure = {
        'have_answer': None,
        'answers': {
            index: {
                'text': '',
                'pos': None,
                'ner': None,
                'rouge': None,
                'span': None,
                'target_p': None
            } for index in range(10)
        },
        'question': {'text': '', 'pos': None, 'ner': None},
        'passages': {
            index: {
                'is_selected': None,
                'text': '',
                'pos': None,
                'ner': None,
                'is_in_question': None,
                'rouge_score': None
            } for index in range(10)
        },
        'golden_span': {'start': 0, 'end': 0, 'passage_index': -1, 'answer_index': -1, 'score': 0}
    }
"""
rouge = Rouge()


def extract_golden_span(instance):
    #start = time.time()
    if not instance['have_answer']:
        instance['golden_span'] = {'start': 0, 'end': 0, 'passage_index': -1, 'answer_index': -1, 'score': 0}
    else:
        for answer_index, answer in instance['answers'].items():
            if answer['text'] is not None:
                for passage_index, passage in instance['passages'].items():
                    if passage['is_selected']:
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
                    if offset > answer_lenth*0.6 else 0
                    for offset, _ in enumerate(passage[start_index: start_index+int(answer_lenth*1.4)])
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
                          if offset > answer_lenth*0.8 else 0
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