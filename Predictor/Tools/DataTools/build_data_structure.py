


def build_data_structure(instance):
    processed_structure = {
        'have_answer': None,
        'answers': {
            index: {
                'text': '',
                'char': '',
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
                'char': '',
                'pos': None,
                'ner': None,
                'is_in_question': None,
                'rouge_score': None
            } for index in range(10)
        },

        'golden_span': {'start': 0, 'end': 0, 'passage_index': -1, 'answer_index': -1, 'score': 0}
    }

    if instance['answers'] == ["No Answer Present."]:
        processed_structure['have_answer'] = False
    else:
        processed_structure['have_answer'] = True
        for index, answer in enumerate(instance['answers']):
            processed_structure['answers'][index]['text'] = answer
    processed_structure['question']['text'] = instance['query']
    for index, passage in enumerate(instance['passages']):
        if index < 10:
            processed_structure['passages'][index]['text'] = passage['passage_text']
            processed_structure['passages'][index]['is_selected'] = passage['is_selected']
    return processed_structure