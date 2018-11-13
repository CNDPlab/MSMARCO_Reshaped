

YesNoHead = ['is', 'are', 'was', 'were', 'do', 'does', 'did',
             'can', 'could', 'should', 'has', 'have', 'may',
             'might', 'am', 'will', 'would']

def build_data_structure(instance):
    answer_num = len(instance['answers'])
    processed_structure = {
        'have_answer': None,
        'yes_no': False,
        'answers': {
            index: {
                'text': '', 'char': '', 'pos': '', 'ner': ''
            } for index in range(answer_num)
        },
        'question': {'text': '', 'pos': '', 'ner': ''},
        'passages': {
            index: {
                'is_selected': 0, 'text': '', 'char': '', 'pos': '', 'ner': ''
            } for index in range(11)
        },

        'golden_span': {'start': None, 'end': None, 'passage_index': None, 'answer_index': None, 'score': 0}
    }

    if instance['answers'] == ["No Answer Present."]:
        processed_structure['have_answer'] = False
        processed_structure['passages'][10]['text'] = 'No Answer Present.'
        processed_structure['passages'][10]['is_selected'] = True
        processed_structure['answers'][0]['text'] = 'No Answer Present.'
    else:
        processed_structure['have_answer'] = True
        processed_structure['passages'][10]['text'] = 'No Answer Present.'
        processed_structure['passages'][10]['is_selected'] = False
        for index, answer in enumerate(instance['answers']):
            processed_structure['answers'][index]['text'] = answer

    processed_structure['question']['text'] = instance['query'].replace("''", '" ').replace("``", '" ')

    for index, passage in enumerate(instance['passages']):
        if index < 10:
            if instance['answers'] == ['Yes'] and instance['query'].split(' ')[0] in YesNoHead:
                processed_structure['passages'][index]['text'] = 'Yes. '+passage['passage_text'].replace("''", '" ').replace("``", '" ')
                processed_structure['passages'][index]['is_selected'] = passage['is_selected']
                processed_structure['yes_no'] = True
            elif instance['answers'] == ['No'] and instance['query'].split(' ')[0] in YesNoHead:
                processed_structure['passages'][index]['text'] = 'No. '+passage['passage_text'].replace("''", '" ').replace("``", '" ')
                processed_structure['passages'][index]['is_selected'] = passage['is_selected']
                processed_structure['yes_no'] = True
            else:
                processed_structure['passages'][index]['text'] = passage['passage_text'].replace("''", '" ').replace("``", '" ')
                processed_structure['passages'][index]['is_selected'] = passage['is_selected']

    processed_structure['passages'] = sorted(processed_structure['passages'].items(), key=lambda x: int(x[0]))

    return processed_structure