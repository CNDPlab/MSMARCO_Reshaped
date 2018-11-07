from utils import smith_waterman
from itertools import repeat
import multiprocessing
from tqdm import tqdm
import numpy as np
import argparse
import random
import bisect
import gzip
import json
import nltk
import time
import re

YesNoHead = ['is', 'are', 'was', 'were', 'do', 'does', 'did',
             'can', 'could', 'should', 'has', 'have', 'may',
             'might', 'am', 'will', 'would']


def preprocess(s):
    return s.replace("''", '" ').replace("``", '" ')


def tokenize(s, context_mode=False):
    nltk_tokens = [t.replace("''", '"').replace("``", '"') for t in nltk.word_tokenize(s)]
    additional_separators = (
        "-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
    # "-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
    tokens = []
    for token in nltk_tokens:
        tokens.extend([t for t in (re.split("([{}])".format("".join(additional_separators)), token)
                                   if context_mode else [token])])
    assert (not any([t == '<NULL>' for t in tokens]))
    assert (not any([' ' in t for t in tokens]))
    assert (not any(['\t' in t for t in tokens]))
    return tokens


def trim_empty(tokens):
    return [t for t in tokens if t != '']


def process(args):
    j, is_test = args

    p = j['passages']
    outputs = []

    query = preprocess(j['query'])
    qtokens = trim_empty(tokenize(query))
    yesno = False

    if qtokens[0].lower() in YesNoHead:
        yesno = True

    context = ' '.join(pp['passage_text'] for pp in p)

    if yesno:
        context = 'Yes No ' + context
    context = preprocess(context)

    ctokens = trim_empty(tokenize(context, context_mode=True))

    normalized_context = ' '.join(ctokens)
    nctokens = normalized_context.split()

    if not is_test:
        for a in j['answers']:
            bad = False
            answer = preprocess(a)
            if answer == '':
                bad = True

            if not bad:
                atokens = trim_empty(tokenize(answer, context_mode=True))
                normalized_answer = ' '.join(atokens).lower()
                normalized_context_lower = normalized_context.lower()
                pos = normalized_context_lower.find(normalized_answer)
                if pos >= 0:
                    start = bisect.bisect(np.cumsum([1 + len(t) for t in nctokens]), pos)
                    end = start + len(atokens)
                    if len(nctokens) < end:
                        bad = True
                else:
                    natokens = normalized_answer.split()
                    try:
                        (start, end), (_, _), score = smith_waterman(normalized_context_lower.split(), natokens)
                        start -= 1
                        ratio = 0.5 * score / min(len(nctokens), len(natokens))
                        if ratio < 0.51:
                            bad = True
                    except:
                        bad = True
                if not bad:
                    output = [str(j['query_id']), j['query_type'],
                              ' '.join(nctokens), ' '.join(qtokens),
                              ' '.join(nctokens[start:end]), str(start), str(end)]
                    outputs.append(output)
    else:
        output = [str(j['query_id']), j['query_type'], ' '.join(nctokens), ' '.join(qtokens)]
        outputs.append(output)

    return outputs


def convert(file_name, outfile, is_test, threads, version, ratio):
    print('Generating', outfile, '...')
    start = time.perf_counter()

    if version == 'v1':
        ##### v1 ####
        data = []
        with gzip.open(file_name, 'rb') as f:
            for line in f:
                data.append(json.loads(line))
    elif version == 'v2':
        #### v2 ####
        data = []
        with gzip.open(file_name, 'rb') as f:
            raw_data = json.load(f)

        for id_ in raw_data['query_id']:
            select = random.random() <= ratio

            if file_name.find('dev') < 0 or select:
                dict_ = dict()
                if outfile != 'test_public.tsv':
                    if raw_data['answers'][id_][0] == 'No Answer Present.':
                        continue
                    else:
                        dict_['answers'] = raw_data['answers'][id_]

                dict_['passages'] = raw_data['passages'][id_]
                dict_['query'] = raw_data['query'][id_]
                dict_['query_id'] = raw_data['query_id'][id_]
                dict_['query_type'] = raw_data['query_type'][id_]
                data.append(dict_)
    else:
        raise NotImplementedError

    outputs = []
    with multiprocessing.Pool(threads) as pool:
        outputs = []
        for output in tqdm(pool.imap(process, zip(data, repeat(is_test))), total=len(data)):
            for o in output:
                outputs.append(o)

    with open(outfile, 'w', encoding='utf-8') as out:
        for output in outputs:
            out.write("%s\n" % '\t'.join(output))

    end = time.perf_counter()
    print('Take', end - start, 'second.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MSMARCO raw data to tsv format')
    parser.add_argument('--threads', help='Number of threads to multi-preprocessing', default=1, type=int)
    parser.add_argument('--ratio', help='Ratio of dev data', default=1, type=float)
    parser.add_argument('version', choices=['v1', 'v2'])
    args = parser.parse_args()
    var = vars(args)

    convert(args.version + '/train.json.gz', 'train.tsv', False, **var)
    convert(args.version + '/dev.json.gz', 'dev.tsv', False, **var)
    convert(args.version + '/test.json.gz', 'test.tsv', True, **var)
    convert(args.version + '/test_public.json.gz', 'test_public.tsv', True, **var)

import numpy as np


def smith_waterman(tt, bb):
    # adapted from https://gist.github.com/radaniba/11019717

    # These scores are taken from Wikipedia.
    # en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
    match = 2
    mismatch = -1
    gap = -1

    def calc_score(matrix, x, y, seq1, seq2):
        '''Calculate score for a given x, y position in the scoring matrix.
        The score is based on the up, left, and upper-left neighbors.
        '''
        similarity = match if seq1[x - 1] == seq2[y - 1] else mismatch

        diag_score = matrix[x - 1, y - 1] + similarity
        up_score = matrix[x - 1, y] + gap
        left_score = matrix[x, y - 1] + gap

        return max(0, diag_score, up_score, left_score)

    def create_score_matrix(rows, cols, seq1, seq2):
        '''Create a matrix of scores representing trial alignments of the two sequences.
        Sequence alignment can be treated as a graph search problem. This function
        creates a graph (2D matrix) of scores, which are based on trial alignments.
        The path with the highest cummulative score is the best alignment.
        '''
        score_matrix = np.zeros((rows, cols))

        # Fill the scoring matrix.
        max_score = 0
        max_pos = None  # The row and columbn of the highest score in matrix.
        for i in range(1, rows):
            for j in range(1, cols):
                score = calc_score(score_matrix, i, j, seq1, seq2)
                if score > max_score:
                    max_score = score
                    max_pos = (i, j)

                score_matrix[i, j] = score

        if max_pos is None:
            raise ValueError('cannot align %s and %s' % (' '.join(seq1)[:80], ' '.join(seq2)))

        return score_matrix, max_pos

    def next_move(score_matrix, x, y):
        diag = score_matrix[x - 1, y - 1]
        up = score_matrix[x - 1, y]
        left = score_matrix[x, y - 1]
        if diag >= up and diag >= left:  # Tie goes to the DIAG move.
            return 1 if diag != 0 else 0  # 1 signals a DIAG move. 0 signals the end.
        elif up > diag and up >= left:  # Tie goes to UP move.
            return 2 if up != 0 else 0  # UP move or end.
        elif left > diag and left > up:
            return 3 if left != 0 else 0  # LEFT move or end.
        else:
            # Execution should not reach here.
            print('qq')
            raise ValueError('invalid move during traceback')

    def traceback(score_matrix, start_pos, seq1, seq2):
        '''Find the optimal path through the matrix.
        This function traces a path from the bottom-right to the top-left corner of
        the scoring matrix. Each move corresponds to a match, mismatch, or gap in one
        or both of the sequences being aligned. Moves are determined by the score of
        three adjacent squares: the upper square, the left square, and the diagonal
        upper-left square.
        WHAT EACH MOVE REPRESENTS
            diagonal: match/mismatch
            up:       gap in sequence 1
            left:     gap in sequence 2
        '''

        END, DIAG, UP, LEFT = range(4)
        x, y = start_pos
        move = next_move(score_matrix, x, y)
        while move != END:
            if move == DIAG:
                x -= 1
                y -= 1
            elif move == UP:
                x -= 1
            else:
                y -= 1
            move = next_move(score_matrix, x, y)

        return (x, y), start_pos

    rows = len(tt) + 1
    cols = len(bb) + 1

    # Initialize the scoring matrix.
    score_matrix, start_pos = create_score_matrix(rows, cols, tt, bb)

    # Traceback. Find the optimal path through the scoring matrix. This path
    # corresponds to the optimal local sequence alignment.
    (x, y), (w, z) = traceback(score_matrix, start_pos, tt, bb)
    return (x, w), (y, z), score_matrix[w][z]
