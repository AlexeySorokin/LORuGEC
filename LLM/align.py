from collections import deque
import os

from nltk import wordpunct_tokenize
from sacremoses import MosesDetokenizer, MosesTokenizer
from edit import Edit

def extract_word_positions(text, words):
    answer, start = [], 0
    for word in words:
        if word == "&quot;":
            word = '"'
        word_start = text.find(word, start)
        if word_start == -1:
            Warning(f"{word} not found in `{text}` from position {start}.")
            while start < len(text) and text[start].isspace():
                start += 1
            word_start, word_end = start, start
        else:
            word_end = word_start + len(word)
        answer.append((word_start, word_end))
        start = word_end
    return answer

def align(first, second, first_word_spans=None, second_word_spans=None, zero_gaps=True, verbose=False):
    if first_word_spans is not None and second_word_spans is not None:
        first_words = {i: first[i:j] for i, j in first_word_spans}
        second_words = {i: second[i:j] for i, j in second_word_spans}
    else:
        first_words, second_words = dict(), dict()
    used = dict()
    queue = deque([((0, 0, None, None), 0)])
    while len(queue) > 0:
        (i, j, k, l), dist = queue.popleft()
        if verbose:
            print((i, j, k, l), dist)
        if (i, j) in used:
            continue
        used[(i, j)] = (k, l)
        if i == len(first) and j == len(second):
            break
        if i < len(first) and j < len(second):
            if i in first_words and j in second_words and first_words[i] == second_words[j]:
                word = first_words[i]
                for k in range(len(word)-1):
                    used[(i+k+1,j+k+1)] = (i+k, j+k)
                queue.appendleft(((i+len(word), j+len(word), i+len(word)-1, j+len(word)-1), dist))
                continue
            elif first[i] == second[j]:
                queue.appendleft(((i+1, j+1, i, j), dist))
            else:
                queue.append(((i+1, j+1, i, j), dist+1))
        if i < len(first):
            is_gap_position = (j == 0 or j == len(second) or second[j-1] == " ")
            if first[i] == " " and zero_gaps and is_gap_position:
                queue.appendleft(((i+1, j, i, j), dist))
            else:
                queue.append(((i+1, j, i, j), dist+1))
        if j < len(second):
            is_gap_position = (i == 0 or i == len(first) or first[i-1] == " ")
            if second[j] == " " and zero_gaps and is_gap_position:
                queue.appendleft(((i, j+1, i, j), dist))
            else:
                queue.append(((i, j+1, i, j), dist+1))
    answer = [(len(first), len(second))]
    while used[answer[-1]][0] is not None:
        answer.append(used[answer[-1]])
    return answer[::-1], dist

def postprocess_word_alignment(alignment):
    answer = alignment[:2]
    for pos, (i, j) in enumerate(alignment[2:], 2):
        (k, l), (r, s) = alignment[pos-2:pos]
        if j == l or i == k:
            answer[-1] = (i, j)
        else:
            answer.append((i, j))
    return answer


def extract_aligned_word_spans(alignment, source_spans, target_spans, use_starts=False):
    source_bounds = {0: 0}
    for i, (start, end) in enumerate(source_spans):
        source_bounds[end] = i+1
        if use_starts:
            source_bounds[start] = i
    target_bounds = {0: 0}
    for i, (start, end) in enumerate(target_spans):
        target_bounds[end] = i+1
        if use_starts:
            target_bounds[start] = i
    word_alignment = []
    for i, j in alignment:
        if i in source_bounds and j in target_bounds:
            word_indexes = (source_bounds[i], target_bounds[j])
            if len(word_alignment) == 0 or word_alignment[-1] != word_indexes:
                word_alignment.append(word_indexes)
    word_alignment = postprocess_word_alignment(word_alignment)
    answer = [((i, j), (r, s)) for (i, r), (j, s) in zip(word_alignment[:-1], word_alignment[1:])]
    return answer

def are_equal_spans(first, second):
    e_table = str.maketrans({"Ё": "Е", "ё": "е"})
    if first.translate(e_table) == second.translate(e_table):
        return True
    if first == '"' and second in list('«»'):
        return False
    hyphens = ['-'] + [chr(x) for x in range(8210, 8213)]
    if first in hyphens and second in hyphens:
        return False
    quotes = list('"«»') + ["``", "''"]
    if first in quotes and second in quotes:
        return False
    return first == second
    
def normalize_change(s):
    e_table = str.maketrans({"Ё": "Е", "ё": "е", '«': '"', '»': '"'})
    return s.translate(e_table)

def extract_edits(source, target_text, use_starts=False, tokenizer=None, detokenizer=None, lang="ru", 
                  to_normalize_spans=True):
    if tokenizer is None:
        tokenizer = MosesTokenizer(lang=lang).tokenize
    if detokenizer is None:
        detokenizer = MosesDetokenizer(lang=lang).detokenize
    source_text = detokenizer(source)
    source_word_positions = extract_word_positions(source_text, source)
    target = tokenizer(target_text)
    target_word_positions = extract_word_positions(target_text, target)
    alignment, _ = align(source_text, target_text, source_word_positions, target_word_positions)
    word_alignment = extract_aligned_word_spans(alignment, source_word_positions, target_word_positions, use_starts=use_starts)
    word_alignment.sort()
    answer = []
    for (i, j), (k, l) in word_alignment:
        source_text_span = source_text[source_word_positions[i][0]:source_word_positions[j-1][1]] if i < j else ""
        target_text_span = target_text[target_word_positions[k][0]:target_word_positions[l-1][1]] if k < l else ""
        if not are_equal_spans(source_text_span, target_text_span):
            if to_normalize_spans:
                target_text_span = normalize_change(target_text_span)    
            answer.append(Edit(i, j, target_text_span))
    return answer



def evaluate(corr_edits, pred_edits):
    diffs, TP, FP, FN = [], 0, 0, 0
    for corr_sent_edits, pred_sent_edits in zip(corr_edits, pred_edits):
        FN_edits, FP_edits = [], []
        corr_sent_edits = {(edit.start, edit.end): edit for edit in corr_sent_edits}
        pred_sent_edits = {(edit.start, edit.end): edit for edit in pred_sent_edits}
        for span, edit in corr_sent_edits.items():
            if edit.start < 0:
                continue
            if span not in pred_sent_edits or pred_sent_edits[span].candidate != edit.candidate:
                FN_edits.append(edit)
                FN += 1
            else:
                TP += 1
        for span, edit in pred_sent_edits.items():
            if edit.start < 0:
                continue
            if span not in corr_sent_edits or corr_sent_edits[span].candidate != edit.candidate:
                FP_edits.append(edit)
                FP += 1
        diffs.append({"FP": FP_edits, "FN": FN_edits})
    metrics = {"P": TP/max(TP+FP, 1), "R": TP/max(TP+FN, 1), "F0.5": TP/max(TP+0.2*FN+0.8*FP, 1), "TP": TP, "FP": FP, "FN": FN}
    return metrics, diffs


if __name__ == "__main__":
    # first = "abc def abc def eg"
    # second = "abc def eg"
    # # first_word_positions = extract_word_positions(first, first.split())
    # # second_word_positions = extract_word_positions(second, second.split())
    # # alignment, diff = align(first, second, first_word_positions, second_word_positions)
    # # print("".join(first[i[0]:j[0]] or "_" for i, j in zip(alignment[:-1], alignment[1:])))
    # # print("".join(second[i[1]:j[1]] or "_" for i, j in zip(alignment[:-1], alignment[1:])))
    # # print(diff)
    # edits = extract_edits(first.split(), second, use_starts=True)
    # for edit in edits:
    #     print(edit)
    from argparse import ArgumentParser
    from read import read_m2_simple

    argument_parser = ArgumentParser()
    argument_parser.add_argument("-i", "--infiles", nargs="+")
    argument_parser.add_argument("-o", "--output_file")

    args = argument_parser.parse_args()
    data = []
    for infile in args.infiles:
        data += read_m2_simple(infile)
    detokenizer = MosesDetokenizer(lang="ru").detokenize
    answer = []
    for sent in data:
        answer.append(extract_edits(sent["source"], detokenizer(sent["correct"]), use_starts=True))
    metrics, diffs = evaluate([x["edits"] for x in data], answer)
    print(metrics)
    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf8") as fout:
            for i, curr_diffs in enumerate(diffs):
                if len(curr_diffs["FP"])+len(curr_diffs["FN"]) > 0:
                    sent = data[i]
                    print(*sent["source"], file=fout)
                    print("-"*40, file=fout)
                    for edit in curr_diffs["FN"]:
                        print(edit, file=fout)
                    print("-"*40, file=fout)
                    for edit in curr_diffs["FP"]:
                        print(edit, file=fout)
                    print("", file=fout)
    