from collections import defaultdict

from edit import Edit, apply_edits


def read_m2_simple(infile, annotator=None, n=None, offset=0, add_noop=True, detokenizer=None, 
                   save_edits=True, by_annotators=False):
    answer, count = [], 0
    curr_data = {"edits": (defaultdict(list) if by_annotators else [])}
    with open(infile, "r", encoding="utf8") as fin:
        mode = "S"
        for line in fin:
            if n is not None and len(answer) >= n:
                break
            line = line.strip()
            if line == "":
                if "source" in curr_data:
                    if by_annotators:
                        curr_data["correct"] = {annotator: apply_edits(curr_data["source"], edits) 
                                                for annotator, edits in curr_data["edits"].items()}
                    else:
                        curr_data["correct"] = apply_edits(curr_data["source"], curr_data["edits"])
                    if count >= offset:
                        answer.append(curr_data)
                    if n is not None and len(answer) >= n:
                        break
                    curr_data = {"edits": (defaultdict(list) if by_annotators else [])}
                    mode = "S"
                    count += 1
                continue
            if line[:2] == mode + " ":
                mode, line = line[0], line[2:].strip()
            if mode == "S":
                if "source" in curr_data:
                    if by_annotators:
                        curr_data["correct"] = {annotator: apply_edits(curr_data["source"], edits) 
                                                for annotator, edits in curr_data["edits"].items()}
                    else:
                        curr_data["correct"] = apply_edits(curr_data["source"], curr_data["edits"])
                    if count >= offset:
                        answer.append(curr_data)
                    if n is not None and len(answer) >= n:
                        break
                    curr_data = {"edits": (defaultdict(list) if by_annotators else [])}
                    count += 1
                curr_data["source"] = line.split()
                mode = "A"
            else:
                splitted = line.split("|||")
                start, end = map(int, splitted[0].split())
                edit_type, correction, edit_annotator = splitted[1], splitted[2], int(splitted[-1])
                if (annotator is None or edit_annotator == annotator) and (by_annotators or add_noop or start >= 0):
                    edit = Edit(start, end, correction, edit_type, edit_annotator)
                    dest = curr_data["edits"][edit_annotator] if by_annotators else curr_data["edits"]
                    dest.append(edit)
        else:
            if "source" in curr_data and count >= offset:
                # if by_annotators:
                #     curr_data["correct"] = {annotator: apply_edits(curr_data["source"], edits) 
                #                             for annotator, edits in curr_data["edits"].items()}
                # else:
                #     curr_data["correct"] = apply_edits(curr_data["source"], curr_data["edits"])
                if by_annotators:
                    curr_data["correct"] = {annotator: apply_edits(curr_data["source"], edits) 
                                            for annotator, edits in curr_data["edits"].items()}
                else:
                    curr_data["correct"] = apply_edits(curr_data["source"], curr_data["edits"])
                if count >= offset:
                    answer.append(curr_data)
    for elem in answer:
        if by_annotators:
            if len(elem["edits"]) == 0:
                elem["edits"] = {0: []}
                elem["correct"] = {0: elem["source"]}
            else:
                if not add_noop:
                    elem["edits"] = {key: [x for x in value if x.start >= 0] for key, value in elem["edits"].items()}
                elem["edits"] = {key: sorted(value, key=(lambda x: (x.start, x.end))) for key, value in elem["edits"].items()}
        else:
            elem["edits"].sort(key=(lambda x: (x.start, x.end)))
    if detokenizer is not None:
        for elem in answer:
            elem["source_text"] = detokenizer(elem["source"])
            if by_annotators:
                elem["correct_text"] = {annotator: detokenizer(text) for annotator, text in elem["correct"].items()}
            elem["correct_text"] = detokenizer(elem["correct"])
    if not save_edits:
        for elem in answer:
            elem.pop("edits")
    return answer


def read_m2_with_labels(infile, label_file=None, **kwargs):
    m2_data = read_m2_simple(infile, **kwargs)
    text_labels = dict()
    if label_file is not None:
        with open(label_file, "r", encoding="utf8") as fin:
            for line in fin:
                line = line.strip()
                if line == "":
                    continue
                text, label = line.split("\t")
                text_labels[text] = label
        for elem in m2_data:
            elem["label"] = text_labels.get(" ".join(elem["source"]))
    return m2_data


def read_raw_file(infile, n=None, offset=0, detokenizer=None):
    answer, count = [], 0
    with open(infile, "r", encoding="utf8") as fin:
        for i, line in enumerate(fin):
            if i < offset:
                continue
            line = line.strip()
            if line != "":
                answer.append({"source": line}) 
                count += 1
                if n is not None and count > n:
                    break
    if detokenizer is not None:
        for elem in answer:
            elem["source_text"] = detokenizer(elem["source"])
    return answer
