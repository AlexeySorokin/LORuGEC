from typing import Optional, Tuple, List, Union
from dataclasses import dataclass


@dataclass
class Edit:
    start: int
    end: int
    candidate: str
    label: Optional[str] = None
    annotator: Optional[int] = 0

    def __str__(self):
        return "|||".join([
            f"{self.start} {self.end}", str(self.label), self.candidate, "REQUIRED", "-NONE-", str(self.annotator)
        ])
    

def apply_edits(sent: List[str], edits: List[Edit]):
    new_sent_words = sent[:]
    reverse_edit_data = [Edit(0, 0, "", None)]
    for i, word in enumerate(sent):
        reverse_edit_data.append(Edit(i, i + 1, word, None))
        reverse_edit_data.append(Edit(i + 1, i + 1, "", None))
    for edit in sorted(edits, key=(lambda x: (-x.end, -x.start))):
        if edit.start < 0:
            continue
        new_sent_words[edit.start:edit.end] = edit.candidate.split()
        reverse_edit = Edit(edit.start, edit.end, " ".join(sent[edit.start:edit.end]), edit.label)
        words_number = len(edit.candidate.split())
        reverse_edits = [reverse_edit] * (2 * words_number + 1)
        reverse_edit_data[2 * edit.start:2 * edit.end + 1] = reverse_edits
    return new_sent_words


def dump_sentence_annotation(curr_data):
    annotators_number = max(curr_data["edits"]) + 1
    curr_edits = [curr_data["edits"][i] for i in range(annotators_number)]
    # noinspection PyTypeChecker
    curr_corrections = [
        apply_edits(curr_data["source"], annotator_edits)["sent"] for annotator_edits in curr_edits
    ]
    answer = {
        "source": curr_data["source"], "correct": curr_corrections, "edits": curr_edits
    }
    return answer