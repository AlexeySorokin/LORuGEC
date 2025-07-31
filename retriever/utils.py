import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForTokenClassification
from tqdm.auto import tqdm

def read_processed_infile(infile, n=None):
    answer, words, labels = [], [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                if len(words) > 0:
                    answer.append({"words": words, "labels": labels})
                words, labels = [], []
                if n is not None and len(answer) == n:
                    break
                continue
            word, label = line.split("\t", maxsplit=1)
            words.append(word)
            labels.append(label)
    if len(words) > 0:
        answer.append({"words": words, "labels": labels})
    return Dataset.from_list(answer)

class Seq2LabelsDatasetCreator:

    def __init__(self, tokenizer, to_tensors=False, label_field="labels", class_label_field=None, has_padding_between_words=True):
        self.tokenizer = tokenizer
        self.to_tensors = to_tensors
        self.label_field = label_field
        self.class_label_field = class_label_field
        self.has_padding_between_words = has_padding_between_words

    @property
    def has_eos(self):
        return self.tokenizer.eos_token_id is not None

    def _insert_padding_tokens(self, tokens, word_starts_in_tokens):
        answer = tokens[:word_starts_in_tokens[0]]
        for i, start in enumerate(word_starts_in_tokens[1:], 1):
            answer.extend(["<pad>"]+tokens[word_starts_in_tokens[i-1]:start])
        return answer
    
    def _make_token_type_ids(self, word_starts):
        answer = [1]
        for start in word_starts[:-2]:
            answer.extend([0, 1])
        return answer

    def make_side_masks(self, word_starts_in_tokens):
        left_mask, right_mask = [], []
        for start in word_starts_in_tokens[:-1]:
            left_mask.extend([start-1, start])
            right_mask.extend([start, start])
        left_mask.pop()
        right_mask.pop()
        return left_mask, right_mask
        
    def _make_label_positions_mask(self, tokens, word_starts_in_tokens):
        answer = np.array([x == "<pad>" for x in tokens])
        starts = np.array(word_starts_in_tokens[:-2])+np.arange(len(word_starts_in_tokens)-2)+1
        answer[starts] = True
        return answer

    def _convert_to_tensor(self, key, value):
        if key in ["input_ids", "token_type_ids", "labels", "left_mask", "right_mask", "mean_mask"]:
            return torch.LongTensor(value)
        elif key in ["mask", "labels_mask", "label_positions_mask"]:
            return torch.BoolTensor(value)
        else:
            return value

    def __call__(self, item):
        if not self.has_padding_between_words:
            words = ["<pad>"]
            for word in item["words"]:
                words.extend([word, "<pad>"])
            real_words = item["words"]
        else:
            words, real_words = item["words"], item["words"][1::2]
        real_word_starts = [0]
        for word in real_words[:-1]:
            real_word_starts.append(real_word_starts[-1]+len(word)+1)
        tokenization = self.tokenizer(" ".join(real_words))
        tokens = [self.tokenizer.decode(inp_id) for inp_id in tokenization["input_ids"]]
        word_starts_in_tokens = [tokenization.char_to_token(start) for start in real_word_starts]
        word_starts_in_tokens.extend([len(tokens)-1, len(tokens)])
        updated_tokens = self._insert_padding_tokens(tokens, word_starts_in_tokens)
        left_mask, right_mask = self.make_side_masks(word_starts_in_tokens)
        answer = {
            "words": words, "input_ids": tokenization["input_ids"], "tokens": updated_tokens, 
            "token_type_ids": self._make_token_type_ids(word_starts_in_tokens),
            "left_mask": left_mask, "right_mask": right_mask, "mean_mask": [-1]+list(tokenization.word_ids()[1:-1])+[-1],
            "mask": np.ones_like(tokens, dtype=bool),
            "label_positions_mask": self._make_label_positions_mask(updated_tokens, word_starts_in_tokens),
            "labels_mask": np.ones_like(words, dtype=bool)
        }
        if self.label_field is not None and self.label_field in item:
            binary_labels = [int(x != "Keep") for x in item[self.label_field]]
            answer.update({"operation_type": item[self.label_field], 
                           "labels": binary_labels, 
                        #    "labels_mask": np.ones_like(item[self.label_field], dtype=bool)
                           })
        if self.class_label_field is not None and self.class_label_field in item:
            answer["class_labels"] = item[self.class_label_field]
        if self.to_tensors:
            answer = {key: self._convert_to_tensor(key, value) for key, value in answer.items()}       
        return answer
    

def multilabel_dataset_flattener(examples, label_column="class_label", output_label_column="class_label", labels_list_column=None):
    answer = {key: [] for key in examples}
    if labels_list_column is not None:
        answer[labels_list_column] = []
    for i, curr_labels in enumerate(examples[label_column]):
        curr_item = {key: examples[key][i] for key in examples if key != label_column}
        for label in curr_labels:
            for key, value in curr_item.items():
                answer[key].append(value)
            answer[output_label_column].append(label)
            if labels_list_column is not None:
                answer[labels_list_column].append(curr_labels)    
    return answer


class HuggingfaceModelWrapper(torch.nn.Module):

    def __init__(self, model):
        super(HuggingfaceModelWrapper, self).__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        model = AutoModelForTokenClassification.from_pretrained(name, **kwargs)
        return HuggingfaceModelWrapper(model=model)

    @property
    def config(self):
        return self.model.config
    
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        result = self.model(input_ids, attention_mask, output_hidden_states=output_hidden_states)
        answer = {
            "log_probs": result["logits"].log_softmax(dim=-1),
            "labels": result["logits"].argmax(dim=-1),
            "last_hidden_state": result["hidden_states"][-1]
        }
        return answer
    


def predict_with_model(dataset, model, collate_func, test_batch_size=64,
                       return_probs=False, output_hidden_states=False,
                       labels_mask_column="labels_mask"):
    """
    Применяет модель к данным и выдает метки классов
    """
    to_remove = []
    for column in ["label_positions_mask", "operation_type"]:
        if column in dataset.column_names:
            to_remove.append(column)
    loader = DataLoader(dataset.remove_columns(to_remove),#, "operation_type"]),
                        batch_size=test_batch_size, shuffle=False, collate_fn=collate_func)
    answer, probs_answer, hidden_states_answer = [], [], []
    kwargs = {"output_hidden_states": output_hidden_states}
    for batch in tqdm(loader):
        
        with torch.no_grad():
            output = model(**batch, **kwargs) ## model.forward(**batch, **kwargs)
        for r, (labels, masks) in enumerate(zip(output["labels"], batch[labels_mask_column])):
            labels = labels.cpu().numpy()[masks.cpu().numpy()]
            answer.append(labels)
            if return_probs:
                probs = torch.softmax(output["log_probs"][r], dim=-1).cpu().numpy()[masks.cpu().numpy()]
                probs_answer.append(probs)
            if output_hidden_states:
                try:
                    hidden_states = output["last_hidden_state"][r].cpu().numpy()[masks.cpu().numpy()]
                except KeyError:
                    hidden_states = output["hidden_states"][r].cpu().numpy()[masks.cpu().numpy()]
                hidden_states_answer.append(hidden_states)
    if return_probs or output_hidden_states:
        full_answer = {"labels": answer, "probs": None, "hidden_states": None}
        if return_probs:
            full_answer["probs"] = probs_answer
        if output_hidden_states:
            full_answer["hidden_states"] = hidden_states_answer
        return full_answer
    else:
        return answer
    

def is_multiword_operation(label):
    return label.startswith("Join") or label.startswith('Hyphenate') or label == 'Delete'

def labels_to_operations(labels):
    answer, state, state_start = [], None, None
    for i, label in enumerate(labels):
        pos = int(i // 2)
        if state is not None:
            if label == state or (state == "Delete" and i % 2 == 0 and label == "Keep"):
                continue
            if pos > state_start + int(state != "Delete"):
                answer.append((state_start, pos, state))
            state, state_start = None, None
        if label != "Keep":
            if i % 2 == 1:
                if is_multiword_operation(label):
                    state, state_start = label, pos
                else:
                    answer.append((pos, pos+1, label))
            else:
                if label.startswith('Insert'):
                    answer.append((pos, pos, label))
    if state is not None:
        pos = int(len(labels) // 2)
        if pos > state_start + int(state != "Delete"):
            answer.append((state_start, pos, state))
    return answer

def score_operations(corr_operations, pred_operations):
    i, j = 0, 0
    counts = defaultdict(int)
    while i < len(corr_operations) and j < len(pred_operations):
        left_span, right_span = tuple(corr_operations[i][:2]), tuple(pred_operations[j][:2])
        if left_span == right_span:
            if corr_operations[i][2] == pred_operations[j][2]:
                counts["TP"] += 1
            else:
                counts["FP"] += 1
                counts["FN"] += 1
            i, j = i+1, j+1
        elif left_span < right_span:
            counts["FN"] += 1
            i += 1
        else:
            counts["FP"] += 1
            j += 1
    counts["FN"] += len(corr_operations)-i
    counts["FP"] += len(pred_operations)-j
    return counts

def evaluate(corr_labels, pred_labels):
    stats = {"TP": 0, "FP": 0, "FN": 0, "corr_sents": 0, "total_sents": len(pred_labels)}
    for corr_sent_labels, pred_sent_labels in zip(corr_labels, pred_labels):
        corr_sent_operations = labels_to_operations(corr_sent_labels)
        pred_sent_operations = labels_to_operations(pred_sent_labels)
        sent_stats = score_operations(corr_sent_operations, pred_sent_operations)
        for key, count in sent_stats.items():
            stats[key] += count
        stats["corr_sents"] += int(corr_sent_operations==pred_sent_operations)
    stats["P"] = stats["TP"]/max(stats["TP"]+stats["FP"], 1)
    stats["R"] = stats["TP"]/max(stats["TP"]+stats["FN"], 1)
    stats["F05"] = stats["TP"]/max(stats["TP"]+0.2*stats["FN"]+0.8*stats["FP"], 1)
    stats["sent_acc"] = stats["corr_sents"]/max(stats["total_sents"], 1)
    return stats


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
