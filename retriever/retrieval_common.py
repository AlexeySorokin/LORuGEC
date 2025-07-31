from collections import Counter, defaultdict
from random import Random
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader

from collate import triple_collate_fn
from evaluation import measure_metrics, measure_metrics_multilabel

def read_sentences_file(infile, text_column=None, detokenizer=None, sep=None):
    with open(infile, "r", encoding="utf8") as fin:
        answer = []
        for line in fin:
            line = line.strip()
            if line == "":
                continue
            sentence, labels = line.split("\t")
            labels = labels.split(sep) if sep is not None else labels # поправить
            text = sentence.strip()
            sample = {"words": text.split(), "class_label": labels}
            if text_column is not None:
                if detokenizer is not None:
                    text = detokenizer(text.split())
                sample[text_column] = text
            answer.append(sample)
    return answer


def prepare_train_valid_split(data, per_class_valid_frac=0.0, class_valid_frac=0.0, random_state=189, class_label_field="class_label"):
    labels = [item[class_label_field] for item in data]
    random_generator = Random(random_state)
    classes_counts = Counter(labels)
    classes = list(classes_counts.keys())
    random_generator.shuffle(classes)
    if class_valid_frac > 0.0:
        n_train_classes = min(int(len(classes)*(1.0-class_valid_frac)), len(classes)-1)
        train_classes, test_classes = classes[:n_train_classes], classes[n_train_classes:]
        print(test_classes, sep="\n")
    else:
        train_classes, test_classes = classes, []
    are_indexes_test = [(label in test_classes) for label in labels]
    if per_class_valid_frac > 0.0:
        order = list(range(len(data)))
        random_generator.shuffle(order)
        are_indexes_test = [False] * len(data)
        train_classes_counts = {label: 0 for label in train_classes}
        for index in order:
            label = labels[index]
            if label in train_classes and train_classes_counts[label] < classes_counts[label]*per_class_valid_frac:
                are_indexes_test[index] = True
                train_classes_counts[label] += 1
    train_data = [elem for elem, flag in zip(data, are_indexes_test) if not flag]
    test_data = [elem for elem, flag in zip(data, are_indexes_test) if flag]
    return train_data, test_data


class LabelBasedShufflerAndBucketer:

    def __init__(self, dataset, bucket_size=None, label_field='class_label', 
                 from_labels_list=False, min_count=1, max_label_bucket_size=None, seed=42):
        self.dataset = dataset
        self.bucket_size = bucket_size
        self.label_field = label_field
        self.from_labels_list = from_labels_list
        self.min_count = min_count
        self.max_label_bucket_size = max_label_bucket_size
        self.shuffler = np.random.default_rng(seed=seed)

    def __iter__(self):
        if self.bucket_size is None:
            yield self.dataset
        item_labels = [item[self.label_field] for item in self.dataset]
        label_counts = Counter(item_labels)
        label_counts = {label: count for label, count in label_counts.items() if count >= self.min_count}
        labels_order = []
        label_buckets_number = 1
        for label, count in label_counts.items():
            if self.max_label_bucket_size is not None:
                label_buckets_number = 1+int((count-1)//self.max_label_bucket_size)
            labels_order.extend([label]*label_buckets_number)
        self.shuffler.shuffle(labels_order)
        indexes_by_labels = defaultdict(list)
        for i, label in enumerate(item_labels):
            indexes_by_labels[label].append(i)
        for label in indexes_by_labels:
            self.shuffler.shuffle(indexes_by_labels[label])
        curr_bucket_indexes = []
        label_offsets = {label: 0 for label in label_counts}
        for label in labels_order:
            label_offset = label_offsets[label]
            if self.max_label_bucket_size is not None:
                curr_block_size = min(label_counts[label]-label_offsets[label], self.max_label_bucket_size)
            else:
                curr_block_size = label_counts[label]
            curr_bucket_indexes.extend(indexes_by_labels[label][label_offset:label_offset+curr_block_size])
            label_offsets[label] += curr_block_size
            if self.bucket_size is not None and len(curr_bucket_indexes) >= self.bucket_size:
                curr_dataset = self.dataset.select(curr_bucket_indexes)
                yield curr_dataset
                curr_bucket_indexes = []
        if len(curr_bucket_indexes) > 0:
            curr_dataset = self.dataset.select(curr_bucket_indexes)
            yield curr_dataset

def make_retrieval_train_dataset(data, indexes, variant_keys=None, common_keys=None):
    if variant_keys is None:
        variant_keys = ["input_ids", "token_type_ids", "left_mask", "right_mask", "mean_mask", "mask", "labels_mask"] # , "labels", "error_type"
    if common_keys is None:
        common_keys = ['position_query_positive', 'position_query_negative', 'position_positive', 'position_negative']
    answer = []
    for elem in indexes:
        sample = dict()
        for suffix in ["", "_positive", "_negative"]:
            sample.update({key+suffix: data[int(elem["index"+suffix])][key] for key in variant_keys})
        for key in common_keys:
            sample[key] = elem[key]
        answer.append(sample)
    return answer

def train_retriever(model, dataset, tokenizer, num_epochs=1, batch_size=8, lr=1e-6, 
                    optimizer=None, random_state=127, 
                    collate_func=None, variant_keys=None, common_keys=None):
    np.set_printoptions(precision=3)
    g = torch.Generator()
    g.manual_seed(random_state)
    collate_fn = partial(triple_collate_fn, collate_func=collate_func, variant_keys=variant_keys, common_keys=common_keys)
    retrieval_collate_fn = partial(collate_fn, bert_tokenizer=tokenizer, device="cuda")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=retrieval_collate_fn, generator=g)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for n in range(num_epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch_output = model(**batch)
            positive_distances = batch_output["positive_distances"].detach().cpu().numpy()
            negative_distances = batch_output["negative_distances"].detach().cpu().numpy()
            loss = batch_output["loss"]
            loss.backward()
            optimizer.step()
            accuracy = np.mean(positive_distances < negative_distances)
            # print(positive_distances, negative_distances)
            # print(n, i, loss.item(), accuracy)
    return

def evaluate_retrieval(test_labels, pred_labels, evaluate_per_class=False, verbose=True, multilabel=False):
    func = measure_metrics_multilabel if multilabel else measure_metrics
    metrics = func(test_labels, pred_labels, d=5, evaluate_per_class=evaluate_per_class)
    if verbose:
        for key, value in metrics.items():
            if "per_class" in key:
                if not evaluate_per_class:
                    continue
                print("")
                for subkey, subkey_data in value.items():
                    print(subkey, end="\t")
                    if isinstance(subkey_data, dict):
                        print("\t".join(f"{x}:{100*y:.2f}" if x != "total" else f"{x}:{y}" for x, y in subkey_data.items()))
                    else:
                        print(subkey_data)
            else:
                print(f"{key} {100*value:.2f}", end="\t")
        print("")
    return metrics