from functools import partial
import torch
import numpy as np


def default_collate_fn(sample, tokenizer):
    L = max(len(elem["input_ids"] for elem in sample))
    dtype, device = sample[0]["input_ids"].dtype, sample[0]["input_ids"].device
    answer = dict()
    answer["input_ids"] = torch.stack([
            torch.cat([
                elem["input_ids"], torch.ones(size=(L-len(elem["input_ids"]),), dtype=dtype, device=device)*(tokenizer.pad_token_id)
            ]) for elem in sample
        ]).to(device)
    if "attention_mask" in sample[0]:
        answer["attention_mask"] = torch.stack([
            torch.cat([
                elem["attention_mask"], torch.zeros(size=(L-len(elem["attention_mask"]),), dtype=int, device=device)
            ]) for elem in sample
        ]).to(device)
    return answer

def collate_fn(sample, bert_tokenizer, dtype=torch.int64, device="cpu"):
    """
    Дополняет input_ids, mask паддингом до макс. длины input_ids
    Дополняет маски индексов, меток, типы токенов, метки до макс. длины меток
    """
    labels_lengths, tokens_lengths = [], []
    for elem in sample:
        for key in ["words", "labels", "labels_mask"]:
            if key in elem:
                labels_lengths.append(len(elem[key])) # elem["labels"] может не быть
                break
        tokens_lengths.append(len(elem["input_ids"]))
    answer = dict()
    answer["input_ids"] = torch.stack([
            torch.cat([
                elem["input_ids"].to(device),
                torch.ones(size=(max(tokens_lengths)-len(elem["input_ids"]),), dtype=dtype, device=device)*(bert_tokenizer.pad_token_id)
            ]) for elem in sample
        ]).to(device)
    if "attention_mask" in sample[0]:
        answer["attention_mask"] = torch.stack([
            torch.cat([
                elem["attention_mask"].to(device),
                torch.zeros(size=(max(tokens_lengths)-len(elem["attention_mask"]),), dtype=torch.bool, device=device)
            ]) for elem in sample
        ]).to(device)
    if "mean_mask" in sample[0]:
        answer["mean_mask"] = torch.stack([
            torch.cat([
                elem["mean_mask"].to(device),
                torch.ones(size=(max(tokens_lengths)-len(elem["mean_mask"]),), dtype=dtype, device=device)*(-100)
            ]) for elem in sample
        ]).to(device)
    for key in ["labels", "left_mask", "right_mask"]:
        if key in elem:
            answer[key] = torch.stack([
                torch.cat([
                    elem[key].to(device),
                    torch.ones(size=(max(labels_lengths)-len(elem[key]),), dtype=dtype, device=device)*(-100)
                ]) for elem in sample
            ]).to(device)
    if "token_type_ids" in sample[0]:
        answer["token_type_ids"] = torch.stack([
            torch.cat([
                elem["token_type_ids"].to(device),
                torch.ones(size=(max(labels_lengths)-len(elem["token_type_ids"]),), dtype=dtype, device=device)*(2)
            ]) for elem in sample
        ]).to(device)
    if "error_type" in answer:
        answer["error_type"] = np.stack([
                np.concatenate([
                    elem["error_type"].to(device),
                    np.zeros(shape=(max(labels_lengths)-len(elem["error_type"]),), dtype=bool, device=device)
                ]) for elem in sample
            ])
    if "labels_mask" in elem:
        answer["labels_mask"] = torch.stack([
                    torch.cat([
                        elem["labels_mask"].to(device),
                        torch.zeros(size=(max(labels_lengths)-len(elem["labels_mask"]),), dtype=torch.bool, device=device)
                    ]) for elem in sample
                ]).to(device)
    if "mask" in sample[0]:
        answer["mask"] = torch.stack([
            torch.cat([
                elem["mask"].to(device),
                torch.zeros(size=(max(tokens_lengths)-len(elem["mask"]),), dtype=torch.bool, device=device)
            ]) for elem in sample
        ]).to(device)
    return answer


def triple_collate_fn(sample, bert_tokenizer, collate_func=None, variant_keys=None, common_keys=None, dtype=torch.int64, device="cpu"):
    if collate_func is None:
        collate_func = partial(collate_fn, bert_tokenizer=bert_tokenizer, dtype=dtype, device=device)
    if variant_keys is None:
        variant_keys = ["input_ids", "token_type_ids", "left_mask", "right_mask", "mean_mask", "mask", "labels", "labels_mask", "error_type"]
    if common_keys is None:
        common_keys = ['position_query_positive', 'position_query_negative', 'position_positive', 'position_negative']
    SUFFIXES = ["", "_positive", "_negative"]
    active_variant_keys = [key for key in variant_keys if all(key+suffix in sample[0] for suffix in SUFFIXES)]
    common_keys = [key for key in common_keys if key in sample[0]]
    flat_sample = []
    for elem in sample:
        for suffix in SUFFIXES:
            flat_sample.append({key: elem[key+suffix] for key in active_variant_keys})
    answer = collate_func(flat_sample)
    for suffix in common_keys:
        answer[suffix] = torch.LongTensor([elem[suffix] for elem in sample]).to(device)
    return answer