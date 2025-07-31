from argparse import ArgumentParser
from functools import partial
import os
import json

from sacremoses import MosesDetokenizer
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from tqdm import tqdm

from read import read_m2_simple, read_raw_file
from utils import HuggingfaceModelWrapper
from retrieval_utils import prepare_storage, predict_closest_sentences


def make_subtoken_mask(mask, mode="first", has_cls=True, has_eos=True, is_cls_word=False):
    if has_cls:
        mask = mask[1:]
    if has_eos:
        mask = mask[:-1]
    is_word = list((first != second) for first, second in zip(mask[1:], mask[:-1]))
    is_word = ([True] + is_word) if mode == "first" else (is_word+[True])
    if has_cls:
        is_word = [is_cls_word] + is_word
    if has_eos:
        is_word.append(False)
    return is_word

class SequenceDatasetCreator:

    def __init__(self, tokenizer, to_tensors=False, aggregation_type="first", token_field="tokens", label_field="labels"):
        self.tokenizer = tokenizer
        self.to_tensors = to_tensors
        self.aggregation_type = aggregation_type
        self.token_field = token_field
        self.label_field = label_field

    @property
    def has_eos(self):
        return self.tokenizer.eos_token_id is not None

    def __call__(self, item):
        tokenization = self.tokenizer(item[self.token_field], is_split_into_words=True)
        subtoken_mask = make_subtoken_mask(tokenization.word_ids(), mode=self.aggregation_type, 
                                           has_eos=self.has_eos, is_cls_word=True)
        assert sum(int(x) for x in subtoken_mask) == len(item[self.token_field])+1
        # return {"input_ids": tokenization["input_ids"], "attention_mask": tokenization["attention_mask"],
        #         "mask": subtoken_mask, **item}
        item.update(tokenization)
        item["mask"] = subtoken_mask
        return item
    


argument_parser = ArgumentParser()
argument_parser.add_argument('-s', "--seed", default=5, type=int)
argument_parser.add_argument('-d', "--data", required=True)
argument_parser.add_argument("-n", default=None, type=int)
argument_parser.add_argument('-D', "--test_data", required=True)
argument_parser.add_argument('-o', '--outfile', required=True)
argument_parser.add_argument('-t', '--tokenizer', default="roberta-base")
argument_parser.add_argument('-m', "--model_name", default="roberta-base")
argument_parser.add_argument("--labels_file")
argument_parser.add_argument('-H', "--from_hf", action="store_true")
argument_parser.add_argument('-K', '--keep_token', default="KEEP")
argument_parser.add_argument('-G', "--aggregation_type", default="first")
argument_parser.add_argument('-L', "--language", default="ru")
argument_parser.add_argument('-k', default=10, type=int)
argument_parser.add_argument('--max_non_keep_per_text', type=int, default=3)
argument_parser.add_argument('--cosine', action="store_true")



if __name__ == "__main__":
    args = argument_parser.parse_args()
    detokenizer = MosesDetokenizer(lang=args.language).detokenize    
    if not args.raw:
        data = read_m2_simple(args.infile, n=args.n, offset=args.offset, detokenizer=detokenizer, label_file=args.label_file)
    else:
        data = read_raw_file(args.infile, n=args.n, offset=args.offset, detokenizer=detokenizer)
    data = Dataset.from_list(data)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, add_prefix_space=True)
    dataset_creator = SequenceDatasetCreator(tokenizer=tokenizer, aggregation_type=args.aggregation_type, token_field="source")
    data = data.map(dataset_creator).filter(lambda x: (len(x["input_ids"]) <= 512)).with_format('pt')
    test_data = read_m2_simple(args.test_data, detokenizer=detokenizer, save_edits=False)
    test_data = Dataset.from_list(test_data)
    test_data = test_data.map(dataset_creator).with_format('pt')
    if args.from_hf:
        model = HuggingfaceModelWrapper.from_pretrained(args.model_name).to("cuda")
        labels2id = model.config.label2id
    else:
        with open(args.labels_file, "r", encoding="utf8") as fin:
            labels2id = json.load(fin)
        model = HuggingfaceModelWrapper.from_pretrained(args.model_name, num_labels=len(labels2id)).to("cuda")
    columns_to_select = ["input_ids", "attention_mask", "mask"]
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    vectors_storage, sentences_storage = prepare_storage(model, data, # data.select_columns(columns_to_select),
                                                         tokenizer, keep_index=labels2id[args.keep_token],
                                                         cosine=args.cosine, max_non_keep_per_text=args.max_non_keep_per_text,
                                                         text_column="source", label_column=None, labels_mask_column="mask")
    answer = predict_closest_sentences(model, test_data, vectors_storage, sentences_storage, tokenizer, 
                                       keep_index=labels2id[args.keep_token], cosine=args.cosine,
                                       text_column="source", labels_mask_column="mask")
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w", encoding="utf8") as fout:
        for elem, answer_elem in zip(tqdm(test_data), answer):
            closest_sentences = [elem[0] for elem in answer_elem]
            curr_answer = {"sentence": elem['source'], "closest_sentences": closest_sentences}
            print(json.dumps(curr_answer, ensure_ascii=False), file=fout)
