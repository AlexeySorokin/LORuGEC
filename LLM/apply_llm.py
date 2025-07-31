from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
import os
import re

import jsonlines
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast
from trl import DataCollatorForCompletionOnlyLM
from sacremoses import MosesDetokenizer
from tqdm.auto import tqdm
import torch

from utils import ChatDatasetFormatter, normalize_edit
from read import read_m2_simple, read_m2_with_labels, read_raw_file
from align import extract_edits, evaluate as evaluate_edits
from edit import apply_edits

def make_fewshot_data(data, database, k=1, use_labels=False, seed=42):
    answer = []
    if use_labels:
        database_labels = defaultdict(list)
        for i, elem in enumerate(database):
            if 'label' in elem:
                database_labels[elem['label']].append(i)
    else:
        database_labels = dict()
    np.random.seed(seed)
    for elem in data:
        label = elem.get('label')
        if label is not None and label in database_labels:
            indexes = database_labels[label]
        else:
            indexes = list(range(len(database)))
        selected_indexes = np.random.choice(indexes, size=k, replace=False)
        if len(selected_indexes) < k:
            selected_indexes += np.random.choice(len(database), size=k-len(selected_indexes))
        examples = [database[index] for index in selected_indexes]
        curr_answer = [{"source_text": example["source_text"], "correct_text": example["correct_text"]} for example in examples]
        answer.append(curr_answer)
    return answer

def make_fewshot_data_from_close_sentences(data, database, close_sentences, k=1, detokenizer=None, seed=42):
    database_dict = {elem["source_text"]: elem["correct_text"] for elem in database}
    close_sentences_dict = dict()
    for elem in close_sentences:
        source = " ".join(elem["sentence"]) if isinstance(elem["sentence"], list) else elem["sentence"]
        targets = elem["closest_sentences"][:]
        if len(targets) > 0 and isinstance(targets[0], list):
            targets = [" ".join(x) for x in targets]
        if detokenizer is not None:
            source = detokenizer(source.split())
            targets = [detokenizer(x.split()) for x in targets]
        targets = [(sent, database_dict[sent]) for sent in targets if sent in database_dict]
        close_sentences_dict[source] = targets
    np.random.seed(seed)
    answer, bad_indexes = [], []
    for i, elem in enumerate(data):
        curr_answer = []
        curr_close_sentences = close_sentences_dict.get(elem["source_text"])
        if curr_close_sentences is not None:
            curr_answer = [{"source_text": source, "correct_text": target} for source, target in curr_close_sentences[:k]]
        else:
            raise ValueError()
        if len(curr_answer) < k:
            bad_indexes.append(i)
        answer.append(curr_answer)
    return answer


argument_parser = ArgumentParser()
argument_parser.add_argument("-i", "--infile", required=True)
argument_parser.add_argument("-r", "--raw", action="store_true")
argument_parser.add_argument("-l", "--label_file", default=None)
argument_parser.add_argument("-D", "--database_file", default=None)
argument_parser.add_argument("-F", "--close_sentences_file", default=None)
argument_parser.add_argument("-L", "--database_label_file", default=None)
argument_parser.add_argument('--use_labels', action="store_true")
argument_parser.add_argument("-n", default=None, type=int)
argument_parser.add_argument("--offset", default=0, type=int)
argument_parser.add_argument("-t", "--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
argument_parser.add_argument("-p", "--prompt_file", required=True)
argument_parser.add_argument("-f", "--modify_prompt_for_fewshot", action="store_true")
argument_parser.add_argument("-m", "--model", default="Qwen/Qwen2.5-1.5B-Instruct")
# argument_parser.add_argument("-A", "--use_adapter", action="store_true")
argument_parser.add_argument("-k", "--fewshot_examples", default=0, type=int)
argument_parser.add_argument('--padding_side', '-P', default=None, choices=["left", "right"])
argument_parser.add_argument("-b", "--batch_size", type=int, default=16)
argument_parser.add_argument("-B", "--beam_size", type=int, default=1)
argument_parser.add_argument("-o", "--output_file", default=None)
argument_parser.add_argument("-O", "--m2_output_file", default=None)
argument_parser.add_argument("-d", "--dump_file", default=None)
argument_parser.add_argument("--output_only_first", action="store_true")
argument_parser.add_argument("-I", "--inspect_only", default=None)
argument_parser.add_argument("--postprocess", action="store_true")


if __name__ == "__main__":
    args = argument_parser.parse_args()
    detokenizer = MosesDetokenizer(lang="ru").detokenize
    if not args.raw:
        data = read_m2_with_labels(args.infile, n=args.n, offset=args.offset, detokenizer=detokenizer, label_file=args.label_file)
    else:
        data = read_raw_file(args.infile, n=args.n, offset=args.offset, detokenizer=detokenizer)
    database = None
    if args.database_file is not None:
        database = read_m2_with_labels(args.database_file, detokenizer=detokenizer, label_file=args.database_label_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, local_files_only=True)
    if args.padding_side is not None:
        tokenizer.padding_side = args.padding_side
    if (isinstance(tokenizer, LlamaTokenizerFast) or "llama" in tokenizer.name_or_path.lower()) and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    with open(args.prompt_file, "r", encoding="utf8") as fin:
        system_prompt = fin.read().strip()
    dataset_formatter = ChatDatasetFormatter(tokenizer, system_prompt)
    if database is not None and args.fewshot_examples > 0:
        if args.close_sentences_file is not None:
            close_sentences = list(jsonlines.open(args.close_sentences_file))
            fewshot_data = make_fewshot_data_from_close_sentences(
                data, database, close_sentences, k=args.fewshot_examples, detokenizer=detokenizer)
        else:
            fewshot_data = make_fewshot_data(data, database, k=args.fewshot_examples, use_labels=args.use_labels)
    else:
        fewshot_data = [None] * len(data)
    dev_dataset = [dataset_formatter(**elem, examples=curr_fewshot_examples, is_train=False) for elem, curr_fewshot_examples in zip(data, fewshot_data)]
    dev_dataset = Dataset.from_list(dev_dataset).with_format('pt')
    possible_response_template_ids = [
        tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False),
        tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False)[2:],
        tokenizer.encode("<s>bot\n", add_special_tokens=False),
        tokenizer.encode("\n[/INST]", add_special_tokens=False)[2:],
        tokenizer.encode(" [/INST]", add_special_tokens=False)[2:],
        tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False),
        tokenizer.encode("\n\n Ассистент:[SEP]", add_special_tokens=False)
    ]
    sample_text = " ".join(f"^{x}$" for x in dev_dataset[0]["input_ids"])
    for response_template_ids in possible_response_template_ids:
        template = " ".join(f"^{x}$" for x in response_template_ids)
        if template in sample_text:
            break
    else:
        raise ValueError("No template found")
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=data_collator)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True).to("cuda")
    answer = []
    model_args = dict(do_sample=False, temperature=1.0, top_p=None, top_k=None, 
                      num_beams=args.beam_size, num_return_sequences=args.beam_size)
    eos_token_id = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []    
    for symbol in ["\n", ".\n", "?\n", "\"\n", "!\n", "<|eot_id|>", "<|im_end|>"]:
        token_ids = tokenizer.encode(symbol)
        if len(token_ids) == 1:
            eos_token_id.append(token_ids[0])
    model_args["eos_token_id"] = eos_token_id
    model_args["pad_token_id"] = tokenizer.pad_token_id
    for batch in tqdm(dataloader):
        batch_size, L = batch["input_ids"].shape
        model_batch = {key: value.to("cuda") for key, value in batch.items() if key in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            model_output = model.generate(**model_batch, max_length=2*L+10, **model_args, 
                                          output_scores=True, output_logits=True, return_dict_in_generate=True)
        for i, input_elem in enumerate(batch["input_ids"]):
            curr_outputs = [tokenizer.decode(output_elem[len(input_elem):], skip_special_tokens=True)
                            for output_elem in model_output.sequences[i*args.beam_size:(i+1)*args.beam_size]]
            answer.append([{"correction": elem} for elem in curr_outputs])
        # for input_elem, output_elem in zip(batch["input_ids"], model_output):
        #     answer.append({"correction": tokenizer.decode(output_elem[len(input_elem):], skip_special_tokens=True)})
    for i, (input_elem, curr_answer) in enumerate(zip(data, answer)):
        for predicted_elem in curr_answer:
            if args.postprocess:
                predicted_elem["correction"] = re.sub(".*here is the corrected text:\s+", "", predicted_elem["correction"]).strip()
                predicted_elem["correction"] = predicted_elem["correction"].replace("«", '"').replace("»", '"')
            edits = extract_edits(input_elem["source"], predicted_elem["correction"], use_starts=True)
            if args.postprocess:
                if len(edits) > 0 and edits[-1].start == edits[-1].end == len(input_elem["source"]):
                    edits.pop()
                edits = [normalize_edit(input_elem["source"], edit) for edit in edits]
                edits = [edit for edit in edits if edit is not None]                
                tokenized_correction = apply_edits(input_elem["source"], edits)
                source_char_length = sum(len(x) for x in input_elem["source"])
                target_char_length = sum(len(x) for x in tokenized_correction)
                if target_char_length > max(source_char_length+20, source_char_length*1.2):
                    edits, tokenized_correction = [], input_elem["source"]
            else:
                tokenized_correction = apply_edits(input_elem["source"], edits)
            predicted_elem["edits"] = edits
            predicted_elem["tokenized_correction"] = tokenized_correction
    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf8") as fout:
            for curr_answer in answer:
                for elem in curr_answer:
                    print(elem["correction"].replace("\n", "#"), file=fout)
                    if args.output_only_first:
                        break
        with open(f"{args.output_file}.tok", "w", encoding="utf8") as fout:
            for curr_answer in answer:
                for elem in curr_answer:
                    print(*(elem["tokenized_correction"]), file=fout)
                    if args.output_only_first:
                        break
    if args.m2_output_file is not None:
        os.makedirs(os.path.dirname(args.m2_output_file), exist_ok=True)
        with open(args.m2_output_file, "w", encoding="utf8") as fout:
            for source_elem, elem in zip(data, answer):
                print(*(source_elem["source"]), file=fout)
                for edit in elem[0]["edits"]:
                    print(edit, file=fout)
                print("", file=fout)
    if not args.raw:
        metrics, diffs = evaluate_edits([x["edits"] for x in data], [x[0]["edits"] for x in answer])
        print(metrics)
        if args.dump_file is not None:
            os.makedirs(os.path.dirname(args.dump_file), exist_ok=True)
            with open(args.dump_file, "w", encoding="utf8") as fout:
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
        