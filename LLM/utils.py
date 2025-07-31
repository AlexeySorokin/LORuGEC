from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import BatchEncoding, EvalPrediction

SRC_TOKEN, TGT_TOKEN, END_TOKEN = "SRC", "TGT", "END"

DEFAULT_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}
{% set offset = 1 %}
{% else %}
{% set offset = 0 %}
{% endif %}
{% for message in messages %}
{% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
{{ raise_exception('Conversation roles must alternate user/bot/user/bot/...') }}
{% endif %}
{% if message['role'] == 'assistant' %}
{% set role = 'bot' %}
{% else %}
{% set role = message['role'] %}
{% endif %}
{{ bos_token + role + '\n' + message['content'] | trim + eos_token }}
{% endfor %}
{% if add_generation_prompt %}
{{ bos_token + 'bot\n' }}
{% endif %}"""

def prepare_sentence(source, tokenizer, target=None, detokenizer=None, is_encoder_decoder=False, **kwargs):
    source = detokenizer(source) if detokenizer is not None else " ".join(source)
    if target is not None:
        target = detokenizer(target) if detokenizer is not None else " ".join(target)
    if is_encoder_decoder:
        raise NotImplementedError()
    else:
        return _prepare_sentence_gpt(source, tokenizer, target, **kwargs)
    
def _prepare_sentence_gpt(source, tokenizer, target=None, strict_target_mask=False,
                          src_token_id=None, tgt_token_id=None, eos_token_id=None):
    first = tokenizer(source, add_special_tokens=False)
    second = tokenizer(target, add_special_tokens=False) if target is not None else None
    input_ids = list(first["input_ids"])
    first_start, first_end = 0, len(input_ids)
    if src_token_id is not None:
        input_ids = [src_token_id] + input_ids
        first_start, first_end = first_start+1, first_end+1
    second_start = first_end
    if tgt_token_id is not None:
        input_ids.append(tgt_token_id)
        second_start += 1 # раньше было second_start, сейчас eos источника входит в target_mask, потому что метка сравнивается со следующей
    second_end = second_start
    if second is not None:
        input_ids.extend(second["input_ids"])
        second_end += len(second["input_ids"])
    all_zeros = np.zeros(shape=(len(input_ids),), dtype=int)
    answer = {
        "input_ids": input_ids, "source_mask": all_zeros.copy(), "target_mask": all_zeros.copy(), 
        # "non_copy_mask": all_zeros.copy()
    }
    answer["source_mask"][first_start:first_end] = 1
    answer["target_mask"][second_start:second_end] = 1
    # answer["non_copy_mask"][second_start:second_end] = make_non_copy_mask(first["input_ids"], second["input_ids"])
    if target is not None and eos_token_id is not None:
        answer["input_ids"].append(eos_token_id)
        answer["source_mask"] = np.concatenate([answer["source_mask"], [0]], axis=0)
        answer["target_mask"] = np.concatenate([answer["target_mask"], [1]], axis=0) # раньше было 1
        # answer["non_copy_mask"] = np.concatenate([answer["non_copy_mask"], [0]], axis=0)
    answer["attention_mask"] = np.ones_like(answer["input_ids"])
    answer["labels"] = answer["input_ids"].copy()
    if strict_target_mask:
        answer["labels"] = np.where(answer["target_mask"], answer["labels"], -100) # -100
    return answer


@dataclass
class LMDataCollator:
    pad_index: int = 0
    pad_to_multiple_of: Optional[int] = None
    padding_side: Optional[str] = "right"
    not_array_keys: Optional[List[str]] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchEncoding:
        LABEL_NAMES = ["label", "labels", "label_ids"]
        batch = dict()
        for key in features[0]:
            if self.not_array_keys is not None and key in self.not_array_keys:
                value = [elem[key] for elem in features]
            else:
                value = [np.array(elem[key]) for elem in features]
                max_length = max(len(x) for x in value)
                if self.pad_to_multiple_of is not None:
                    max_length += self.pad_to_multiple_of - 1
                    max_length -= max_length % self.pad_to_multiple_of
                for i, elem in enumerate(value):
                    diff = max_length - len(elem)
                    if diff > 0:
                        padding = np.tile(np.full_like(elem[:1], -100 if key in LABEL_NAMES else self.pad_index), diff)
                        to_concatenate = [padding, elem] if self.padding_side == "left" else [elem, padding]
                        value[i] = np.concatenate(to_concatenate, axis=0)
            batch[key] = value
        batch = BatchEncoding(batch, tensor_type="pt")
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch
    

class ChatDatasetFormatter:

    NO_SYSTEM_PROMPT_TOKENIZERS = ["mistral", "yandex"]

    def __init__(self, tokenizer, prompt=None, use_system_prompt=True, use_fewshot_dialogue_format=True, 
                 use_assistant_format=True, source_prefix="", target_prefix=""):
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.use_system_prompt = use_system_prompt
        assert use_assistant_format >= use_fewshot_dialogue_format
        self.use_fewshot_dialogue_format = use_fewshot_dialogue_format
        self.use_assistant_format = use_assistant_format
        self.source_prefix = source_prefix
        self.target_prefix = target_prefix

    def __call__(self, source_text, correct_text="", examples=None, is_train=True, **kwargs):
        examples = examples or []
        if not is_train:
            correct_text = ""
        prefix = f"{self.prompt}\n\n" if self.prompt else ""
        messages = []
        if self.prompt is not None:
            if self.use_system_prompt and all(x not in self.tokenizer.name_or_path for x in self.NO_SYSTEM_PROMPT_TOKENIZERS):
                prefix = ""
                messages.append({"role": "system", "content": self.prompt})
        if self.use_fewshot_dialogue_format:
            for example in examples:
                messages.append({"role": "user", "content": prefix+self.source_prefix+example['source_text']})
                messages.append({"role": "assistant", "content": self.target_prefix+example['correct_text']})
                prefix = ""  
            # if self.is_train:
            #     messages.append({"role": "assistant", "content": self.target_prefix+correct_text})
            # else:
            #     messages.append({"role": "assistant", "content": self.target_prefix})
        else:
            for example in examples:
                prefix += self.source_prefix+example['source_text']+"\n"
                prefix += self.target_prefix+example['correct_text']+"\n"
        source_text = prefix+self.source_prefix+source_text
        add_generation_prompt, continue_final_message = False, not(is_train or self.use_assistant_format)
        if self.use_assistant_format:
            messages.append({"role": "user", "content": source_text})
            if is_train or self.target_prefix:
                messages.append({"role": "assistant", "content": (self.target_prefix+correct_text).rstrip()})
            else:
                add_generation_prompt = True
        else:
            source_text+="\n"+(self.target_prefix+correct_text).rstrip()
            messages.append({"role": "user", "content": source_text})
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, 
                                                        continue_final_message=continue_final_message)
        if is_train and messages[-1]["role"] == "assistant":
            default_input_text = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=add_generation_prompt, 
                continue_final_message=continue_final_message)
            if input_text == default_input_text:
                input_text += " "+messages[-1]["content"]        
        if self.prompt is not None:
            assert self.prompt in input_text
        answer = self.tokenizer(input_text)
        return answer
    
def normalize_edit(source, edit):
    edit_source = " ".join(source[edit.start:edit.end])
    if len(edit_source) == 1 and len(edit.candidate) == 1:
        if 8210 <= ord(edit_source) <= 8212 and 8210 <= ord(edit.candidate) <= 8212:
            return None   
    if len(edit.candidate) == 1 and ord(edit.candidate) in [8211, 8212]:
        edit.candidate = chr(8210)
    if edit_source == edit.candidate:
        return None
    return edit