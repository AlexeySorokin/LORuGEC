## Inference

1. Basic zeroshot inference

```bash
python apply_llm.py -m MODEL_PATH [-t TOKENIZER] -i TEST_FILE [-r] -p PROMPT_FILE -b BATCH_SIZE -P left [-d DUMP_FILE_PATH] [-o OUTPUT_FILE_PATH] [-O M2_OUTPUT_FILE_PATH] --postprocess
```

If no tokenizer is provided, it is assumed that tokenizer has the same name as the model. By default, .m2 files are expected as input, use `-r` to work with pretokenized text files (one word per line).

2. Fewshot inference with a precomputed closest sentences file

```bash
python apply_llm.py -m MODEL_PATH [-t TOKENIZER] -i TEST_FILE [-r] -D TRAIN_FILE -F DATABASE_FILE -p PROMPT_FILE -k K -b BATCH_SIZE -P left [-d DUMP_FILE_PATH] [-o OUTPUT_FILE_PATH] [-O M2_OUTPUT_FILE_PATH] --postprocess
```

For example, use the following command to run 5-shot inference on LORuGEC test data with our best sentence retrieval method:
```bash
python apply_llm.py -m Qwen/Qwen2.5-7B-Instruct -i LORUGEC_TEST_PATH -D LORUGEC_TRAIN_PATH 
-F data/samples/similar_sentences/lorugec_test_gector_ft -k 5 -p data/prompts/prompt_Russian -b 32 -P left -o OUTPUT_PATH --postprocess
```
