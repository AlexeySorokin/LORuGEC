This repository contains code for applying GECTOR-like sequence labelers to retrieve similar sentences from the training GEC corpus.

## Retrievers trained via HF

```bash
python retrieval_inference_gector.py -m MODEL_ID -t TOKENIZER_ID -H -d DATABASE_M2_FILE -D TEST_FILE [-r] -o OUTPUT_PATH -K KEEP_TOKEN [--cosine]
```
* Add `-r` when the TEST_FILE contains one sentence per line, otherwise the test file should be also in .M2 format.
* Add `--cosine` to use cosine similarity (recommended) instead of L2 one.
* `KEEP_TOKEN` is the `NO_EDIT` label of the GECTOR ($KEEP for the original implementation).

For example, use the following command to apply our reimplementation of GECTOR model trained on cLang8 corpus.

**Example: TBD**


## Custom retriever (for Russian)

In our work **[Regina Nasyrova, Alexey Sorokin Grammatical Error Correction via Sequence Tagging for Russian](https://aclanthology.org/2025.acl-srw.82.pdf)** we use a custom implementation of GECTOR approach. 

**Instruction: TBD**.

## Original implementation of GECToR

Unfortunately, it is impossible to combine our code with original implementation of GECTOR.