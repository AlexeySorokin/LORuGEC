This is the code repo for our paper **Alexey Sorokin, Regina Nasyrova LLMs in alliance with Edit-based Models: Advancing In-Context Learning for Grammatical Error Correction by Specific Example Selection**, accepted to 20th Workshop on Innovative Use of NLP for Building Educational Applications ([BEA 2025](https://sig-edu.org/bea/2025)). 

In this paper we show that GECTOR-like models might be used to improve fewshot example selection for grammatical error correction.

The code contains two folders:
* `LLM` -- the implementation of few-shot LLM inference on GEC data.
* `retriever` -- the implementation of similar examples retrieval using GECTOR-like models.

See the `readme.md` files in subfolders for the detailed instructions.

## Installation

**TBD**

## Citing

*  Alexey Sorokin, Regina Nasyrova [LLMs in alliance with Edit-based Models: Advancing In-Context Learning for Grammatical Error Correction by Specific Example Selection](https://aclanthology.org/anthology-files/pdf/bea/2025.bea-1.38)
```
@inproceedings{sorokin-nasyrova-2025-llms,
    title = "{LLM}s in alliance with Edit-based models: advancing In-Context Learning for Grammatical Error Correction by Specific Example Selection",
    author = "Sorokin, Alexey  and
      Nasyrova, Regina",
    editor = {Kochmar, Ekaterina  and
      Alhafni, Bashar  and
      Bexte, Marie  and
      Burstein, Jill  and
      Horbach, Andrea  and
      Laarmann-Quante, Ronja  and
      Tack, Ana{\"i}s  and
      Yaneva, Victoria  and
      Yuan, Zheng},
    booktitle = "Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.bea-1.38/",
    pages = "517--534",
    ISBN = "979-8-89176-270-1",
    abstract = "We release LORuGEC {--} the first rule-annotated corpus for Russian Grammatical Error Correction. The corpus is designed for diagnostic purposes and contains 348 validation and 612 test sentences specially selected to represent complex rules of Russian writing. This makes our corpus significantly different from other Russian GEC corpora. We apply several large language models and approaches to our corpus, the best F0.5 score of 83{\%} is achieved by 5-shot learning using YandexGPT-5 Pro model.To move further the boundaries of few-shot learning, we are the first to apply a GECTOR-like encoder model for similar examples retrieval. GECTOR-based example selection significantly boosts few-shot performance. This result is true not only for LORuGEC but for other Russian GEC corpora as well. On LORuGEC, the GECTOR-based retriever might be further improved using contrastive tuning on the task of rule label prediction. All these results hold for a broad class of large language models."
}
```
If you use our GECToR-like model, cite also

*  Regina Nasyrova, Alexey Sorokin [Grammatical Error Correction via Sequence Tagging for Russian](https://aclanthology.org/2025.acl-srw.82)
```
@inproceedings{nasyrova-sorokin-2025-grammatical,
    title = "Grammatical Error Correction via Sequence Tagging for {R}ussian",
    author = "Nasyrova, Regina  and
      Sorokin, Alexey",
    editor = "Zhao, Jin  and
      Wang, Mingyang  and
      Liu, Zhu",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-srw.82/",
    pages = "1036--1050",
    ISBN = "979-8-89176-254-1",
    abstract = "We introduce a modified sequence tagging architecture, proposed in (Omelianchuk et al., 2020), for the Grammatical Error Correction of the Russian language. We propose language-specific operation set and preprocessing algorithm as well as a classification scheme which makes distinct predictions for insertions and other operations. The best versions of our models outperform previous approaches and set new SOTA on the two Russian GEC benchmarks {--} RU-Lang8 and GERA, while achieve competitive performance on RULEC-GEC."
}
```


## See also

* [LORuGEC corpus paper and repo](https://github.com/ReginaNasyrova/LORuGEC)
* [Our adaptation of GECToR approach for Russian language](https://github.com/ReginaNasyrova/RussianGEC_SeqTagger)