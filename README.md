# MaiBhoj-NLP

A multi-task benchmark for **Maithili** and **Bhojpuri**, two low-resource Indo-Aryan languages. It covers news headline generation, abstractive summarization, emotion classification and sentiment classification. It also an **Affective Preservation** framework, which measures whether generated headlines and summaries keep the emotion and sentiment of the source article.

> Paper: A Multi-task Benchmark for Maithili and Bhojpuri NLP, Deepak Prakash et al., IIT Patna (under review at IEEE Access).

## Dataset

| | Bhojpuri | Maithili | Total |
|---|---|---|---|
| Full corpus (machine-annotated, human-corrected) | | | 23,629 |
| Gold experimental subset | 2,178 | 2,010 | 4,188 |
| Test split | 218 | 201 | 419 |

Each article has four annotations:

- a headline
- an abstractive summary
- a dominant emotion: Ekman's six basic emotions plus *neutral*
- a sentiment label: positive, negative or neutral

`gpt-4o-mini` drafted the annotations, and native speakers then reviewed and corrected every article. Inter-annotator agreement was measured on 100 articles per language. The gold subset is split 80/10/10 with random seed 42.

## Tasks and models

- **Generation:** headline generation and abstractive summarization.
- **Zero-shot LLM baselines:** Airavata, Sarvam-1, OpenHathi, Llama-3.1-8B-Instruct, Gemma-2-9B-IT.
- **IndicBART:** pretrained (no task training), and fine-tuned separately for each language and task.
- **Affect classifiers (zero-shot):** GPT-OSS-20B, GPT-OSS-120B, Gemma-3-27B, Qwen3-32B, LLaMA-3.1-8B.

All generation systems use the same test split and decoding settings: beam search with 4 beams, length penalty 1.0, no repeated trigrams, and at most 64 / 200 new tokens for headlines / summaries.

## Main results (ROUGE-1)

| System | Mai headline | Bho headline | Mai summary | Bho summary |
|---|---|---|---|---|
| Best zero-shot LLM | 0.045 | 0.111 | 0.144 | 0.259 |
| Pretrained IndicBART | 0.138 | 0.204 | 0.332 | 0.525 |
| Fine-tuned IndicBART | **0.156** | **0.247** | **0.378** | **0.525** |

Fine-tuned IndicBART scores higher than every zero-shot LLM in all four settings, and the 95% bootstrap confidence intervals do not overlap. It cannot be statistically separated from pretrained IndicBART on Maithili headlines or Bhojpuri summaries. Pretrained IndicBART largely copies the opening of the article, and native-speaker evaluators preferred the fine-tuned outputs. Full results, including BLEU, ROUGE-2/L, BERTScore and the human evaluation, are in the paper.

## Affective Preservation

Each generated output is classified by the five LLM classifiers. Their predictions are then compared with the source article's gold emotion and sentiment labels, both with Cohen's κ and relative to how well the same classifiers do on the source articles themselves. Summaries keep more of the source's affect than headlines. This should be read as an association, not a causal effect, because summaries are also longer.

## Repository structure

```
data/        # train / validation / test splits for each language
prompts/     # zero-shot prompts for each language and task
finetuned/     # inference, fine-tuning and evaluation
zeroshot/     # models predictions on the test sets

## Limitations

- The labels are human-corrected machine drafts. Annotators saw the draft while reviewing, so their labels may be anchored to it.
- Affect preservation is measured with LLM classifiers. No human annotation of affect was done on the generated outputs.
- The results apply only to the models evaluated above.


Deepak Prakash, deepak_2321cs13@iitp.ac.in
