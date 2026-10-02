# Lelapa AI Buzuzu-Mavi Challenge

Reconstructed research notebook for the Lelapa AI Buzuzu-Mavi Challenge. It fine-tunes [InkubaLM-0.4B](https://huggingface.co/lelapa/InkubaLM-0.4B) with separate LoRA adapters for sentiment analysis, AfriXNLI, and machine translation across Hausa and Swahili. It also includes an optional three-class sentiment classifier and an experimental BLEU-reward translation stage.

> **Reproducibility status:** The notebook was reconstructed from an earlier competition notebook. Code cells were checked for Python syntax, but training, inference, and challenge submissions have **not** been run against the supplied datasets or evaluated for accuracy. The dataset and model checkpoints are not included in this repository.

## Repository contents

| File | Purpose |
| --- | --- |
| [`Lelapa AI.ipynb`](Lelapa%20AI.ipynb) | Setup, data preparation, model initialization, training, evaluation, and submission generation. |
| `README.md` | Setup instructions, task definitions, and implementation notes. |

## Tasks and models

| Task | Languages | Default model | Output |
| --- | --- | --- | --- |
| Sentiment | Hausa, Swahili | InkubaLM causal LM + sentiment LoRA adapter | Class ID: `0` positive, `1` neutral, `2` negative |
| AfriXNLI | Hausa (`hau`), Swahili (`swa`) | InkubaLM causal LM + XNLI LoRA adapter | Class ID: `0` neither, `1` true, `2` false |
| Translation | English → Hausa, English → Swahili | InkubaLM causal LM + translation LoRA adapter | Generated translation string |

Sentiment has an optional alternative `AutoModelForSequenceClassification` baseline with three output classes. The optional BLEU-reward stage starts from the supervised translation adapter and saves a separate experimental adapter. The source notebook contained a conflicting sentiment label heading; the explicit training mappings above were used. **Verify class IDs against the competition data dictionary before submitting.**

## Requirements

- Python 3 with a Jupyter kernel; a CUDA GPU is recommended for fine-tuning.
- The challenge Parquet datasets in the paths below.
- Access to the InkubaLM model. If its repository requires execution of custom modeling code, review that code before setting `TRUST_REMOTE_CODE = True` in the notebook.
- Dependencies used by the notebook:

```bash
python -m pip install "transformers==4.45.2" "peft==0.13.2" "datasets==3.0.2" "accelerate==1.0.1" pyarrow sacrebleu pandas numpy torch jupyter
```

The notebook does not store credentials. If model access requires authentication, provide `HF_TOKEN` as an environment variable or secure notebook secret. Do not commit credentials or generated checkpoints.

## Data layout

Set `LELAPA_DATA_DIR` to the parent of these directories (default: `./data`):

```text
data/
├── Sentiment Analysis/
│   ├── Sentiment Analysis_train.parquet
│   └── Sentiment Analysis_test.parquet
├── AfriXNLI/
│   ├── AfriXNLI_train.parquet
│   └── AfriXNLI_test.parquet
└── Machine translation/
    ├── MT_train.parquet
    └── MT_test.parquet
```

The notebook expects `ID`, `langs`, `inputs`, `instruction`, and `targets` in training files. AfriXNLI also needs `premise`; test files need the input fields and may contain missing targets. Sentiment uses `hausa`/`swahili`; XNLI uses `hau`/`swa`; translation uses `eng-hau`/`eng-swa`.

## Run the notebook

1. Supply the dataset files and install the dependencies above.
2. Configure the data location and output location, for example:

   ```bash
   export LELAPA_DATA_DIR="$(pwd)/data"
   export LELAPA_OUTPUT_DIR="$(pwd)/output"
   jupyter notebook "Lelapa AI.ipynb"
   ```

3. Run the setup and preprocessing definition cells in order. Review a few source rows to verify the label conventions and language codes.
4. In section 4, uncomment the training loop to initialize, fine-tune, and save all three adapters:

   ```python
   for task_name in ("sentiment", "xnli", "mt"):
       train_task(task_name, tokenizer)
   ```

5. If you have a separate labeled development set, use `evaluate_labeled(task, dev_frame, tokenizer)` to assess classification accuracy or corpus BLEU. Avoid treating metrics from training rows as held-out performance.
6. Run `make_submission(tokenizer, split="test")` to generate `output/submission.csv` with columns `ID,Response`. It requires the three saved task adapters. To use the optional trained sentiment classifier instead, pass `sentiment_backend="classifier"`.

The default adapter paths are `output/sentiment_adapter`, `output/xnli_adapter`, and `output/mt_adapter`. Each task starts from a fresh base model before its adapter is trained; Hausa and Swahili examples for a task share that task's adapter. The initial epochs and learning rates reproduce the source notebook's intended settings and can be adjusted in `TRAIN_SETTINGS`.

## Implementation details

- Supervised examples mask prompt tokens with `-100` so the causal LM loss applies to the response. A custom collator pads sequences and labels independently.
- Sentiment and XNLI predictions compare log likelihoods of the full candidate response strings, instead of relying on fixed token IDs or parsing free-form generated answers.
- Translation predictions use deterministic generation and decode only the newly generated tokens.
- The optional sequence classifier has independent training, checkpoint loading, and logits-based inference.
- The BLEU experiment uses sampled sequence log probabilities with a greedy BLEU baseline as a reward signal. It is opt-in and does **not** automatically replace the supervised translation checkpoint in submission generation.

## Limitations

Model compatibility, memory usage, label correctness, translation quality, and submission validity require testing in an environment with the datasets and checkpoints. The original notebook depended on custom model code and local Google Drive paths; this version uses configurable directories and explicit model initialization. The README documents the notebook as written, not measured competition results.
