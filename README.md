# Industry classification — final project

**Verified assignment validation score: 67.50%.** The final financial-calibrated ensemble obtains **78.09% classification weighted F1** and **0.135585 Brier** on **3,171 validation records**. The notebooks now contain genuine outputs from the supplied course datasets.

## Evaluation metric

We evaluated confidence using `confidence_score`:

**assignment score = weighted F1 × (1 − Brier)**

For this run: `0.780881 × (1 − 0.135585) = 0.675006`, or **67.50%**. This is printed under **MAIN GRADED SCORE (LOCAL VALIDATION)**. Weighted F1 measures predicted industries; Brier measures how well confidence matches correctness. Confidence is not an F1 sample weight.

The other numbers answer different questions:

| Metric/output | Purpose |
|---|---|
| **67.50% assignment score** | Confidence-weighted validation metric for the final model. |
| **78.09% weighted F1** | Classification component of the final ensemble's score. |
| Macro F1 and accuracy | Additional diagnostics, including minority-class performance. |
| LR/k-NN comparison | Mandatory comparison of the two required classifiers. |
| Original skeleton example | Scoring illustration; separate from the final candidate. |

## Files

| File | Purpose |
|---|---|
| `industry_classification.ipynb` | **Clean submission notebook**, executed with real outputs and concise findings. Completed report using the original skeleton titles; exact scorer and example retained. |
| `industry_classification_commented.ipynb` | Same executable code, independently executed, with English comments and study notes. |
| `EXPLANATION.md` | Comprehensive English guide to methods, choices, formulas, Python details and oral questions. |
| `validation_results.json` | Machine-readable results, data checksums, software versions and execution evidence. |
| `PROJECT_DECISIONS.md` | Technical decisions, verification history and remaining limits. |
| `archive/` | Untouched original implementation/output, its explanation and the supplied skeleton. |
| `tests/verify_project.py` | Synthetic regression and edge-case checks, separate from the submission. |

For manual submission to the private course repository, use the clean notebook and supporting files specified in the submission instructions. This repository is `ClPRlAN/FDS26`; a push here does not submit work to the separate private course repository. The annotated notebook and guide provide detailed explanations alongside the concise submission report.

## Run all cells

Use Python 3.12. From the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Prepare `.venv` with the commands above when starting from a fresh checkout. Choose its Python kernel, open the notebook and select **Restart Kernel and Run All Cells**. Run with the project directory as the working directory. Use the original trusted course files:

```text
data/df_train.pkl
data/df_financials_train.pkl
```

The supplied training subset contains 31,706 company records, 31,705 IDs, 25 industries and 768-coordinate embeddings. The financial table has 29,561 records. The assignment describes a wider dataset of 40,000+ companies; extra records were not fabricated. Data and the virtual environment are excluded from Git. Pickle files can execute code when loaded, so use only trusted files. No GPU, API key or new embedding model is needed.

## Current measured results

All rows below use the same 3,171 local validation companies. The financial-calibrated candidate was specified before evaluating this comparison. The three ensemble variants have identical labels; only confidence differs.

| Method | Accuracy | Weighted F1 | Macro F1 | Brier | Assignment score |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 78.15% | 77.78% | 71.75% | 0.137087 | 67.12% |
| k-NN | 77.70% | 77.14% | 71.05% | 0.141496 | 66.23% |
| Raw ensemble | 78.62% | 78.09% | 72.31% | 0.137380 | 67.36% |
| Text-calibrated ensemble | 78.62% | 78.09% | 72.31% | 0.135923 | 67.47% |
| **Financial-calibrated ensemble** | **78.62%** | **78.09%** | **72.31%** | **0.135585** | **67.50%** |

LR is the stronger required classifier. Voting adds about 0.31 percentage points of weighted F1; finances add only 0.026 percentage points of assignment score over text calibration. These are modest gains on one split, not proof of robust superiority on every dataset. The formula in the teaching attachment `classification.ipynb` differs from the assignment skeleton and is not used.

## Execution and verification

The submission and study notebooks each completed **all 19 code cells**, with **8 plots** and **no errors**, from separate fresh kernels. The preserved original baseline was also rerun without changing its archived file. Baseline, submission and study predictions are identical; probabilities, PCA variance, calibration features and confidence match within `atol=rtol=1e-10` on the real data.

The submission run including verification took **87.3 seconds in the Linux cloud**. Sampled summed runner-plus-kernel RSS peaked at **1,594,036,224 bytes (1.48 GiB)**. This is an upper bound that may count shared pages twice, not a precise exclusive-memory measurement. Runtime includes the scoring example and verification. The assignment requires training within **45 minutes on mid-tier hardware with 16 GB RAM**; the cloud result is not a benchmark on the student's laptop.

Checks cover assignment coverage and original scorer/example preservation, class-column alignment, disjoint company IDs, finite normalized probabilities, full-validation inference without targets, bounded confidence, input order/immutability, missing/infinite ratios, median aggregation, empirical ties and the 19/20-observation profile cutoff. To repeat the synthetic regression checks:

```bash
.venv/bin/python -m pip check
.venv/bin/python tests/verify_project.py
```

Synthetic checks create temporary data and write artifacts to `/tmp/fds26-verification/`; their scores are not course-data measurements. The visible notebook outputs and `validation_results.json` use the original uploaded datasets.

## Predictions after training

```python
industry_names, confidence = predict_industry(companies)
industry_names, confidence = predict_industry(companies, financials)
```

`companies` requires `business_description_embedding` as a Python-list string or a vector, with the training dimension. Financial ratios are `net_profit_margin`, `ebitda_margin` and `asset_turnover`; unavailable ratios become NaN. A separate financial table requires `id` in both tables, replaces any prejoined ratios, aggregates duplicates by median and preserves company-row order. It may omit individual ratios. The function returns industry names and confidence vectors in input order, never reads target labels and does not mutate the caller's table.

Do not refit the base models after calibration without rebuilding and evaluating confidence for the changed system.

## Limits and study

Company IDs do not overlap across subsets, but 455 validation records share exact fitting embeddings: accuracy is 99.34% on these versus 75.15% on 2,716 unseen embeddings. Seventeen embedding groups contain conflicting labels. Validation has also been inspected in earlier project development. These limits matter when estimating private-test performance.

The uncertainty method uses empirical financial compatibility and binary logistic calibration. Compatibility is not a correctness probability or a formal combined p-value. Confidence below 0.8 flags 36.9% of validation records and captures 73.2% of errors; flagged cases remain in every score.

The project was prepared with AI assistance. Use the annotated notebook and guide to understand the methods, explain the choices and discuss their limits independently. Results describe local validation performance.
