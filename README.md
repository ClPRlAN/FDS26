# Industry classification — final project

The project uses the course's supplied text embeddings, logistic regression and k-NN. An ensemble combines their probabilities; a separate logistic model estimates whether the chosen prediction is correct, using text and financial evidence.

## Which files to use

| File | Purpose |
|---|---|
| `industry_classification.ipynb` | **Submission notebook**: all implementation code, no Python comments, short English Markdown explanations. |
| `industry_classification_commented.ipynb` | The same executable code with detailed English comments and study explanations. |
| `EXPLANATION.md` | Oral-exam guide: methods, formulas, reasons, limitations, examples and practice questions. |
| `PROJECT_DECISIONS.md` | Verified project status, technical decisions and remaining work. |
| `archive/` | Unmodified original notebook with historical outputs and its original English explanation. |
| `tests/verify_project.py` | Technical checks using artificial data, separate from the submission. |

For your manual upload to the official course repository, use the clean submission notebook and, if required, this README and `requirements.txt`. Any later instructor instructions about filenames or interfaces take precedence. This repository is `ClPRlAN/FDS26`; uploading here does not submit anything to the separate private course repository.

## Running the project

Use Python 3.12. From the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

The cloud checkout already has a prepared `.venv`. Open the notebook and choose **Restart Kernel and Run All Cells**, using that environment's Python kernel. Run with the project directory as the working directory. The following original, trusted course files are required:

```text
data/df_train.pkl
data/df_financials_train.pkl
```

Pickle files can execute code when loaded, so use only trusted course datasets. Data and the virtual environment are already excluded from Git. No GPU, new API, new embeddings or secret keys are needed.

## Verification status

**The original datasets were unavailable during this revision.** The final notebook and the baseline were both executed completely with synthetic data in a temporary directory. Split membership, selected voting weight, predictions, probabilities, calibration features and confidence matched within `1e-10`. Checks also cover the official scorer, independence from inference targets, row order, unchanged caller inputs, missing/infinite financial values and aggregation by ID. These checks do not measure performance on the course data.

The clean notebook contains neither copied historical outputs nor artificial results presented as real measurements. Before submission, execute it with both original datasets, inspect the results and save the executed notebook. Different data or software versions can produce different results.

### Historical results preserved in the archive

These numbers come from outputs already saved in the repository, using 3,171 validation records. **They are not a new real-data evaluation of the final revision.**

| Method | Accuracy | Weighted F1 | Macro F1 | Brier | Official score |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 78.15% | 77.78% | 71.75% | 0.13709 | 67.12% |
| k-NN | 77.70% | 77.14% | 71.05% | 0.14150 | 66.23% |
| Raw ensemble | 78.62% | 78.09% | — | 0.13738 | 67.36% |
| Text-calibrated ensemble | 78.62% | 78.09% | — | 0.13592 | 67.47% |
| Financial-calibrated ensemble | 78.62% | 78.09% | — | 0.13558 | 67.50% |

The skeleton's score is **weighted F1 × (1 − Brier)**. The binary formula in the teaching attachment `classification.ipynb` is different and is not used. The available official attachments specify no numerical bonus threshold; neither bonus points nor private-test performance can be guaranteed. With weighted F1 of 78.09%, changing confidence alone cannot produce an official score of 85%.

The final revision preserves the baseline's splits and hyperparameters. The submission has 206 physical Python lines, including blank lines, compared with the previous 276. No new tuning was attempted without the datasets. Readability takes priority over saving a few lines through dense expressions.

## Making predictions after training

```python
industry_names, confidence = predict_industry(companies)
industry_names, confidence = predict_industry(companies, financials)
```

`companies` must contain `business_description_embedding`, either as a string representation of a Python numeric list or as an already converted vector. Its dimension must match training. The financial ratios used are `net_profit_margin`, `ebitda_margin` and `asset_turnover`; missing ratios become `NaN`. When supplying a separate `financials` table, both tables must contain `id`. That table **replaces** any financial ratios already in `companies`, aggregates repeated IDs by median and retains companies without matching finances. Missing ratio columns in the separate table are also treated as unavailable.

The output is a tuple of vectors in input-row order: industry names and confidence in `[0, 1]`. The function does not read `industry` targets or implicitly retrieve training finances. Do not refit the base classifiers after calibration without rebuilding and evaluating calibration for the changed system.

## Repeating the checks

```bash
.venv/bin/python -m pip check
.venv/bin/python tests/verify_project.py
```

The second command creates temporary synthetic data, executes both notebooks and compares numerical outputs. Artifacts are written to `/tmp/fds26-verification/`. It does not write datasets into the project. Synthetic scores are not course-test results. No benchmark has been performed on the student's laptop.

The project was prepared with AI assistance. The study materials explain the implementation and support reasoning about the methods and their limitations.
