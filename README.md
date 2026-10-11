# Industry classification practical

`industry_classification.ipynb` follows this year's official skeleton, including
its exact scoring function. The saved outputs come from all 31,706 supplied
company records. `EXPLANATION.md` explains the Python, methods, and oral-exam questions.

## Measured validation results

Model choices use fitting and tuning companies; confidence uses separate
calibration companies. The following results use 3,171 final validation records.

| Method | Accuracy | Weighted F1 | Brier | Official score |
|---|---:|---:|---:|---:|
| Logistic regression | 78.15% | 77.78% | 0.1371 | 67.12% |
| k-NN | 77.70% | 77.14% | 0.1415 | 66.23% |
| Raw ensemble | 78.62% | 78.09% | 0.1374 | 67.36% |
| Text-calibrated ensemble | 78.62% | 78.09% | 0.1359 | 67.47% |
| Financial-calibrated ensemble | 78.62% | 78.09% | 0.1356 | 67.50% |

The official score is **weighted F1 × (1 − Brier)**. Financial information changes
confidence, so all ensemble variants share the same predicted industries and F1.
The financial variant was fixed before final evaluation; the notebook reports the
text-only comparison even if its score is higher. The **85% bonus target was not
reached locally**. The private test set is unavailable, so no private score or grade
is claimed.

The final cloud run executed 14 code cells and produced 8 plots in
81.5 seconds, including verification. The runner-plus-kernel peak-memory
upper bound was 1.37 GiB. These are Linux cloud
measurements; the 45-minute limit on an Intel Mac has not been measured.

## Data and running

Keep the supplied course files in this folder layout:

```text
industry_classification.ipynb
data/df_train.pkl
data/df_financials_train.pkl
```

Embeddings are supplied; no text model or GPU is needed. Data is excluded from Git
because publication rights are unconfirmed. Use the original data copies already
on your computer when running the downloaded project.

Use Python 3.12. From the project folder on the Mac:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

These commands create and activate a Python environment, install the pinned
packages, and open Jupyter. Open the notebook and choose **Kernel → Restart Kernel
and Run All Cells**. The cloud and Mac have separate files and environments.
In this cloud workspace, use the prepared `/workspace/.venv` environment.

## Method and limitations

- Company-ID groups are split into fitting (25,681 records), tuning (1,427),
  confidence calibration (1,427), and final validation (3,171), using seed 42.
- Logistic regression uses the original 768 coordinates and C=10; distance-weighted
  k-NN uses 15 neighbors and 128 fitting-only PCA components. Numerical-library
  threads and k-NN parallel jobs are each limited to two.
- The notebook reports imbalance, PCA, accuracy, weighted/macro F1, per-industry
  reports, both confusion heatmaps, common confusion pairs, and shared errors.
- Tuning chooses the soft-vote weight from 25%, 50%, and 75% logistic regression;
  this run selects 25%. Small development comparisons informed C, neighbor
  count, and PCA dimension. Additional nonlinear and financial-label classifiers
  did not justify their complexity and are excluded. These comparisons establish
  the strongest tested choice, not a global optimum.
- Within-company median aggregation resolves repeated financial IDs. Infinities
  become missing; negative values remain. Industry percentiles provide bounded
  financial compatibility and an availability signal, without normality assumptions.
- Separate binary logistic models learn confidence from text evidence alone or
  text plus financial evidence. Compatibility is not a correctness probability or
  a formal p-value. Confidence bins compare estimates with observed accuracy.
- A fixed confidence-below-0.8 policy flags 36.9% of validation
  records and captures 496 of 678 errors. Flagged companies still
  receive predictions and remain in every score.

Company IDs do not overlap across splits, but descriptions can repeat across
different IDs. **455 validation records (14.35%) have an exact embedding string
seen during fitting.** Seen/unseen-description accuracy is reported separately;
the score estimates performance on unseen IDs rather than wholly unseen text.
There are also 17 identical-embedding groups with conflicting industry labels.
Rare industries and small confidence bins have less stable estimates.

`predict_industry(companies)` returns `(industry_names, confidence)` in input-row
order and does not read the target industry. Its input needs the embedding column
and the three financial columns; absent ratios must be `NaN`. Join any separately
supplied private-test finances by ID, using the same aggregation as the notebook.
The official materials do not specify a private-test entry-point name or file format.

## Verification and submission

The real-data run passed checks for the exact official scorer, disjoint company
IDs, aligned probability columns, vector sizes, bounded confidence, row order,
target independence, and missing financial inputs. Earlier artificial-data checks
tested unusual financial values; their scores are not assignment results.

The GitHub project files are exactly:

1. `industry_classification.ipynb`
2. `README.md`
3. `EXPLANATION.md`
4. `requirements.txt`
5. `.gitignore`

The original empty `readme` remains unchanged. Data, environments, trial models,
and verification helpers are excluded. A commit records a version in Git; pushing
sends it to GitHub. The student approved uploading these five files to the connected
FDS26 repository after reviewing the file list.

AI assistance was used with the student's stated permission. The accompanying
guide supports explaining every part of the implementation.
