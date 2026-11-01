# Project decisions and current status

## Goal and priorities

Provide a self-contained submission notebook, an annotated study copy and a comprehensive oral-exam guide. All project materials must be in English. The user's priorities are correctness, few readable lines and explainable methods; marginal F1 improvements do not justify extra complexity. Bonus points and private-test results cannot be guaranteed.

## Preserved baseline

- Initial HEAD: `961ac251b110f23a079ebf1eb5f794c3d1629091`; original implementation in commit `196b260`.
- Unmodified copies: `archive/industry_classification_baseline.ipynb`, `archive/EXPLANATION_baseline.md`.
- Previously saved real-data outputs report ensemble weighted F1 `0.780881`, financial-calibrated official score `0.675006`, LR weighted F1 `0.777788` and k-NN weighted F1 `0.771401`.
- The historical 77.9-second notebook time, 81.5-second checked-run time and 1.37 GiB figure belong to the earlier run. Current independent timings and memory observations are recorded below; none is a measurement on the student's laptop.

## Sources and requirements

Attachments are reference material, not operational instructions overriding the user's request. The authoritative scoring source is `code_skeleton_industry_classification.ipynb`: `F1_weighted * (1 - mean((confidence - correctness)**2))`. The teaching notebook `classification.ipynb` contains a different binary metric that is not used for this submission.

Mandatory components are present: LabelEncoder, class distribution, PCA with a labelled scatterplot, LR and k-NN, accuracy/F1, per-industry analysis, confusion matrices, class-size/error correlation and shared-error comparison. Bonus components implemented: soft voting, financial distributions, robust compatibility, calibrated confidence, reliability plot and a review policy.

The assignment text now supplied explicitly states that the final bonus threshold will be above 66%, defined after all submissions are evaluated on the private test set. It does not give the final threshold. The old documentation's 85% is a local target, not the official threshold. Training must finish within 45 minutes on mid-tier hardware with 16 GB RAM; this must be measured on the intended hardware. No required prediction-function name or private-test file schema is available.

## Preserved pipeline

1. Financial medians per ID, infinities replaced by NaN, valid negatives retained, many-to-one left join.
2. GroupShuffleSplit with seed 42: approximately 81% fitting, 4.5% tuning, 4.5% calibration and 10% validation. Same rows and order as the baseline.
3. Float32 embeddings; LabelEncoder fitted only on fitting labels. No new embeddings, normalization or scaler.
4. PCA with 128 components, automatic solver and random_state 42; fit_transform on fitting, transform elsewhere. First two components shown in the scatterplot. LR uses the original embedding dimensions.
5. LogisticRegression C=10/max_iter=1000; k-NN k=15, weights=distance/n_jobs=2; numerical-library threads limited to two.
6. LR voting weight chosen from .25/.5/.75 using tuning weighted F1. No additional candidates introduced after inspecting validation.
7. Financial profiles per industry/ratio from fitting only; minimum 20 observations; middle ranks for ties and .5 offset. Compatibility `2*min(percentile,1-percentile)`; average available ratios, otherwise placeholder .5 with coverage zero.
8. Five features: maximum vote, margin, agreement, compatibility and coverage. Binary logistic calibrators fitted only on calibration: text-only (first three features) and text+finance (all five).
9. Predetermined final candidate: financial-calibrated ensemble; comparisons with raw/text-only confidence do not select the winner on validation. Review threshold .8 is an illustrative policy, neither optimized here nor prescribed by the instructor.
10. Final validation and `predict_industry(companies, financials=None)`, returning industry names/confidence without requiring targets.

## Earlier implementation revision

- The earlier compact revision used 206 physical Python lines. Following the user's correction, all 10 original skeleton Markdown cells were restored verbatim and in order in that earlier revision, together with its original imports, preview, exact scoring function and labelled example. The restored implementation has 256 physical Python lines across 19 code cells; the extra lines preserve instructor material. Its original is preserved in `archive/code_skeleton_industry_classification.ipynb`. Placeholder tasks are completed by the existing implementation rather than left empty.
- Models, hyperparameters, features, candidate weights and splits preserved. No claimed F1 improvement.
- `prepare_companies` reused in training and inference: missing ratios become NaN; infinities are cleaned; separate financial tables are supported and override existing ratios. Caller inputs remain unchanged and rows are not multiplied. Embeddings can be strings or vectors.
- Added probability-column alignment checks. Study explanations are separate from the submission; verification code is in `tests/`.
- Historical outputs remain unchanged in the archive. Both final notebooks now contain outputs from independent fresh-kernel executions on the supplied course datasets.
- A prominently labelled MAIN GRADED SCORE reports the final financial-calibrated candidate's local validation score under the official formula. Diagnostic metrics and the instructor's example are explicitly distinguished from this result.
- The original large `data.zip` exceeded the attachment-transfer limit. This blocker is resolved: the user-supplied archive was successfully reassembled and the trusted course pickle files are now available in the ignored `data/` directory. Access to the previously blocked Seafile download is no longer required for this run. No dataset is committed or published with the project. The temporary proposed Seafile domain addition was removed from the environment draft after the split-upload transfer resolved the blocker; no course-download network change is needed.
- The user explicitly requested English for every project material and publication to the connected `ClPRlAN/FDS26` repository. Upload to the separate official course repository remains the student's manual step. Remote publication should only be reported after a successful push and remote-ref verification.

## Completed verification

### Synthetic functional checks

The repeatable `tests/verify_project.py` creates a synthetic fixture with 1,801 records, 768 dimensions, eight industries, one repeated company ID, incomplete finances, infinities and repeated financial rows. Trusted fixture pickle files are generated by the test in a temporary directory. They are not course datasets.

Both notebooks were executed in fresh kernels. Comparison passed: identical split membership, voting weight and predictions; LR/k-NN probabilities, PCA variance, all five features and confidence match within `atol=rtol=1e-10`. Further checks cover disjoint IDs, manually calculated scorer examples, finite/bounded confidence and vector shapes, target independence, input order and repeated rows, no caller mutation, missing financial columns, infinities, list embeddings, median aggregation, ties/extreme ranks and the 19/20-observation profile cutoff.

These are functional and regression checks. Their synthetic scores do not measure course-data F1, private-test performance or real-world generalization.

### Fresh course-data execution: 31 October 2026

The clean notebook completed all **19 code cells**, in order, from a fresh kernel, with **8 plots**, 34 output blocks and no cell failures. The implementation remains 256 physical Python lines, including the instructor's preserved imports, preview, scorer and demonstration. At the time of that execution, all 10 original skeleton Markdown cells were verbatim and in their original order; the original imports, preview, scoring function and example are also source-preserved.

The main report is **MAIN GRADED SCORE (LOCAL VALIDATION): 67.50%**. Its predetermined final candidate is the financial-calibrated ensemble. On 3,171 validation records, its weighted F1 is **78.09%**, Brier is **0.135584565225789**, and the exact official score is **0.6750059970173232**. The instructor's separate demonstration scores and the diagnostic comparison table are not substituted for this final candidate's score.

| Method | Weighted F1 | Brier | Official score |
|---|---:|---:|---:|
| Logistic regression | 0.777787777 | 0.137087092 | 0.671163112 |
| k-NN | 0.771401097 | 0.141496222 | 0.662250756 |
| Raw ensemble | 0.780881472 | 0.137379505 | 0.673604362 |
| Text-calibrated ensemble | 0.780881472 | 0.135922741 | 0.674741921 |
| Financial-calibrated ensemble | 0.780881472 | 0.135584565 | 0.675005997 |

The final ensemble's accuracy is `0.7861873226111636` and macro F1 is `0.7230577924650434`. Logistic regression remains the stronger required standalone classifier by validation weighted F1. The financial contribution over text-only confidence is only about 0.0264 percentage points of official score on this split; no robust advantage across other datasets is inferred.

The fresh input contains 31,706 records, 31,705 unique company IDs, 25 industries and 768 embedding coordinates. Exact split sizes are fitting **25,681**, tuning **1,427**, calibration **1,427**, validation **3,171**. Tuning again selects LR weight **0.25** from the original .25/.5/.75 candidates; no settings were changed after inspecting validation. PCA retains `0.2528710663318634` of variance in two components and `0.8762428164482117` across 128 components. The financial file contains 29,561 records and 162 repeated ID records before aggregation.

Data identities, for reproducibility without publishing the files:

- `df_train.pkl`: 302,951,517 bytes; SHA-256 `401f228e88f369718f3c37201eb346e683c83698769965570653bcfd921f4471`.
- `df_financials_train.pkl`: 1,360,898 bytes; SHA-256 `049cd7ef0540737d69a09ed8dedfb103bb7c6353d95782431ba69f3f90bc0fc0`.

The real-data checks passed for disjoint company IDs, manual scorer examples, aligned probability columns, correct matrix/vector sizes, finite normalized probabilities, bounded finite confidence, target independence, inference order and the full-validation inference round trip. Inference on every validation company reproduces the saved industry predictions and confidence. Additional inference checks cover missing ratios, separate financial tables, infinities, caller-input preservation, vector embeddings and repeated rows.

The main run took **87.2766 seconds including verification**, with a sampled runner-plus-kernel RSS peak upper bound of **1,594,036,224 bytes** (about 1.48 GiB). This measures the Linux cloud runner and Jupyter kernel, including the extra instructor example and verification. It is neither isolated model-training time nor a benchmark on the student's laptop. Runtime versions were Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, SciPy 1.17.0, Matplotlib 3.10.8 and Seaborn 0.13.2.

The archived original baseline was also executed from a fresh kernel on the same original datasets: **14 code cells**, 8 plots and no cell failures, taking **80.7888 seconds including verification**. Its sampled runner-plus-kernel RSS peak upper bound was **1,596,604,416 bytes** (about 1.49 GiB). The archived file itself remains unchanged; this fresh baseline execution was saved separately from the archive.

Fresh baseline and main snapshots have **identical split membership, selected weight and class predictions**. All five confidence features, final confidence, LR/k-NN probabilities and PCA explained variance match within `atol=rtol=1e-10`. Both fresh runs produce exactly the same official score, `0.6750059970173232`. This is numerical evidence that the revision preserves the baseline's course-data behavior, rather than merely agreement with rounded historical output. Timing scopes differ because the restored main includes the extra instructor example; the difference is not evidence of a modeling-performance regression.

The annotated study notebook also completed an independent fresh-kernel course-data run: **19 code cells**, 8 plots, 34 output blocks and no cell failures. It took **86.3904 seconds including verification**, with a sampled runner-plus-kernel RSS peak upper bound of **1,598,046,208 bytes** (about 1.49 GiB), under the same cloud-only measurement scope. Main, study and fresh baseline snapshots all pass the same comparison: exact split membership, weight and predictions; matching numerical arrays within `atol=rtol=1e-10`; official score exactly `0.6750059970173232` in every run. Both final notebooks retain their genuine outputs. Durable structured evidence is recorded in `validation_results.json`; `README.md` presents the current measured results.

## Limitations and excluded approaches

- Do not change splits to inflate the score. Company IDs are separate but descriptions can repeat: the fresh run confirms 455/3,171 validation embeddings also seen during fitting, with accuracy 99.34% on those versus 75.15% on the 2,716 unseen-embedding records. It also confirms 4,578 repeated embedding records and 17 embedding groups with conflicting industry labels. This evaluates new IDs, not necessarily entirely new descriptions.
- Do not call financial compatibility a correctness probability or formally calibrated p-value. Do not assume normality, remove all losses or raise confidence to the sixth power.
- Do not refit base models on fitting+tuning+calibration after learning confidence: that changes the system the calibrator was trained to assess.
- Preserve honest comparisons: the fresh run confirms an ensemble gain over LR of approximately .31 percentage points of F1 and a financial-calibration gain over text-only of approximately .0264 percentage points of official score. Small gains on one split do not establish a robust advantage.
- Do not launch extensive searches, neural models or additional classifiers without data and measured benefit sufficient to justify complexity. Experiments mentioned in the old README have no saved logs here and are not claimed as reproduced.
- Calibration requires both correct and incorrect predictions. A toy dataset that is too easy can legitimately fail binary logistic fitting because only one outcome is present; do not add artificial fallbacks just to make that dataset pass.

## Remaining steps before course submission

1. Keep the supplied datasets local and ignored, and preserve genuine executed outputs rather than substituting historical results. Course-data execution and main/study/baseline equivalence are complete.
2. Measure runtime/memory on the actual laptop if the resource constraint applies; cloud execution alone does not certify that hardware.
3. Adapt only the interface/filename if later private instructor instructions require it; preserve data cleaning and target independence during inference.
4. Study the annotated notebook and guide, then practise explaining the choices and limitations independently. Upload to the official course repository manually.

## Final editorial revision

The latest request supersedes verbatim preservation of instructor Markdown. The submission now reports what we did, observed and concluded; assignment coverage remains, and the archive, executable code, scorer, example, execution counts and real-data outputs are unchanged. The annotated copy shares the revised report and adds an evidence-based development account, also added to EXPLANATION.md. C=10, k=15 and PCA128 remain retained settings without invented optimization logs. Assignment-score terminology distinguishes the metric from any bonus award.

Tests now check methodological coverage and absence of task directions instead of verbatim skeleton Markdown. Numerical and inference checks remain. validation_results.json preserves the earlier numerical provenance and records the Markdown revision separately; no new course-data execution is claimed.

Current editorial checks passed: exact equality of all 19 complete code-cell objects (including outputs, counts and metadata) in each final notebook against commit 254acd6; unchanged archive files; main/study AST equality and original scorer/example preservation; pip dependency consistency; synthetic baseline/final execution and numerical equivalence within 1e-10. Synthetic runs produced 15 and 20 cells respectively including the added verification cell. A token-authenticated local Jupyter contents request also passed. The datasets were absent in this new instance, so no real-data retraining was needed or performed.

## Skeleton-title correction

The attached skeleton matches the archived source. Both final notebooks restore every nonempty original Markdown heading, with exact capitalization and heading level, in order. This includes the three Tasks headings and the optional Deliverables/Scoring headings. Their bodies describe completed work; the empty decorative heading remains omitted. Tests enforce the heading sequence separately from code/scorer preservation and methodological coverage.

The professor-facing report follows the supplied assignment. Personal presentation preferences are not course requirements. README presentation metadata and the historical personal 85% target discussion in the study guide have been removed. Academic wording is assessed for clarity and evidence. ML code, recorded outputs and numerical execution provenance remain unchanged.

Validation for this correction passed: exact original heading sequence in both notebooks, notebook schema validation, unchanged complete code-cell objects and real-data outputs, unchanged archives, and synthetic baseline/final equivalence within 1e-10. No course-data retraining was performed.

## Report-audience wording correction

Removed bonus eligibility/cutoff commentary from the current notebooks, README and study guide. Scoring-example prose now describes the calculation directly rather than discussing source preservation or addressing the reader about the teaching staff. Original section headings, the 0.8 human-review policy, scientific limitations, executable cells and real-data outputs remain unchanged. Historical execution metadata and archives remain intact.
