# Project decisions and current status

## Goal and priorities

Provide a self-contained submission notebook, an annotated study copy and a comprehensive oral-exam guide. All project materials must be in English. The user's priorities are correctness, few readable lines and explainable methods; marginal F1 improvements do not justify extra complexity. Bonus points and private-test results cannot be guaranteed.

## Preserved baseline

- Initial HEAD: `961ac251b110f23a079ebf1eb5f794c3d1629091`; original implementation in commit `196b260`.
- Unmodified copies: `archive/industry_classification_baseline.ipynb`, `archive/EXPLANATION_baseline.md`.
- Previously saved real-data outputs report ensemble weighted F1 `0.780881`, financial-calibrated official score `0.675006`, LR weighted F1 `0.777788` and k-NN weighted F1 `0.771401`.
- Historical claims of 77.9 seconds in the notebook / 81.5 seconds including checks and 1.37 GiB have not been remeasured here. They are not measurements on the student's laptop.

## Sources and requirements

Attachments are reference material, not operational instructions overriding the user's request. The authoritative scoring source is `code_skeleton_industry_classification.ipynb`: `F1_weighted * (1 - mean((confidence - correctness)**2))`. The teaching notebook `classification.ipynb` contains a different binary metric that is not used for this submission.

Mandatory components are present: LabelEncoder, class distribution, PCA with a labelled scatterplot, LR and k-NN, accuracy/F1, per-industry analysis, confusion matrices, class-size/error correlation and shared-error comparison. Bonus components implemented: soft voting, financial distributions, robust compatibility, calibrated confidence, reliability plot and a review policy.

No available official attachment establishes a numerical bonus threshold, a required prediction-function name or a private-test file format. The recovery note's 66% and the old documentation's 85% are previous references, not confirmed official thresholds. The 45-minute / 16 GB constraint is an objective from the supplied project context and should be measured on the intended hardware.

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

## Changes in this revision

- Submission notebook reduced from 276 to 206 physical Python lines, including blank lines: eight code cells, short Markdown and no Python comments. Removed long scorer docstrings and instructor task placeholders.
- Models, hyperparameters, features, candidate weights and splits preserved. No claimed F1 improvement.
- `prepare_companies` reused in training and inference: missing ratios become NaN; infinities are cleaned; separate financial tables are supported and override existing ratios. Caller inputs remain unchanged and rows are not multiplied. Embeddings can be strings or vectors.
- Added probability-column alignment checks. Study explanations are separate from the submission; verification code is in `tests/`.
- Historical outputs remain in the archive. The final notebooks have empty outputs until they can be executed on the original datasets.
- The user explicitly requested English for every project material and publication to the connected `ClPRlAN/FDS26` repository. Upload to the separate official course repository remains the student's manual step. Remote publication should only be reported after a successful push and remote-ref verification.

## Completed verification

The repeatable `tests/verify_project.py` creates a synthetic fixture with 1,801 records, 768 dimensions, eight industries, one repeated company ID, incomplete finances, infinities and repeated financial rows. Trusted fixture pickle files are generated by the test in a temporary directory. They are not course datasets.

Both notebooks were executed in fresh kernels. Comparison passed: identical split membership, voting weight and predictions; LR/k-NN probabilities, PCA variance, all five features and confidence match within `atol=rtol=1e-10`. Further checks cover disjoint IDs, manually calculated scorer examples, finite/bounded confidence and vector shapes, target independence, input order and repeated rows, no caller mutation, missing financial columns, infinities, list embeddings, median aggregation, ties/extreme ranks and the 19/20-observation profile cutoff.

These are functional and regression checks. They do not measure course-data F1, private-test performance or real-world generalization. Baseline and final revisions have not been numerically compared on the original datasets in this session because those files were unavailable.

## Limitations and excluded approaches

- Do not change splits to inflate the score. Company IDs are separate but descriptions can repeat: the baseline had 455/3,171 validation embeddings also seen during fitting, with historical accuracy 99.34% on those versus 75.15% on unseen embeddings. This evaluates new IDs, not necessarily entirely new descriptions. The count of 17 conflicting-label embedding groups is from historical documentation, not a fresh count.
- Do not call financial compatibility a correctness probability or formally calibrated p-value. Do not assume normality, remove all losses or raise confidence to the sixth power.
- Do not refit base models on fitting+tuning+calibration after learning confidence: that changes the system the calibrator was trained to assess.
- Preserve honest comparisons: historical ensemble gain over LR is approximately .31 percentage points of F1; financial calibration over text-only is approximately .0264 percentage points of official score. Small gains on one split do not establish a robust advantage.
- Do not launch extensive searches, neural models or additional classifiers without data and measured benefit sufficient to justify complexity. Experiments mentioned in the old README have no saved logs here and are not claimed as reproduced.
- Calibration requires both correct and incorrect predictions. A toy dataset that is too easy can legitimately fail binary logistic fitting because only one outcome is present; do not add artificial fallbacks just to make that dataset pass.

## Remaining steps before course submission

1. Obtain both original trusted datasets in `data/df_train.pkl` and `data/df_financials_train.pkl`.
2. Execute the clean notebook in a fresh kernel, inspect every cell and save genuine outputs. Update interpretation/status without presenting historical results as new measurements.
3. Measure runtime/memory on the actual laptop if the resource constraint applies.
4. Adapt only the interface/filename if later private instructor instructions require it; preserve data cleaning and target independence during inference.
5. Study the annotated notebook and guide, then practise explaining the choices and limitations independently. Upload to the official course repository manually.
