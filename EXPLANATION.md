# A guide to understanding and explaining the project

This is the study guide. `industry_classification.ipynb` is the clean notebook to run and submit; its annotated version connects the code to the ideas explained here. The aim is to understand the sequence **data → model → prediction → evaluation**, rather than memorize answers.

AI assistance was used to prepare the project. This guide is intended to help you explain the choices independently and verify them. Run the code, inspect the results and describe the limitations in your own words. The clean notebook has now been executed on the supplied real datasets: all 19 code cells completed and eight plots were produced. Section 20 reports these fresh local measurements, separately from private-test performance and laptop requirements.

### Where to start: five key concepts

Study these in this order before focusing on Python details:

1. **Splits and leakage (§5):** draw the four subsets and explain why they have different roles.
2. **LR, k-NN and PCA (§9–12):** explain what each method learns, then reason about increasing C, k or the number of components.
3. **Metrics and scoring (§13 and §17):** calculate a small F1 example and a Brier example; explain why changing confidence does not change F1.
4. **Financial data, ranks and p-values (§14–15):** use a short array to compare a central value, an extreme value and a missing value; explain what these scores cannot establish.
5. **Calibrating correctness (§16–18):** explain the new binary target and reason about a group with confidence 0.9 but accuracy 0.7.

Then use the oral questions (§22) to check your reasoning. Change the numbers in an example and reconstruct the answer. This is a better test of understanding than recalling its wording.

## 1. The project in one minute

> I want to predict a company's industry from its business description, which is already represented by a numerical vector called an embedding. I compare logistic regression and k-nearest neighbors. Logistic regression uses the original embeddings; for k-NN, I reduce their dimensionality with PCA. I separate the companies used to fit classifiers, choose the ensemble weight, learn confidence and evaluate the result. I combine the models' probabilities and use another small model to estimate how likely the prediction is to be correct. This estimate also considers whether the company's finances resemble those of the predicted industry. I evaluate both classification and confidence without using private-test labels.

Keep three ideas separate:

- **Prediction:** which known industry is the most plausible?
- **Confidence:** how plausible is it that this particular prediction is correct?
- **Evaluation:** how well do these estimates work on companies held aside?

Financial information changes confidence, rather than the industry chosen by the ensemble. Therefore, the raw, text-calibrated and financial-calibrated variants have identical predicted labels and F1; their Brier scores differ.

## 2. The machine learning problem

This is **supervised multiclass classification**. It is supervised because the development data contains the target, `industry`. It is multiclass because each record belongs to one of several industries. It is not multilabel classification: the model does not assign several industries to one record at the same time. It is not regression on a continuous quantity: the word “regression” in *logistic regression* does not change its use as a classifier here.

The **features** are information available when making predictions. The **target** is what we want to predict. The target cannot be a feature: that would give the model the answer it is supposed to estimate.

`id` links records of the same company and connects the two tables. It is not a predictive feature. The numerical difference between two IDs does not measure how similar two companies are.

We do not have the instructor's private test set. Good local validation does not guarantee the same private-test score or the award of bonus points.

## 3. Files, tables and array shapes

Run the notebook from the project directory with the original datasets:

```text
data/df_train.pkl
data/df_financials_train.pkl
```

The first table contains `id`, `industry` and `business_description_embedding`. The second contains `id` and three financial ratios:

| Column | Basic meaning |
| --- | --- |
| `net_profit_margin` | Net income relative to revenue. |
| `ebitda_margin` | EBITDA relative to revenue; this is not net profit margin. |
| `asset_turnover` | Revenue relative to total assets. |

The precise definitions depend on the dataset. EBITDA is not the same as net income or cash flow. A negative margin may represent a real loss rather than an error.

Only open pickle files from a trusted source: the format can execute code while loading. The model does not require additional data downloads or credentials.

The supplied datasets contain 31,706 description records, 31,705 distinct company IDs, 25 industries and 768-coordinate embeddings. These counts were checked again in the fresh real-data execution. There are 29,561 raw financial records before aggregation by ID.

| Object | Shape in the original data | Meaning |
| --- | --- | --- |
| `X_fit` | `(n_fit, 768)` | One feature row per record. |
| `y_fit` | `(n_fit,)` | One encoded industry per record. |
| `X_fit_pca` | `(n_fit, 128)` | Reduced features for k-NN. |
| Each classifier's probabilities | `(n, 25)` | One probability per company and industry. |
| `features_cal` | `(n_cal, 5)` | Five signals for estimating correctness. |
| Final output | two vectors of shape `(n,)` | Industry name and confidence for each record. |

The first dimension counts rows; the second counts features or classes. `axis=1` works across the columns within a row: `argmax(axis=1)` finds each company's most probable industry. `argmax` returns the position of the maximum, rather than its value.

Row `i` of the embeddings, target and financial data must always refer to the same company. An inconsistent order can produce incorrect results without causing a Python error.

## 4. Embeddings and conversion

An embedding is a numerical representation of text produced by a pretrained language model. Semantically similar descriptions may have nearby vectors: for example, “produces cars” and “manufactures vehicles,” compared with “operates supermarkets.” A single coordinate does not necessarily have an interpretable label such as “bank” or “revenue”: meaning is distributed across many coordinates.

The supplied `example_text_embeddings.ipynb` shows how to generate embeddings and illustrates cosine similarity. This project uses the provided embeddings: it does not train a language model or download a sentence transformer again. That keeps the project shorter and less expensive to run.

`embeddings` converts the values to a `float32` array. If the file contains a string such as `"[0.1, -0.2]"`, `ast.literal_eval` interprets it as a list. We do not use `eval`, which would execute code. If a value is already a list or array, we do not interpret it as text again.

`float32` uses four bytes per number, compared with eight for `float64`. It is a compact choice for these features; it does not mean that every internal library operation uses 32-bit arithmetic.

The notebook **does not apply additional L2 normalization**. Its k-NN uses Euclidean distance after PCA, rather than cosine similarity. PCA centers the coordinates, but does not make every vector have length one.

## 5. Four subsets with four roles

| Subset | Approximate share of all records | Role |
| --- | ---: | --- |
| Fitting | 81% | Fit PCA, classifiers and financial reference profiles. |
| Tuning | 4.5% | Choose the LR/k-NN ensemble weight. |
| Calibration | 4.5% | Learn the relationship between confidence signals and correctness. |
| Final validation | 10% | Evaluate the system after those choices. |

First, we reserve about 10% for validation. Of the remaining 90%, we take about 90% for fitting: `0.9 × 0.9 = 0.81`. The remaining 9% is divided equally between tuning and calibration. `df_train` means the whole 90% development portion, rather than only the rows used to fit the base classifiers.

The helper `split(companies, size)` returns two subsets using `GroupShuffleSplit`. The percentages are approximate because the split operates on groups and the number of rows is an integer. The fresh execution has 25,681 fitting records, 1,427 tuning records, 1,427 calibration records and 3,171 validation records.

### Why group by ID?

The same company may have several records. If one entered fitting and another entered validation, we would evaluate the model on a company it had already seen. `GroupShuffleSplit` keeps all rows of an ID in the same subset.

`random_state=42` makes the split repeatable. It does not automatically improve the score or make the data “more random.” The split is not stratified by industry: class proportions can change. We check that every held-out class exists in fitting because a conventional supervised classifier cannot learn an unseen class.

### What is leakage?

Leakage means using evaluation information, even indirectly, to prepare the model. It includes more than adding the target to the features. Examples:

- Fitting PCA on all companies before the split: the validation-feature distribution would influence its directions.
- Choosing the ensemble weight with the highest validation F1, then presenting that same number as an independent evaluation.
- Learning confidence on records already used to fit the classifiers: training errors do not represent errors on new companies.
- Building financial profiles with validation companies or using their true industry to choose the reference profile.

In this notebook, PCA, classifiers and profiles use fitting; the ensemble weight uses tuning; the calibrators use calibration. Results are measured on validation. The base models remain frozen after fitting: retraining them on additional data would change the system whose confidence we calibrated.

The original validation results have already been inspected during earlier versions of the project. They should not be described as a test that has remained unseen throughout the project's history. Repeated human choices made after examining the score can also make it optimistic. The pipeline separates the operations, but does not erase that historical limitation.

## 6. Preparing financial data

`prepare_companies(companies, financials=None)` applies the same rules during development and inference. When given a separate financial table, it:

1. Converts infinite values to NaN.
2. Groups by ID and takes the median of each available ratio.
3. Links financial data to companies with a left join.
4. Checks that the relationship is many-to-one.

If companies already have financial columns and we supply a separate table, the separate table replaces those columns. We do not combine two incompatible versions. Without a separate table, we use the existing columns. Any missing ratio becomes NaN, including a ratio missing from the separate financial table, and infinite values are cleaned. A separate table still needs the `id` column to link its records.

`how='left'` keeps every company, including those without financial data. An inner join would remove companies with no matching financial records, changing the cases being classified. `validate='many_to_one'` prevents multiple right-hand rows from accidentally multiplying left-hand rows. `sort=False` avoids sorting companies by the join key.

The median is less affected than the mean by one extremely large value. It is a simple way to obtain one record per company, rather than a guarantee of recovering the “true” value. If duplicate rows describe different periods, aggregation loses that information.

This aggregation uses only records of the same company, rather than statistics across companies or their targets. It differs from learning a global imputation value jointly from fitting and validation.

The fresh data-quality check found 162 repeated financial IDs. Before infinities were treated as missing, the three ratios had 1,111, 19,279 and 1,363 missing observations, respectively. They also had 2,131, 824 and 5 infinite values. In particular, EBITDA margin was already missing for about 65.2% of the 29,561 raw financial records. Coverage is therefore a substantial part of the problem, rather than an exceptional edge case.

**NaN does not mean zero:** a zero margin is a specific observation; a missing margin is unknown. We do not remove all negative values, because a loss-making company can still be classified correctly. We do not cut all extremes, because ranks limit their influence without assuming that they are false.

## 7. LabelEncoder and probability alignment

`LabelEncoder` assigns an integer to each industry name, for example `Banks → 0`, `Energy → 1`, `Materials → 2`. These integers are identifiers, rather than quantities. Industry 2 is not “twice” industry 1, and their codes do not define an economic distance.

`fit` builds the mapping from fitting classes. `transform` reuses it on other subsets. `inverse_transform` converts predicted codes back to readable names. An assertion checks that every held-out class is known.

The columns of the LR and k-NN probability vectors must represent the same industries in the same order. Another assertion checks this. Adding columns with different meanings would be incorrect even if the arrays had matching shapes.

## 8. Class imbalance

The frequency chart shows how many training companies belong to each industry. In the current run, `Materials` has 7,570 records, or 26.5% of training; `Consumer Staples Distribution & Retail`, the smallest class, has 160. Always choosing the largest class would give nonzero accuracy, but would fail to recognize the others.

Large classes provide more examples and contribute more to weighted F1. This does not mean that every large class is easy: semantic overlap, description quality and label errors also matter. Per-class reports and macro F1 expose weaknesses that a weighted average can hide.

We do not use `class_weight='balanced'`. It would increase the influence of small classes during training, but would not automatically improve weighted F1. We avoid adding options without a comparison. Omitting class weighting does not mean that imbalance has been “solved.”

## 9. PCA: 128 components for the model, two for visualization

**Principal Component Analysis** finds orthogonal directions that capture substantial feature variance. The first captures the greatest possible variance in one direction; the second captures the greatest remaining variance while being orthogonal to the first, and so on.

PCA is unsupervised: it does not use industry labels. Therefore, substantial variance does not necessarily mean substantial information for classification. A low-variance direction might still help distinguish similar industries.

`PCA(n_components=128, random_state=42)` reduces the original embeddings from 768 to 128 coordinates. This makes k-NN distances less expensive and may remove less useful variation. The choice of 128 comes from the baseline configuration; it is not a proven global optimum.

`fit_transform(X_fit)` learns the mean and directions and projects fitting data. We use `transform` on other subsets so that they share the same coordinate system. A new `fit_transform` on validation would create a different transformation and use information that should be held aside.

The scatterplot shows the first two of the 128 components because we can draw two axes. **k-NN uses all 128**, while LR uses every original coordinate. In the current run, the first two retain 25.3% of variance and all 128 retain 87.6%. Those figures are neither accuracy percentages nor percentages of “information useful for the target.”

A separated group of same-colored points suggests structure. Overlap in two dimensions does not prove that industries are indistinguishable in the original dimensions: the projection discards information.

PCA centers each coordinate. We do not use StandardScaler, whitening or additional normalization. The code should not be described as standardizing the features. Scaling is often important for variables measured in very different units; embedding coordinates have a different nature from a mixture of meters, euros and percentages.

## 10. Multiclass logistic regression

For each industry, the model combines the embedding coordinates linearly, then turns scores into probabilities using softmax:

```text
z_c = w_c · x + b_c
p_c = exp(z_c) / sum_j exp(z_j)
```

`x` is the embedding, `w_c` is the industry's weight vector and `b_c` is the intercept. Probabilities are positive and sum to one across classes. The prediction takes the largest probability.

“Linear” describes the scores and boundaries between classes in feature space; softmax itself is not a straight line. The language model has also already transformed text into a complex representation.

LR does not maximize F1 directly. It minimizes regularized log-loss: it penalizes assigning a low probability to the true class and discourages excessively large weights. F1 is measured afterward to evaluate the decisions.

We use `C=10` and `max_iter=1000`. `C` is inversely related to regularization strength: a larger value means less weight penalization, while a smaller value means more. `max_iter` limits optimizer iterations; it is not a number of trees or a requirement to perform exactly that many iterations. A convergence warning needs investigation rather than suppression.

This is a standard, fast and compact model that can use information distributed across embeddings. It cannot represent every nonlinear boundary, and its probabilities are not automatically perfectly calibrated confidence estimates.

## 11. k-Nearest Neighbors

k-NN searches for fitting companies nearest to a new one. Here it uses 15 neighbors in the 128-dimensional PCA representation. It does not learn coefficients like LR: it stores a representation of the examples and prepares neighbor search.

With these parameters, distance is **Euclidean**: the square root of the sum of squared coordinate differences. It is not automatically cosine similarity. `weights='distance'` gives closer neighbors greater influence. An industry's probability is its share of the weighted vote.

A small `k` is very local and sensitive to noise. A large `k` makes voting more stable but can hide useful details and favor frequent classes. Fifteen is the baseline compromise, rather than a universally correct number.

In many dimensions, distances may become less informative and more expensive. PCA helps, but does not eliminate every dimensionality problem. k-NN's main cost may occur during prediction rather than fitting: measuring `.fit()` alone does not measure its total cost.

`n_jobs=2` and `threadpool_limits(2)` limit resources used by some operations. They do not change the classifiers' definitions. Previously computed probabilities are reused in comparisons to avoid repeating neighbor searches.

## 12. Ensemble: combining probabilities

LR uses global rules; k-NN uses local comparisons. They may make mistakes on different examples, although many errors remain shared. **Soft voting** combines their probability vectors:

```text
votes = lr_weight × proba_lr + (1 − lr_weight) × proba_knn
prediction = the industry with the highest value in votes
```

For a two-class example, LR gives `[0.8, 0.2]` and k-NN gives `[0.3, 0.7]`. With LR weight 0.25, the result is `[0.425, 0.575]`, so the second class wins. Weight 0.25 does not mean LR is correct 25% of the time: it specifies its contribution to the average.

The notebook compares weights `0.25`, `0.5` and `0.75` using tuning weighted F1. The search is small and understandable; the best weight is frozen before calibration and ensemble evaluation. The fresh run selects 0.25: the three tuning F1 values are approximately 79.18%, 78.37% and 78.22%, respectively. These are tuning results, separate from the final validation F1.

The best of three candidates is not necessarily a global optimum. We do not select on validation or optimize against the private test. An ensemble does not always improve performance: we check metrics and shared errors.

## 13. Classifier metrics

### Accuracy, precision, recall and F1

Accuracy is the fraction of records classified correctly. To define the other metrics in a multiclass problem, consider one industry at a time as positive:

- TP: belongs to the industry and is predicted as that industry.
- FP: belongs elsewhere but is incorrectly predicted as this industry.
- FN: belongs to this industry but is predicted elsewhere.

```text
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 × precision × recall / (precision + recall)
```

Example: there are 20 actual banks, 16 are recognized, and four other companies are incorrectly called banks. TP=16, FN=4 and FP=4: precision, recall and F1 are all 0.8. If we called many additional companies banks, recall might stay high while precision fell.

F1 is the harmonic mean of precision and recall: it penalizes a substantial imbalance between them. It is neither their arithmetic mean nor the percentage of correctly classified companies.

### Weighted and macro F1

We calculate an F1 for each industry. **Macro F1** gives every industry the same weight. **Weighted F1** weights each by its support: the number of true records of that class in the evaluated subset.

With classes containing 90 and 10 records, their F1 scores have weights 0.9 and 0.1 in weighted F1, but 0.5 each in macro F1. Weighted F1 can therefore hide weaknesses in rare industries. `zero_division=0` assigns zero to undefined ratios, such as precision for a class never predicted. It avoids an undefined result; it does not fix the predictive problem.

### Confusion matrices and error comparisons

Rows represent true industries; columns represent predicted industries. The diagonal contains correct predictions. The plots normalize each row: a cell of 0.2 means that 20% of the row's true industry was assigned to the column's industry. This also makes small classes visible.

The most-confused industry pairs use absolute counts, which are affected by class size. The LR/k-NN comparison counts errors made by both, only LR and only k-NN. Complementary errors support trying a combination, but do not guarantee that it will help.

The correlation between training size per industry and validation error describes an association rather than a cause. A negative correlation suggests fewer errors in larger classes; it does not prove that adding records alone would resolve semantic confusion.

In the fresh validation, LR is stronger than k-NN overall: weighted F1 is 77.78% versus 77.14%, with accuracy 78.15% versus 77.70%. Their macro F1 values, 71.75% and 71.05%, show weaker results when every industry receives equal weight. LR recognizes Banks very well (97.17% F1) and Materials well (92.43%), but struggles with Commercial & Professional Services (37.59%), Household & Personal Products (44.44%) and Consumer Staples Distribution & Retail (46.15%). The last two estimates use only 19 and 18 validation records, respectively.

Both models often predict Materials for true Capital Goods companies (28 LR cases, 33 k-NN cases), and Capital Goods for Technology Hardware & Equipment (21 and 24 cases). These related business activities are plausible sources of semantic confusion, although the counts alone do not prove a particular cause. The class-size/error correlations are −0.362 for LR and −0.284 for k-NN. Both models are wrong on 520 records; 173 errors belong only to LR and 187 only to k-NN. The ensemble can use some of those differences, but the substantial shared errors limit its gain.

## 14. What financial plots show

The tables show available counts and medians by industry; boxplots summarize medians and quartiles. `showfliers=False` hides extreme dots in the plot, but does not remove those observations from the data or reference profiles.

The `symlog` scale displays negative, zero and positive values across different orders of magnitude. An ordinary logarithm would be undefined for negative values. This is a plotting choice rather than an additional transformation of classifier features.

We do not assume that every ratio follows a normal or lognormal distribution. The p-value material suggests checking unusual values, but does not automatically justify removing losses or concluding “unusual finances mean an incorrect industry.” A company may be young, distressed, exceptionally profitable or subject to different accounting conventions. Financial data provides supporting evidence.

## 15. Financial compatibility from empirical percentiles

For each `(industry, ratio)` pair, `profiles` stores the fitting subset's finite values in sorted order. Profiles with fewer than 20 observations are skipped. This is a practical minimum to avoid very small references, rather than a statistical law.

For a new company, we use the **predicted industry's** profile, never the true industry's. `searchsorted` locates two positions for its value `x`:

- `left`: how many reference observations are strictly smaller than `x`;
- `right`: how many are smaller than or equal to `x`.

For a reference of `m` values:

```text
q = (0.5 × (left + right) + 0.5) / (m + 1)
ratio compatibility = 2 × min(q, 1 − q)
```

Averaging left and right handles ties with a middle rank. The `+0.5` and `m+1` adjustment avoids percentiles exactly equal to zero or one.

Example: reference `[1, 2, 2, 4]`, new value 2. Left=1 and right=3, giving `q=(2+0.5)/5=0.5` and compatibility=1. A value above every reference gives `q=4.5/5=0.9` and compatibility=0.2. More reference observations allow finer distinctions in the tails.

The score is high in the center and low in both tails. A value enormously above the maximum has the same rank as a value just above it: its magnitude cannot dominate the score. We do not estimate a variance that one outlier could inflate greatly.

We average the available ratios' scores and call the result **empirical compatibility**, rather than a correctness probability or an exact p-value. Its form resembles a two-sided tail score. Averaging three scores, especially correlated ones, is not a formal method of combining p-values.

A properly defined p-value is the probability of observing data at least as extreme under a specified null hypothesis. It is not the probability that the hypothesis is true. A financially unusual value does not establish that the predicted industry is wrong.

### Missing ratios

`available` counts ratios that have a finite observation and a usable reference profile. Dividing by three gives coverage 0, 1/3, 2/3 or 1. When none is usable, compatibility is 0.5 and availability is 0.

This does not mean “the financial data gives 50% probability.” It is a placeholder accompanied by explicit missingness information. It differs from observed compatibility 0.5 with availability 1. `np.divide(..., where=available > 0)` avoids division by zero; it does not reconstruct missing finances.

## 16. Learning confidence as binary correctness

On the calibration subset, we freeze the ensemble prediction and compare it with the target. This creates a new target:

```text
correct_cal = 1 if prediction == y_cal, otherwise 0
```

This is a second supervised problem with two outcomes: correct or incorrect prediction. A small logistic regression estimates the probability of outcome 1 from five signals:

| Input | Why it may help |
| --- | --- |
| Largest ensemble probability | How strong the leading candidate is. |
| Difference between the largest and second-largest probabilities | How clear its lead is. |
| LR/k-NN agreement | Whether both models would choose the same industry. |
| Financial compatibility | How central the financial values are in the predicted industry. |
| Fraction of available ratios | How much financial evidence was observed. |

For `[0.60, 0.39, 0.01]`, the margin is 0.21; for `[0.60, 0.20, 0.20]`, it is 0.40. The maximum probability is the same, but the leading candidate is more separated in the second case. The calibrator learns whether this signal is useful in the data.

We do not require agreement or compatibility always to increase confidence: coefficients are learned from correctness. Two models may agree on the same error. Financial data might add no information.

The text-only calibrator uses the first three inputs; the financial calibrator uses all five. They share the target and calibration subset. This comparison better isolates the financial contribution than comparing financial calibration only with raw probabilities.

The logistic calibrator also minimizes regularized log-loss, rather than directly minimizing Brier or maximizing F1. We choose it as a simple probabilistic model of correctness. Validation measures whether this training actually improves Brier and the assignment score; improvement is not automatic.

`predict_proba(... )[:, 1]` selects the probability of the “correct” class. It is not the probability of industry code 1: this second model's classes are correctness 0 and 1.

Confidence 0.8 is an estimate to check over many similar cases: among those assigned about 0.8, we should observe roughly 80% correctness. It does not certify an individual prediction. The calibration subset needs both errors and correct predictions; otherwise the binary model has only one outcome to learn from.

## 17. The skeleton's scoring formula

The relevant source is `confidence_score` in `code_skeleton_industry_classification.ipynb`, cells 18–19. The teaching notebook `classification.ipynb` presents a different binary metric that weights TP, FP and FN; that is not this function.

```text
correct_i = 1 if the predicted class is correct, otherwise 0
Brier = mean((confidence_i − correct_i)²)
score = weighted_F1 × (1 − Brier)
```

Weighted F1 weights industries by support. **Confidence is not an F1 sample weight:** it affects the Brier term. Here we compare one confidence per record with binary correctness, rather than compare the full probability vector with every label as in a common multiclass Brier score.

| Outcome | Confidence | Contribution to Brier |
| --- | ---: | ---: |
| Correct | 0.9 | `(0.9 − 1)² = 0.01` |
| Incorrect | 0.9 | `(0.9 − 0)² = 0.81` |
| Correct | 0.2 | `(0.2 − 1)² = 0.64` |
| Incorrect | 0.2 | `(0.2 − 0)² = 0.04` |

Reducing every confidence decreases the cost of errors but increases the cost of correct predictions. Neither absolute certainty nor constant caution is automatically a good strategy. For a group with true correctness frequency `p`, the expected loss of constant confidence `c` is `p(c−1)² + (1−p)c²`; its minimum occurs at `c=p`. This explains why we estimate correctness rather than inflate or shrink numbers arbitrarily.

With confidence in `[0,1]`, Brier lies between zero and one. Confidence changes the Brier term; it cannot raise the assignment score above classification weighted F1.

The assignment states that the final bonus threshold **will be above 66% and will be defined after all submissions have been evaluated on the private test set**. The final value is not known. A local validation score above 66% is therefore not proof of earning the bonus. Private-test performance and the instructor's grading remain outside this evaluation.

In the notebook, **MAIN GRADED SCORE (LOCAL VALIDATION)** identifies the official formula applied to the final financial-calibrated ensemble. It is the local estimate of the bonus metric. Other F1 values compare the required classifiers or diagnose minority-class performance; the original skeleton's example is labelled separately.

## 18. Reliability plots and human review

We divide confidence into bins of width 0.1. For each, we report its count, mean confidence and observed accuracy. The plot compares these means with the ideal diagonal.

A point below the diagonal indicates overconfidence: for example, mean confidence 0.85 but accuracy 0.75. A point above indicates underconfidence. Ten records give a noisier estimate than a thousand, which is why counts are shown. The plot does not establish perfect calibration and depends on the chosen bins.

The current highest bin contains 1,385 records, with mean confidence 93.74% and accuracy 94.87%. The 0.8–0.9 bin contains 617 records, with confidence 85.79% but accuracy 82.01%, so it is somewhat overconfident. The lowest populated bin has just 11 records and should not support a strong conclusion by itself.

The policy flags confidence `< 0.8` for human review. This is the baseline's practical threshold, rather than one required by the instructor or optimized here on validation. Every company remains in the score: difficult cases are not removed to improve the metric.

We examine how many companies would require review and how many errors the policy captures. Raising the threshold can find more errors but increases workload. A real application would need error costs, a review budget and validation of the policy.

At the current 0.8 threshold, 1,169 of 3,171 records are flagged, or 36.9%. These include 496 of the ensemble's 678 errors, about 73.2%. Accuracy is 57.57% among flagged records and 90.91% among unflagged records. The policy therefore concentrates many errors in a smaller subset, at the cost of reviewing more than one third of predictions.

## 19. Inference without targets

`predict_industry(companies, financials=None)` returns two vectors: industry names and confidence values. It prepares the data, converts embeddings, applies the already fitted PCA, obtains probabilities from the already fitted classifiers, combines votes, builds the five signals and applies the already fitted calibrator.

It does not perform another `fit` or require `industry`. The notebook example explicitly removes that column. Output follows input-row order even when the pandas index is not numbered sequentially from zero.

You can pass already joined finances or a separate table for the same company IDs being predicted. Missing values remain missing. Private labels cannot be used, and unprovided financial observations cannot be assumed known.

The attachments do not specify a mandatory function name, a private-test schema or a submission CSV. This simple entry point can be adapted to the actual instructions. We do not invent official formats.

Models remain in memory after execution. The deliverable is a complete notebook, as in the skeleton, rather than a published Python package. Adding classes, frameworks or serialization without a specific need would lengthen the code without helping explain the methods.

## 20. Current results and limits of the evidence

The following results were freshly measured on the supplied original datasets, using ID-grouped splitting with seed 42. All 19 code cells completed in a fresh kernel, all eight plots were produced, and additional checks passed for split separation, probability and confidence bounds, the assignment scorer and prediction over the whole validation set. These numbers reproduce the previously archived metrics; they are now supported by a new execution rather than copied outputs.

Synthetic regression tests serve a different purpose: they check code behavior and edge cases, rather than measure course-data performance. The table below uses the real course data.

| Current method | Accuracy | Weighted F1 | Macro F1 | Brier | Assignment score |
| --- | ---: | ---: | ---: | ---: | ---: |
| LR | 0.781457 | 0.777788 | 0.717515 | 0.137087 | 0.671163 |
| k-NN | 0.777042 | 0.771401 | 0.710471 | 0.141496 | 0.662251 |
| Raw ensemble | 0.786187 | 0.780881 | 0.723058 | 0.137380 | 0.673604 |
| Text-calibrated ensemble | 0.786187 | 0.780881 | 0.723058 | 0.135923 | 0.674742 |
| Financial-calibrated ensemble | 0.786187 | 0.780881 | 0.723058 | 0.135585 | 0.675006 |

The final candidate is the financial-calibrated ensemble. Its **main local graded score is 67.50%**, with **78.09% weighted F1**, Brier **0.135585** and **3,171 validation records**. The notebook prints these together under `MAIN GRADED SCORE (LOCAL VALIDATION)`. The instructor's retained logistic-regression example prints separate example scores; those are not the final candidate's result.

LR is slightly stronger than k-NN among the two required classifiers. The ensemble gains about 0.31 percentage points of weighted F1 over LR. Finances gain about 0.026 percentage points of assignment score over text calibration: a very small change, rather than robust superiority established across datasets. A local score above 66% is not proof of the bonus, because the final threshold will exceed 66% and is decided from private-test submissions afterward.

The fresh cloud run took **87.28 seconds including verification and the additional instructor example**. The upper bound obtained by summing runner and kernel resident memory peaked at **1,594,036,224 bytes, about 1.48 GiB**; this can count shared pages more than once and is not a precise exclusive-memory measurement. These figures cover the Linux cloud runner plus Jupyter kernel, rather than training alone or the student's laptop. The assignment explicitly requires training within 45 minutes on mid-tier hardware with 16 GB RAM. That hardware-specific check remains to be performed on the intended machine.

### Repeated descriptions

Splitting by ID does not prevent different companies from having identical embeddings. The fresh run confirms that 455 validation records, about 14.35%, have an embedding seen in fitting. Ensemble accuracy is 99.34% on those records and 75.15% on the 2,716 unseen ones.

The overall result therefore concerns new companies from a distribution containing repeated descriptions. It does not establish the same quality on entirely new descriptions. The notebook reports the distinction rather than hiding the limitation or removing cases from scoring.

Grouping by description as well would be a stricter, different experiment whose figures would not be directly comparable. The fresh data check finds 4,578 repeated embedding records and 17 identical-embedding groups with conflicting labels. A classifier using only those features cannot distinguish such rows based on information it does not have.

### Hyperparameters and experiments

The previous documentation mentioned parameter comparisons without retaining a complete experimental log here. Therefore, `C=10`, 128 components and 15 neighbors are described as the baseline configuration, rather than experiments newly performed now or mathematically proven optima.

The priority is compact, correct and explainable code. We avoid adding neural networks, many-level stacking or large searches without evidence of usefulness. Fewer lines should not mean compressing everything into unreadable expressions: a few small functions clarify reused steps.

## Development rationale and recorded comparisons

This account reconstructs the logic of the final pipeline from saved evidence. It is not a complete chronological experiment diary. The comparisons below use the recorded real-data executions.

**A. Inspect the data and separate roles.** We first need meaningful units of evaluation. Company-ID grouping keeps repeated company records together, and four subsets separate fitting, voting-weight selection, calibration and evaluation. This protects against several forms of leakage, but the observed repeated embeddings show why ID separation does not ensure new descriptions. The split is not stratified and validation was inspected in earlier development.

**B. Establish simple references.** The supplied C=1 logistic example illustrates the exact assignment scorer: approximately 0.6577 with model confidence versus 0.5972 with confidence always one. It demonstrates that certainty is penalized on mistakes, rather than establishing the best classifier. A majority-class predictor would be a useful conceptual reference for imbalance, but no measured majority-baseline experiment is claimed here.

**C. Explore PCA and compare required classifiers.** PCA compresses 768 coordinates to 128 for k-NN; two plotted components retain 25.287% variance and all 128 retain 87.624%. Visualization shows both concentrations and overlap. LR weighted F1 is 0.777788 versus k-NN's 0.771401, supporting LR as the stronger standalone model on this split. C=10, k=15 and PCA128 were retained settings, not newly optimized choices. Increasing C weakens regularization; increasing k smooths local votes; reducing dimension lowers distance cost but can discard useful signal. None is universally preferable.

**D. Inspect errors before combining.** Both models miss 520 cases, while 173 errors are unique to LR and 187 to k-NN. This suggests complementary information, although many shared errors remain. Common confusion directions include Capital Goods predicted as Materials (28 LR, 33 k-NN), and Technology Hardware & Equipment predicted as Capital Goods (21, 24). Negative class-size/error correlations do not prove that frequency causes those mistakes.

**E. Select the voting weight on tuning.** LR weights 0.25, 0.50 and 0.75 yield tuning weighted F1 of 0.791847, 0.783686 and 0.782247. We therefore retain 0.25 LR and 0.75 k-NN. The weight favors k-NN despite LR's stronger standalone validation result: these are different comparisons on different subsets. Matching probability-column class order is essential before addition. The resulting validation weighted F1 is 0.780881, only about 0.31 percentage points above LR.

**F. Evaluate raw confidence.** Taking the maximum ensemble vote is a simple reference. It produces Brier 0.137380 and assignment score 0.673604. A high vote is evidence of model preference, not automatically a calibrated correctness probability.

**G. Calibrate text-based correctness.** A binary logistic regression learns correct=1 versus incorrect=0 using maximum vote, margin and agreement on calibration companies. This separates predicting an industry from estimating whether that prediction is right. Text calibration gives Brier 0.135923 and score 0.674742. Its regularized log-loss objective differs from Brier, so improvement is an observed result rather than a mathematical guarantee.

**H. Add financial compatibility and coverage.** Fitting-only empirical ranks avoid a blanket Gaussian assumption for ratios with negative values and long tails. Missingness matters: raw EBITDA margin is absent in 65.2% of financial rows, with another 824 infinite values. Middle ranks handle ties; coverage distinguishes an unavailable profile from typical finances. The predefined financial candidate gives Brier 0.135585 and score 0.675006, just 0.026 percentage points above text calibration. All three confidence variants retain the same labels. The validation comparison does not select the final candidate after seeing its score, nor establish a robust financial advantage. Direct financial-feature classifiers and Gaussian models are conceptual alternatives, not recorded failed experiments.

**I. Examine reliability and review workload.** The highest populated bin has mean confidence 93.74% and accuracy 94.87% across 1,385 cases; the 0.8–0.9 bin is overconfident at 85.79% versus 82.01%. Flagging confidence below 0.8 sends 36.9% of cases to review and captures 73.2% of errors. This illustrates a workload/error tradeoff, not guaranteed correction by a human or an optimized deployment policy. Every flagged record remains in evaluation.

**J. Verify reproducibility and limits.** Earlier independent real-data kernels executed all 19 cells in both final notebooks, with eight plots each. Baseline, submission and study predictions matched exactly; probabilities, PCA variance, features and confidence matched within 1e-10. The 87.3-second main cloud execution and 1.48 GiB summed-RSS upper bound are not laptop measurements. Synthetic tests exercise behavior separately from course-data scores. Keeping two required classifiers, one small voting choice and binary correctness calibration makes the system explainable without neural models or a large search. Such alternatives have not been measured here. Private-test quality, stability across new splits and compliance on the intended laptop remain unproven.

## 21. Python concepts that explain the method

- `fit`, `transform`, `predict`, `predict_proba`: learn parameters, apply a learned transformation, select a class, produce probabilities. These are different operations.
- `iloc`: selects by position; splitters return positions rather than necessarily pandas index labels.
- `groupby(...).median()`: groups and summarizes. Grouping by ID prepares one financial row per company; grouping by industry describes fitting distributions.
- `reindex(columns=...)`: adds the required columns with NaN where absent; it does not invent observations.
- `merge`: links information by a key rather than concatenating arbitrary rows.
- Boolean masks: select records satisfying conditions, such as a predicted industry and a finite ratio.
- `np.isfinite`: excludes NaN and infinities, rather than excluding a number merely because it is large.
- `np.sort(..., axis=1)`: sorts votes within each company to find the highest and second-highest values; it does not reorder companies.
- `np.column_stack`: places the five signals side by side to form the calibrator's feature matrix.
- The `profiles` dictionary: maps an industry/ratio pair to a sorted array; it is not another classifier.
- `astype(int)`: converts correctness booleans to targets 1 and 0.
- List comprehension: applies the same operation to several subsets; it does not pool their data during fitting.
- `assert`: checks a required condition; it does not improve the score.
- `perf_counter`: measures elapsed time; it is not a feature.

## 22. Likely oral questions

### “What are the input and output?”

The description embedding is the essential input; available finances support confidence. I return an industry and the estimated probability that this particular chosen label is correct.

### “Why not use the text directly?”

The task provides embeddings. I use the pretrained semantic representation and fit the required classical models without training a language model again.

### “Is PCA a classifier?”

No: it is an unsupervised transformation. LR and k-NN perform classification. PCA supports visualization and reduces distance-computation cost.

### “Why two components in the plot and 128 in the model?”

Two can be plotted, but they discard substantial information. k-NN uses all 128 and LR uses the originals. Retained variance is different from accuracy.

### “If the colors overlap, why attempt classification?”

The two-dimensional projection loses information. Classes overlapping in that plot may be distinguished by coordinates it does not show. Held-out evaluation tests this.

### “Why fit PCA only on fitting if it does not use labels?”

The validation-feature distribution should also remain outside what is learned before those companies are evaluated.

### “What is the difference between LR and k-NN?”

LR learns global weights and applies softmax; k-NN compares local examples and votes. Their probabilities come from different mechanisms and they may have complementary errors.

### “Why can k-NN be slow?”

It searches for neighbors during prediction. PCA reduces the number of coordinates, and caching avoids repeating searches. Fast fitting does not imply cheap prediction.

### “Why C=10 and k=15?”

They are baseline parameters. C inversely controls regularization and k controls how local the vote is. They are not claimed as global optima; establishing better choices would require recorded tuning comparisons.

### “Does LR maximize F1?”

No: it optimizes regularized log-loss. F1 is the metric used to compare decisions and select the ensemble weight among the candidates.

### “Are accuracy and weighted F1 interchangeable?”

No. Accuracy counts correct rows; weighted F1 aggregates per-class precision and recall with support weights. Macro F1 gives an additional perspective with equal industry weights.

### “Why not use balanced class weights?”

They could help small classes but do not guarantee a better official metric. I avoid adding an untested change and show class imbalance through per-class metrics.

### “Does an ensemble always improve performance?”

No. It is plausible when models have complementary information, but needs testing. The fresh real-data run confirms a small gain over LR, rather than a large improvement.

### “Why not select the weight using calibration?”

The calibrator should learn correctness for a system that is already fixed. Using the same targets to select and calibrate the combination would merge those two roles.

### “Why not retrain on 100% afterward?”

Final training can be designed, but it must preserve a valid way to learn confidence. Simply adding calibration companies to the classifiers leaves a calibrator fitted for a different system. This version keeps the system frozen and verifiable without more complex cross-fitting.

### “How do you choose the financial profile?”

From the predicted industry, never the true target. Otherwise I would use an answer unavailable when predicting a new company.

### “Why ranks?”

I avoid assuming normality and limit the influence of extremes. I estimate relative position in the industry's reference sample, with the limitations of finite data.

### “Is it a p-value?”

I call it empirical compatibility. I do not claim formal testing guarantees; it is not the probability that the industry is correct. Averaging ratios' scores is not a formal p-value combination.

### “Do unusual finances mean the industry is wrong?”

No: they may be real company characteristics. Here they contribute to confidence through a model trained on observed correctness, rather than directly changing the label.

### “Why not fill missing values with zero?”

Zero would be an observed ratio, which differs from absence. I use available ratios and a coverage signal. Compatibility 0.5 with zero availability is a placeholder rather than a financial probability.

### “How do you learn confidence?”

On companies not used to fit the classifiers, I compare predictions and targets. I create a correct/incorrect target and fit logistic regression using five signals that can also be calculated without targets during inference.

### “Why is the maximum probability not enough?”

It may be poorly calibrated. Margin, agreement and finances may better describe error frequency. I compare raw, text-calibrated and financial-calibrated Brier scores on the same cases.

### “Does confidence enter F1?”

Not in this skeleton. The score is weighted F1 multiplied by one minus Brier. The teaching materials contain another formula that should not be confused with the project's scoring function.

### “Why not always report low confidence?”

Brier also penalizes correct predictions assigned low confidence. Expected loss is minimized when confidence matches the actual correctness frequency.

### “Does confidence 0.9 certify an individual case?”

No: it is an estimate. Calibration is checked on groups with similar confidence, while also considering how many observations support each group.

### “Did you obtain the bonus?”

That cannot be claimed. The fresh real-data validation gives about 78.09% F1 and 67.50% assignment score, which are different metrics. The final bonus threshold will be above 66% and set after all submissions; neither it nor our private-test score is known. We have verified the local result, not the bonus award.

### “What is the main weakness?”

One split, historically inspected validation, no private test and repeated descriptions across different IDs. Performance on unseen embeddings alone was substantially lower; that distinction is reported.

### “What would be a sensible next experiment?”

The real-data run now reproduces the archived metrics. A sensible next experiment would test one simple change using tuning and a recorded log, rather than add complexity based on validation alone. To investigate generalization, add a description-grouped evaluation while clearly stating that it measures a different scenario. The laptop runtime check is also still useful.

## 23. How to prepare

1. Read the saved outputs and then rerun from a fresh kernel using both supplied datasets. Inspect every table rather than only the final score.
2. Draw the four subsets and identify which objects learn from each: PCA, classifiers, profiles, ensemble weight and calibrators.
3. Combine two three-class probability vectors by hand, each summing to one; find the maximum, margin and predicted class.
4. Calculate precision, recall, F1 and Brier with small numbers: they are different concepts.
5. Calculate percentiles using a short reference array, including a tie and a missing observation.
6. Interpret an overconfident bin and a bin with few records in the reliability plot.
7. Try answering the questions without reading. For every method, explain what it receives, what it learns and what it returns.
8. Follow the instructor's actual submission instructions for the clean notebook and required files. The annotated copy and this guide are study materials unless requested otherwise.

A useful explanation distinguishes **why a choice is reasonable**, **what was measured**, **what remains unproven** and **how it could be checked**. It does not require treating every choice as the only possible one.
