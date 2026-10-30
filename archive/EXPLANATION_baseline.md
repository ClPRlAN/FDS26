# Understanding the industry classification project

Read the notebook from top to bottom. Its printed results and plots describe the actual run; these notes explain the ideas behind them. A validation result is an estimate on the supplied data, not a guarantee of the professor's private-test score.

## 1. What the data represents

A **variable** is a name that refers to a value, such as a number, a table, or a trained model. Assigning `x = 3` gives the name `x` the value `3`. A function accepts inputs and returns an output; for example, a scoring function accepts predictions and returns a number.

A NumPy **array** stores values in a regular shape. A vector is a one-dimensional array; a matrix is a two-dimensional array. If there are `n` companies and `d` embedding coordinates, the feature matrix has shape `(n, d)`. The industry labels and predictions have shape `(n,)`: one value per company. Rows must stay in the same company order when features, labels, and predictions are compared.

A pandas **DataFrame** is a table with named columns. Here, a row describes a company. The company ID links its business-description data to its financial information; it is an identifier, not a meaningful numerical feature.

An **embedding** is a vector of numbers representing the meaning of a company's business description. Descriptions with similar meaning often have nearby vectors. Individual coordinates do not have simple labels such as “manufacturing” or “retail.” The supplied embeddings each have **768 coordinates**; we use them directly as features.

The **target** is the known industry. The model learns a relationship between the features and that target. At prediction time, the target is unknown and must never be included among the input features.

The main dataset contains **31,706 records, 31,705 unique company IDs, and 25 industries**. One company has two different embedding records with the same industry. There are 4,578 repeated embedding strings, including 17 embedding groups with conflicting industry labels. Identical text features can therefore correspond to different targets. Matching financial IDs are available for 29,400 main records; that does not mean every financial ratio is present.

## 2. Why we separate the companies

The notebook first reserves approximately 10% of company-ID groups for validation. It splits the remaining groups into fitting and development portions, then divides development equally into tuning and calibration. Each part has a separate job:

| Part | Approximate share | Records in this split | Purpose |
|---|---:|---:|---|
| Fitting | 81% | 25,681 | Fit classifiers, PCA, and financial reference profiles. |
| Tuning | 4.5% | 1,427 | Choose model settings and the ensemble weight. |
| Calibration | 4.5% | 1,427 | Learn confidence after the industry classifier is fixed. |
| Final validation | 10% | 3,171 | Evaluate the completed system. |

Because a company group can contain multiple records, row percentages need not match exactly. The broader “training portion” includes fitting, tuning, and calibration records.

The **fitting set** teaches the classifiers and PCA. Financial reference profiles also use this set. These fitted parts must not learn from validation companies.

The **tuning set** compares development choices, including probability-voting weights. It is separate from confidence calibration so a choice made using tuning correctness does not influence the calibration targets.

The **calibration set** teaches the confidence adjustment. It contains companies used neither to fit the classifiers nor to choose the ensemble weight. For each company, we compare the fixed prediction with its true industry to obtain a correctness target.

The **validation set** evaluates the completed prediction procedure. Its targets are used to calculate metrics, not to train the classifier or calibrator. Repeatedly choosing settings after inspecting this score makes it less honest as an estimate of performance on unseen data.

**Leakage** means allowing information from an evaluation company to influence the fitted system. Examples include fitting PCA on all companies, choosing a confidence transformation from validation correctness, or placing copies of the same company in both fitting and validation. Financial reference profiles follow the same separation rules as text preprocessing.

`GroupShuffleSplit` keeps all records with the same company ID in one split. The two records of the repeated company are retained together, preventing the model from learning from that company and then being evaluated on another record of it. The financial merge uses `validate='many_to_one'`: multiple main records may match one aggregated financial record, while duplicate financial IDs would fail the check.

A fixed `random_state=42` makes the splits repeatable. The grouping is by company ID, not by embedding string, and the split is not stratified by industry. Class proportions can vary; the label assertion checks that each held-out industry has fitting examples. A rare industry gives fewer examples to learn from and a less stable performance estimate.

## 3. Labels and the required plots

`LabelEncoder` maps industry names to integers, for example `"Retail" → 0`. The integers are identifiers: industry 2 is not twice industry 1. The notebook fits the mapping on fitting companies, then reuses it for tuning, calibration, validation, and probability columns. `inverse_transform` converts predicted integers back to industry names. An assertion checks that every held-out industry exists among the fitting labels.

The **class-frequency plot** shows how many companies belong to each industry. **Class imbalance** means some industries have many more examples than others. A high overall score can conceal poor performance on a small industry, so inspect per-industry results as well.

**PCA**, or principal component analysis, constructs directions that preserve as much variation in the feature vectors as possible. The notebook fits **128 components on fitting embeddings**, using scikit-learn's default automatic solver choice and `random_state=42`. k-NN uses all 128; the scatterplot displays the first two, with one dot per fitting company colored by industry.

The two-dimensional plot is a compressed view. Overlap there does not prove that industries are inseparable in the original features. Logistic regression uses the **full 768-coordinate embeddings**. The printed explained-variance totals describe how much variation two and 128 components preserve. A development comparison informed the choice of 128; it does not establish a global optimum.

## 4. The two classifiers

**Logistic regression** learns a score for each industry from a weighted combination of input coordinates. It converts those scores into probabilities that sum to one across industries. Regularization discourages excessively large weights. Here, `C=10` controls regularization: larger `C` means weaker regularization. `max_iter=1000` limits optimization iterations; it is not a promise that the model needs or completes exactly 1,000 iterations.

**k-nearest neighbors**, or **k-NN**, compares a new company with fitting companies and lets nearby examples vote. This notebook uses **15 neighbors in the 128-dimensional PCA representation**. `weights='distance'` gives closer neighbors more influence. Its probabilities come from weighted neighbor votes and should not automatically be treated as reliable confidence estimates.

Numerical libraries are limited to two CPU threads, and k-NN uses `n_jobs=2`. These are resource choices, not evidence that a runtime target has been met. Development comparisons informed model settings; they do not prove those settings optimal for other splits or datasets.

Both models predict the industry with the highest probability. **Predicted probability** means the model's estimate, not certainty about the real world. A prediction with probability 0.9 can still be wrong.

The **ensemble** combines aligned probability vectors using a weighted average:

`votes = lr_weight × proba_lr + (1 − lr_weight) × proba_knn`

The code compares logistic-regression weights **0.25, 0.5, and 0.75** on tuning weighted F1. The selected weight is **0.25**, so logistic regression contributes 25% and k-NN 75%. Choose the industry with the largest vote. This is soft voting. The smaller weight does not mean logistic regression is unnecessary: it can still affect close decisions.

In development, `C=1` was stronger for logistic regression on its own, while `C=10` contributed better to the ensemble. A model setting can therefore be selected for the final combination rather than its standalone ranking. All these choices precede final validation.

Agreement can provide useful evidence, but both models can learn similar mistakes. Combining models does not automatically improve a score. Financial information adjusts confidence here; it does not change the ensemble's chosen industry.

## 5. Financial information and cleaning

Financial features include profit margins and asset turnover. They may help distinguish industries, but they may also be absent or unusual for valid reasons.

- A **missing value** means information is unavailable; it does not mean zero profit or zero turnover.
- **Infinity** is not a usable finite observation. Treat it as unavailable before calculating statistics.
- An **outlier** is far from typical values. It may be a real company or a data issue. Robust transformations and statistics can reduce its influence.
- A **negative margin** can represent a loss. Removing every negative value would remove valid financial information.
- Repeated company IDs are resolved before joining the tables. `groupby('id')[financial_columns].median()` gives one median per ratio and company, ignoring missing values. This prevents the join from duplicating companies, though aggregation loses distinctions between the original records.

The notebook replaces infinities with missing values, retains valid negatives, and uses per-company medians. Missing observations are excluded from financial rank calculations. Rank-based scores bound the influence of extreme magnitudes. Boxplots hide outlier dots only for readability; this does not delete data. Their symmetric logarithmic axis displays both negative and positive magnitudes.

**Empirical financial consistency** asks whether a company's observed finances resemble fitting companies in its predicted industry. For each industry and ratio, the notebook sorts finite fitting values and retains profiles with **at least 20 observations**. A new value's percentile uses the middle rank for ties and a small offset at the extremes. Its compatibility is `2 × min(percentile, 1 − percentile)`: high near the middle, low in the tails.

Available ratios are averaged. If none are usable, compatibility is a placeholder of 0.5 and availability is 0. Availability otherwise records the fraction of the three usable ratios. This allows the confidence model to distinguish missing evidence from observed compatibility of 0.5.

Compatibility is supporting evidence, not a probability of correctness or a formal combined p-value. A financially unusual company can still have the correct industry, and a typical company can still be misclassified.

Likewise, a **p-value** measures how surprising data would be under a stated statistical model. It is not the probability that a predicted industry is correct. Any financial signal used to adjust confidence must demonstrate useful behavior on held-out companies.

## 6. What the metrics mean

**Accuracy** is the fraction of companies assigned the correct industry.

To evaluate one industry, treat that industry as the positive class and all other industries as negative:

- **TP**, true positives: the prediction and true industry both match this industry.
- **FP**, false positives: the prediction selects this industry, but the true industry is another one.
- **FN**, false negatives: the true industry is this one, but the prediction selects another one.

**Precision** is `TP / (TP + FP)`: the fraction of this industry's predictions that are correct. **Recall** is `TP / (TP + FN)`: the fraction of this industry's actual records found correctly. **F1** combines them as `2 × precision × recall / (precision + recall)`. The report uses zero when a division is undefined.

An industry's **support** is its number of true records in the evaluated set. **Weighted F1** is `sum(support_i × F1_i) / sum(support_i)`, so large industries contribute more. **Macro F1** averages industry F1 scores with equal weight and can reveal weak performance on rare ones.

A **confusion matrix** counts true-versus-predicted industry pairs. The diagonal contains correct predictions; off-diagonal cells show confusion. In this notebook, rows are true industries and columns are predicted industries. The heatmaps normalize each true-industry row to sum to one, making rare industries visible; the printed most-confused pairs use raw counts. Compare the models' errors record by record as well as by industry.

The official bonus metric is:

`score = weighted_F1 × (1 − Brier)`

For this assignment, the Brier score is the mean of `(confidence − correctness)²`. Correctness is 1 for a correct industry prediction and 0 for an incorrect one. Each confidence must be between 0 and 1. This formula concerns the confidence of the chosen prediction; it is not the usual full-vector multiclass Brier calculation.

A correct prediction with confidence 0.9 contributes `(0.9 − 1)² = 0.01`. An incorrect prediction with confidence 0.9 contributes `(0.9 − 0)² = 0.81`. Confident mistakes are expensive. Low confidence on correct predictions also has a cost. Weighted F1 measures classification quality, while Brier measures confidence quality.

The old `confidence ** 6` trick belongs to a different scoring formula. Arbitrarily shrinking confidence does not guarantee a better score under this year's rule. The confidence adjustment must be learned and checked against the current scorer.

## 7. Calibration and human review

**Calibration** connects reported confidence to observed correctness. Among many predictions with confidence close to 0.8, a well-calibrated system should be correct about 80% of the time. Calibration cannot be established from a single prediction.

Two binary logistic-regression calibrators learn from the **1,427 calibration records**. Their target is **1 when the fixed ensemble is correct and 0 otherwise**. These calibrators are separate models from the 25-industry classifier. The text-only calibrator uses three inputs; the financial calibrator uses all five:

1. Largest ensemble probability.
2. Gap between the largest and second-largest probabilities.
3. Whether the two base models agree.
4. Financial compatibility.
5. Fraction of usable financial ratios.

The probability of class 1 is the adjusted confidence. The final validation compares raw ensemble confidence, text-calibrated confidence, and financial-calibrated confidence. These share the same industry predictions and F1; their Brier scores can differ. The financial version is the fixed inference candidate, rather than being selected by whichever validation result looks best.

A **confidence-bin table** groups predictions into ten intervals and compares average confidence with actual accuracy in each interval. Include counts: a bin with only a few examples provides weak evidence. A point below the diagonal in the reliability plot indicates confidence greater than observed accuracy; above the diagonal indicates confidence lower than observed accuracy. Calibration and discrimination are different: a cautious model can be calibrated while still making many classification errors.

A **human-review flag** identifies predictions with confidence **below 0.8**. This is a starting policy, not a course requirement or an optimized threshold. It trades review workload against errors captured. Every company still receives a prediction and remains in the score. The notebook reports the fraction flagged, accuracy in each group, and how many errors the flags capture.

## 8. Reading the Python syntax

Read each expression as an operation on named values. The notebook uses these patterns:

`=` assigns a value; `==` compares values. A colon introduces a function, loop, or condition, and indentation shows which lines belong to it. `len(...)` counts items.

| Syntax | Meaning here |
|---|---|
| `import numpy as np` | Imports a package and gives it a shorter name. `pd`, `plt`, and `sns` are aliases for pandas, plotting, and seaborn. |
| `pd.read_pickle(path)` | Loads the supplied saved DataFrame. |
| `df.merge(..., how='left')` | Attaches finances by ID while retaining every main-data row, including rows with no financial match. |
| `.isna()`, `np.isinf(...)`, `np.isfinite(...)` | Check for missing, infinite, or usable finite values. `np.nan` represents a missing numerical value. |
| `def embeddings(companies):` | Defines a reusable function; `return` sends its result back to the caller. |
| `.apply(ast.literal_eval).tolist()` | Parses each embedding string as values, then collects the lists. `np.array(..., dtype=np.float32)` builds a numerical matrix using 32-bit floating-point values. |
| `groups=df['id']` | Tells the splitter which records belong to the same company. |
| `next(splitter.split(...))` | Takes the first train/test index pair from the split generator. Here `n_splits=1` requests one pair. |
| `df.iloc[train_index]` | Selects rows by integer position. Later indices refer to the DataFrame being split, such as `df_train` or `df_development`. |
| `set(...).union(...).issubset(...)` | Collects held-out industry names and checks that all exist among fitting labels. `assert` stops if the condition is false. |
| `model.fit(X, y)` | Learns a model from features and targets. |
| `pca.fit_transform(X_fit)` | Learns PCA and projects fitting rows. `pca.transform(X_val)` reuses the learned projection without fitting again. |
| `model.predict_proba(X)` | Returns one row per record and one probability column per industry. The calibrators instead have two columns: incorrect and correct. |
| `{'Model': prediction}` | A dictionary stores values under keys; `predictions.items()` supplies each name and prediction. |
| `profiles[industry, column]` | The key is a tuple `(industry, column)`, identifying one financial profile. |
| `return names, confidence` | Returns a tuple of two vectors. `names, confidence = ...` unpacks it. |
| `enumerate(le.classes_)` | Supplies each industry's integer position and name together. |
| `mask = y_fit == label` | Creates a Boolean vector: `True` for rows matching that industry. |
| `values[mask]` | Keeps only rows whose mask is `True`. `&` combines conditions; `~` reverses a Boolean condition. |
| `array.shape[1]` | Reads the number of columns in a matrix; counting starts at zero. |
| `argmax(axis=1)` | Finds the largest-probability column separately in each company row. |
| `[f1_score(...) for weight in weights]` | A list comprehension calculates one score for each candidate weight. `weights[np.argmax(tuning_f1)]` retrieves the best-scoring weight. |
| `np.sort(votes, axis=1)` | Sorts the probabilities within each row. `[:, -1]` takes the largest; `[:, -2]` takes the second-largest. |
| `features[:, :3]` | All rows, columns 0, 1, and 2: the text-only confidence inputs. The upper slice bound is excluded. |
| `np.column_stack([...])` | Places the five feature vectors side by side as a matrix. |
| `np.searchsorted(reference, value, side=...)` | Finds insertion positions in sorted values. Left and right positions identify the span of ties. |
| `np.divide(..., out=..., where=available > 0)` | Divides only where a ratio is available; other rows keep the supplied placeholder. |
| `groupby('id').median()` | Groups records by company ID and summarizes each group using medians. |
| `.loc[mask, column].dropna().to_numpy()` | Selects one industry's ratio, removes missing observations, and converts it to an array. |
| `if len(reference) < 20: continue` | Skips a profile with insufficient observations and proceeds to the next loop iteration. |
| `total[mask] += ...` | Adds each usable ratio's compatibility to the relevant rows. `available[mask] += 1` counts usable ratios. |
| `predict_proba(features)[:, 1]` | Takes the calibrator's probability of class 1: the prediction being correct. |
| `le.inverse_transform(prediction)` | Converts numerical industry codes back into names. |

The results and plotting cells use a few additional table operations:

- `value_counts()` counts records per industry. `.sort_values()` orders a table or column, and `.head(5)` keeps its first five rows.
- `results.append([...])` collects a row of scores. `pd.DataFrame(results, columns=[...])` makes a labeled results table.
- `.T` transposes a table. `.loc[le.classes_]` retains industry rows in the desired order. `zero_division=0` gives precision/F1 a defined value when no predictions support a calculation.
- `.reindex(report.index)` aligns industry counts with report rows before `.corr(...)` computes correlation. A correlation describes association, not cause.
- `np.fill_diagonal(pairs.values, 0)` removes correct-count cells from the printed confusion ranking. `.stack()` lists all true/predicted pairs for sorting; it leaves the heatmap unaffected.
- `.agg(['count', 'median'])` calculates financial observation counts and medians. Named aggregations such as `Count=('Correct', 'size')` name the output and specify the input column and operation.
- `pd.cut(..., np.linspace(0, 1, 11))` groups confidences using 11 equally spaced boundaries, producing ten intervals. `observed=True` reports bins that contain records.
- `plt.xlabel`, `plt.ylabel`, and `plt.title` label a plot. `plt.tight_layout()` helps labels fit; `plt.show()` displays it. `alpha` controls scatter-point transparency.
- `perf_counter()` reads an elapsed-time clock. Subtracting the start reading measures runtime. An expression such as `f'{share:.1%}'` formats a fraction as a percentage with one decimal place.

For a two-dimensional array, axis 0 runs down the company rows and axis 1 runs across the columns. `np.sum(mask)` counts `True` values because they behave as 1, while `False` behaves as 0. Their mean gives a fraction. `.astype(int)` turns a correctness mask into explicit 0/1 targets.

A **generator** supplies results one at a time. The split generator supplies a pair of index arrays: positions retained for training and positions held out. `next` retrieves that pair, tuple unpacking names its two parts, and `.iloc` selects the corresponding records.

`predict_industry(companies)` returns **`(industry_names, confidence)`** in input-row order. The input needs the embedding column and all three financial columns, with unavailable financial observations represented by `NaN`. The function does not read true industry labels.

## 9. Reproducibility and interpretation

Run the notebook from a fresh kernel in cell order. Keep dataset paths, seeds, preprocessing, model settings, and package versions documented. Save the results from the actual run and avoid mixing outputs from different experiments.

Explain both strengths and failures. A useful result says what was evaluated, which companies were held out, and what the observed limitations are. Private-test performance, compliance with a runtime limit, and grading outcomes require their own evidence; a local validation score cannot establish them.

## Oral-exam practice

**Why use embeddings?** They give numerical features that summarize the meaning of the supplied company descriptions.

**Why is the industry code not an ordinary numerical feature?** Its value only identifies a class; its numerical size has no industry meaning.

**Why have a calibration split?** Confidence needs to be learned from predictions on companies the classifier did not train on, while keeping the final evaluation separate.

**Why also have a tuning split?** It chooses settings and voting weights before a separate set teaches confidence, leaving validation for the completed procedure.

**Why group by company ID?** One company has two records; keeping them together prevents learning from one and evaluating on the other.

**Does a PCA plot show everything the model sees?** No. Two components discard information, so the plot is an interpretation aid rather than a complete description of separability.

**How do the required classifiers differ?** Logistic regression learns feature weights; k-NN predicts from the labels of nearby training examples.

**Why can high accuracy hide a problem?** A large industry can dominate accuracy while a rare industry is classified poorly.

**Why is model agreement insufficient?** The models may make correlated mistakes.

**Why keep negative margins?** A negative margin can be a valid loss, not a missing value or error.

**Why is financial consistency not correctness probability?** Typical finances are only supporting evidence, and atypical companies can still have the correct industry label.

**What is calibration's purpose?** To improve the reliability of confidence estimates; evaluation checks whether it succeeds. In this notebook it keeps predicted industries and F1 unchanged.

**Why not reuse last year's confidence exponent?** This year's scorer penalizes confidence errors separately, so that transformation has no general justification.

**What prevents a private-score guarantee?** The private companies and labels are unknown; local validation is an estimate rather than a promise.
