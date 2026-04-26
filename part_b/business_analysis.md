# Part B: Business Case Analysis

---

## B1. Problem Formulation

### B1(a)

This is a **supervised regression problem**. The target variable is `items_sold` — the number of items sold at a store in a given month under a given promotion. At inference time it doubles as a recommendation engine: run the model five times per store (once per promotion type), pick the one with the highest predicted output.

Candidate input features: store size, location type, competition density, monthly footfall, promotion type, calendar features (month, is_weekend, is_festival, season), and historical sales patterns by store and location.

Regression is the right framing because the output is a continuous count. Classification would lose information by bucketing sales into bands, and the business needs a ranked comparison across promotion options — that requires numeric predictions.

---

### B1(b)

Revenue is a product of price × volume. When a promotion reduces price (e.g., flat discount), revenue may drop even if far more items are sold. This makes revenue a noisy signal — the same promotion can look like a failure on revenue and a success on volume simultaneously.

`items_sold` directly measures customer response: how many people bought something. It isn't affected by markdown depth, pricing strategy, or mix shift across product categories.

The broader principle: the target variable should directly measure the behavior you want to influence, be stable across the conditions you're testing, and not be confounded by variables outside the model's control. Revenue fails the last two criteria here.

---

### B1(c)

A single global model treats all 50 stores as identical, which they aren't. A rural store and a flagship urban store have completely different footfall, basket sizes, and customer demographics — training one model across both forces it to average out real structural differences.

Better approach: **train separate models per location type** (urban / semi-urban / rural). Three models instead of fifty keeps it manageable while still capturing the biggest source of variation. Each model gets a cleaner signal from stores that actually behave similarly.

A more compact alternative is to include `store_id` as a categorical feature (or a location-type embedding) so a single model still learns store-specific effects without needing explicit splits.

---

## B2. Data and EDA Strategy

### B2(a)

Join order:
1. Start with **transactions** as the base (this is the grain table)
2. Join **store attributes** on `store_id` — adds size, location type, competition density, footfall
3. Join **promotion details** on `promotion_id` or `promotion_type` — adds promotion metadata
4. Join **calendar** on `transaction_date` — adds `is_weekend`, `is_festival`

Final dataset grain: **one row = one store × one calendar month × one promotion type**

Aggregations before modeling: `SUM(items_sold)`, `SUM(revenue)`, `COUNT(transactions)` (footfall proxy), and `AVG(basket_size)` — all grouped at the store-month-promotion level.

---

### B2(b)

Four analyses before modeling:

**1. Promotion-wise average items_sold (bar chart)** — shows which promotions drive more volume on average. If BOGO consistently outperforms others, that's a strong prior to encode. Influences whether promotion_type needs interaction terms.

**2. Heatmap: location_type × promotion_type → mean items_sold** — directly checks if promotion effectiveness varies by location. If it does, that's the clearest signal to build location-stratified models instead of a single global one.

**3. Monthly time series of items_sold** — reveals seasonality peaks (festive months, year-end) and overall trends. Determines whether we need lag features, month embeddings, or to be careful about which months end up in train vs test.

**4. Distribution of items_sold (histogram + boxplot by store)** — checks for skew and outliers. Heavily right-skewed target may benefit from a log transform. Outlier stores (e.g., a flagship that sells 10× the median) can distort models and may need separate handling.

---

### B2(c)

If 80% of records have no promotion, the model sees very few examples of how promotions shift behavior. It will tend to predict close to the baseline (no-promotion) values regardless of which promotion is passed in — effectively learning that promotion type doesn't matter much.

Steps to address:
- **Oversample promoted records** in training using random oversampling or SMOTE
- **Stratify the train-test split** by promotion presence so both sets have similar ratios
- **Explicit interaction features** (e.g., `promotion_type × location_type`, `promotion × is_festival`) give the model more signal to work with from limited promotion examples
- As a sanity check after training: compare predicted items_sold for promo vs no-promo for the same store — the gap should be directionally sensible

---

## B3. Model Evaluation and Deployment

### B3(a)

With 3 years of monthly data across 50 stores, the split should be time-ordered: train on months 1–24, validate on months 25–30, test on months 31–36. Never shuffle before splitting.

Random split is wrong because sales data has temporal structure — seasonality, trends, and promotion carry-over effects mean future months are not independent of past ones. Training on December and predicting January while knowing next December's patterns is unrealistic and inflates metrics.

Metrics:
- **RMSE** — primary metric; penalizes large misses more, which matters because a badly wrong recommendation (predicting 800 items for a promotion that only drives 200) has real cost
- **MAE** — easier to communicate: "predictions are off by X items on average"
- **MAPE** — useful for comparing error across stores of very different sizes without the large-store bias in RMSE

---

### B3(b)

The model recommends differently for Store 12 in December vs March because the input features are different — specifically, December likely has `is_festival=1`, higher historical footfall, and a different month encoding. The model learned that loyalty points work better when customers are already in buying mode (festive season), while flat discounts drive more conversions when traffic is lower.

To investigate: extract SHAP values for both predictions. A SHAP waterfall plot for each shows exactly which features pushed the output toward each promotion and by how much. For December, `is_festival` and `month` likely have large positive contributions toward loyalty points. For March, `competition_density` or baseline footfall may be the dominant drivers shifting the recommendation.

To communicate to marketing: "In December, festive traffic means customers need less price incentive — loyalty points keep margins healthier while still rewarding repeat visits. In March, foot traffic is lower and a flat discount is doing more work to convert casual visitors."

---

### B3(c)

**Saving the model:** after training, serialize the full pipeline (preprocessor + model) using `joblib.dump()`. This captures encoding, scaling, and model weights as one artifact so training and inference are guaranteed to use identical transformations.

**Monthly inference:** at the start of each month, a script assembles the feature table — one row per store per promotion type (250 rows for 50 stores × 5 promotions). It pulls the latest store attributes, calendar flags for the coming month, and any updated competition data. The saved model is loaded and `.predict()` is called on all 250 rows in one shot. The row with the highest predicted items_sold per store becomes the recommendation.

**Monitoring:**
- After each month, compare predicted vs actual items_sold for stores that followed the recommendation. Track rolling RMSE and MAE. If error drifts 20%+ above baseline, trigger a review.
- Monitor input feature distributions monthly (footfall, competition_density) using summary statistics. A significant shift in these indicates the training distribution is no longer representative.
- Retrain on a rolling 24-month window every quarter. Log model version and performance on a held-out recent month before promoting the new model to production.
- Maintain a simple fallback rule (e.g., the historically best-performing promotion per location type) in case the model is flagged for degradation and a retrained version isn't ready yet.
