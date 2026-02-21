# Cherry Blossom Prediction - Improvement Strategies Research
## Based on 2024-2025 Competition Winners & Research

**Current Status:** MAE = 4.23 days
**Target:** MAE < 4.0 days (Competition winning level: 3.5-4.0)
**Gap to Close:** ~0.23-0.73 days

---

## 🏆 What Competition Winners Are Doing

### 1. The "Beat the Bot" Insight
> "Science is not algorithmic and still requires some amount of human ingenuity...with a little bit of thought, a little cleverness and some data manipulation, a human can beat the bot"

**Key Takeaway:** While advanced ML helps, **domain knowledge + clever features** beat pure algorithms.

### 2. ChatGPT/XGBoost Winner Approach (2025)
A notable 2025 entry used XGBoost with these key features:
- ✅ **Dormancy release metrics**
- ✅ **Cumulative growing degree days** (we have this!)
- ✅ **Winter chill** (we have this!)
- ✅ **Spring warmth indicators**
- ✅ **Location-specific calibration**
- ✅ **Cross-validation for hyperparameter tuning**

**Our Status:**
- ✅ GDD: We have GDD_winter, GDD_jan_feb, GDD_nov_dec
- ✅ Chilling: We have chill_hours_winter, chill_hours_nov_dec
- ⚠️ Spring warmth: We could add March/April GDD
- ⚠️ Dormancy release: We could add explicit dormancy metrics
- ⚠️ Location calibration: Currently using one global model

---

## 📊 Recent Research Findings (2024-2025)

### Growing Degree Days & Chilling Hours

**From 2024-2025 Phenology Research:**

#### Model Performance Benchmarks:
- Growing Degree Day models: **67% variance explained**
- Biome-BGC phenology model: **79% variance explained**
- Number of Growing Days model: **73% variance explained**
- Chilling Days-GDD hybrid: **68% variance explained**

#### Critical Insights:
1. **Chilling × Heat Interaction is Crucial**
   > "For deciduous temperate crops, it is crucial to consider the interactions between chilling requirements and heat accumulation to improve model precision"

2. **Base Temperature Selection Matters**
   - GDD models performed better for **84% of leaf unfolding** and **70% of flowering**
   - Performance depends on **optimal T_base and t0 selection**
   - Species/cultivar-specific calibration improves accuracy

3. **Temporal Windows Matter**
   - Different accumulation periods (Nov-Dec vs Winter vs Jan-Feb) capture different biological signals
   - We already have this variety - good!

### XGBoost & Machine Learning Best Practices

**From 2024-2025 Crop Yield/Phenology Studies:**

#### Feature Engineering:
- **Phenology-derived growth windows improved accuracy by ~10%** vs fixed seasons
- **Recursive Feature Elimination (RFE)** + **Mutual Information Regression** for feature selection
- Ground-based phenology with XGBoost: R² = 0.668, MAE = 0.84

#### What This Means for Us:
- ✅ Use location-specific growth windows (not calendar months)
- ⚠️ Apply feature selection (RFE) to our 58 features
- ⚠️ Consider mutual information for feature ranking

---

## 🚀 Advanced Time Series Techniques (2025 Research)

### Lag Features & Dynamic Lagging

**From 2025 E-Commerce & Energy Forecasting Research:**

#### Multi-Lag Approach:
- **Different lag sizes are better for different time series** in the same dataset
- **Ensemble of models with different lags** outperforms single-lag models
- Hybrid Prophet-XGBoost-CatBoost with multi-lag: **~5% MAPE reduction**

#### Recommended Lags for Our Problem:
```
lag_1 = previous year bloom (t-1)
lag_3 = 3-year rolling average (t-1 to t-3)
lag_5 = 5-year rolling average (t-1 to t-5)
lag_trend = slope of last 10 years
```

### Location-Specific Modeling

**From 2025 Forecasting Research:**

> "User-specific behavioral variability...can be addressed through individualized behavioral modeling approaches"

**Applied to Cherry Blossoms:**
- Kyoto (continental Asia) ≠ Washington DC (mid-Atlantic) ≠ Vancouver (Pacific coast)
- Different microclimates, different chill requirements, different warming trends
- **Recommendation: Train 5 separate models** (one per location)

### Ensemble Methods

**From 2025 Hybrid Architecture Research:**

#### Prophet-XGBoost-CatBoost Ensemble:
- **Prophet:** Captures long-term trends and seasonality
- **XGBoost:** Handles non-linear interactions
- **CatBoost:** Robust to categorical features and missing data

#### Weighted Ensemble Strategy:
```
Final prediction = 0.4 × GBR + 0.3 × XGBoost + 0.2 × LightGBM + 0.1 × Linear
```

**Why This Works:**
- Different models make different errors
- Averaging reduces variance and overfitting
- More robust to outliers

---

## 🎯 Actionable Improvements (Ranked by Expected Impact)

### **TIER 1: High Impact, Medium Effort** ⭐⭐⭐

#### 1. Add Lag Features (Previous Year Bloom)
**Expected Gain:** 0.2-0.4 days MAE improvement
**Effort:** 30 minutes
**Implementation:**
```python
# Add to feature engineering
df['bloom_lag1'] = df.groupby('location')['bloom_doy'].shift(1)
df['bloom_avg_3yr'] = df.groupby('location')['bloom_doy'].rolling(3).mean().shift(1)
df['bloom_trend_5yr'] = df.groupby('location')['bloom_doy'].rolling(5).apply(
    lambda x: (x.iloc[-1] - x.iloc[0]) / 5
).shift(1)
```

**Why It Works:**
- Trees have "memory" - bloom timing shows autocorrelation
- Recent history predicts near future better than distant past
- Captures year-to-year momentum

#### 2. Location-Specific Models
**Expected Gain:** 0.3-0.6 days MAE improvement
**Effort:** 1 hour
**Implementation:**
```python
# Train 5 separate models
for location in ['kyoto', 'liestal', 'washingtondc', 'vancouver', 'nyc']:
    loc_data = df[df['location'] == location]
    model = train_model(loc_data)
    models[location] = model
```

**Why It Works:**
- Different locations have different climate patterns
- Hyperparameters can be tuned per location
- Removes cross-location noise

#### 3. Add Spring Warmth Features
**Expected Gain:** 0.1-0.3 days MAE improvement
**Effort:** 20 minutes
**Implementation:**
```python
# Add to master dataset processing
GDD_march = calculate_gdd(temp_march, base_temp=0)
GDD_feb_mar = calculate_gdd(temp_feb_march, base_temp=0)
GDD_spring = GDD_feb + GDD_march + GDD_april_first_half
```

**Why It Works:**
- Current GDD stops at winter (Jan-Feb)
- Spring warmth triggers actual bloom
- Fill the missing temporal gap

---

### **TIER 2: Medium Impact, Low Effort** ⭐⭐

#### 4. Feature Selection with RFE + Mutual Information
**Expected Gain:** 0.1-0.2 days MAE improvement
**Effort:** 30 minutes
**Implementation:**
```python
from sklearn.feature_selection import RFECV, mutual_info_regression

# Remove bottom 20% features
mi_scores = mutual_info_regression(X, y)
keep_features = features[mi_scores > mi_scores.quantile(0.2)]
```

**Why It Works:**
- Too many features (58) can cause overfitting
- Remove noise, keep signal
- Research shows ~10% accuracy improvement

#### 5. Chilling × GDD Interaction Features
**Expected Gain:** 0.1-0.2 days MAE improvement
**Effort:** 15 minutes
**Implementation:**
```python
df['chill_x_GDD_winter'] = df['chill_hours_winter'] * df['GDD_winter']
df['chill_x_GDD_spring'] = df['chill_hours_nov_dec'] * df['GDD_march']
df['chill_deficit'] = optimal_chill - df['chill_hours_winter']  # if known
```

**Why It Works:**
- Research emphasizes chilling × heat interaction
- Not enough chill → delayed bloom even with warmth
- Captures biological requirement

#### 6. Hyperparameter Tuning with CV
**Expected Gain:** 0.1-0.3 days MAE improvement
**Effort:** 1-2 hours (run in background)
**Implementation:**
```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [600, 800, 1000, 1200],
    'learning_rate': [0.01, 0.015, 0.02, 0.025],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.7, 0.8, 0.9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(gbr, param_grid, cv=5, n_iter=50)
```

---

### **TIER 3: High Impact, High Effort** ⭐

#### 7. Ensemble of Multiple Models
**Expected Gain:** 0.2-0.4 days MAE improvement
**Effort:** 2-3 hours
**Implementation:**
```python
from sklearn.ensemble import VotingRegressor, StackingRegressor
import xgboost as xgb
import lightgbm as lgb

ensemble = VotingRegressor([
    ('gbr', GradientBoostingRegressor(...)),
    ('xgb', xgb.XGBRegressor(...)),
    ('lgb', lgb.LGBMRegressor(...)),
    ('rf', RandomForestRegressor(...))
])
```

**Why It Works:**
- Different models make different errors
- Ensemble averages out mistakes
- Proven 5% improvement in research

---

## 📋 Recommended Implementation Order

### **Quick Wins (Next 1 Hour):**
1. ✅ Review error analysis (running in background)
2. ⚡ Add lag features (30 min)
3. ⚡ Add spring GDD features (20 min)
4. ⚡ Add chilling × GDD interactions (10 min)

**Expected Result:** 4.23 → **3.8-3.9 days MAE**

---

### **Medium Effort (Next 2-3 Hours):**
5. 🎯 Train location-specific models (1 hour)
6. 🎯 Feature selection with RFE (30 min)
7. 🎯 Hyperparameter tuning (1-2 hours background)

**Expected Result:** 3.8 → **3.5-3.7 days MAE** (Competition winning!)

---

### **Polish (If Time Permits):**
8. 🏆 Ensemble with XGBoost + LightGBM (2 hours)
9. 🏆 Dormancy release metrics (research)

**Expected Result:** 3.5 → **3.3-3.5 days MAE** (Top tier!)

---

## 🔍 What to Look for in Error Analysis

When error analysis completes, check for:

### 1. Location Bias
- [ ] Is one location much worse than others? → Location-specific model needed
- [ ] Is MAE variance high between locations? → Different calibration needed

### 2. Temporal Trends
- [ ] Are recent years (2015-2025) worse? → Need better climate change features
- [ ] Is there a systematic drift over time? → Add year interactions

### 3. Extreme Event Handling
- [ ] Are very early/late blooms poorly predicted? → Add anomaly detection
- [ ] Is error higher at the tails? → Robust loss function needed

### 4. Systematic Bias
- [ ] Mean error > 0.5 days? → Apply bias correction
- [ ] Over-predicting or under-predicting consistently? → Calibrate predictions

### 5. Feature Importance
- [ ] Are enhanced features in top 10? → Keep them
- [ ] Are some features at 0% importance? → Remove them

---

## 📚 Sources

### Competition Insights:
- [International Cherry Blossom Prediction Competition](https://competition.statistics.gmu.edu/)
- [AI joins George Mason University's Cherry Blossom prediction competition](https://www.fairfaxtimes.com/articles/ai-joins-george-mason-university-s-cherry-blossom-prediction-competition/article_72f752dc-fab6-11ef-93fb-d7797413435d.html)
- [Cherry blossoms—not just another prediction competition](https://statmodeling.stat.columbia.edu/2024/02/05/cherry-blossoms-not-just-another-prediction-competition/)
- [XGBoost Approach to Predict Cherry Blossom Peak Bloom (2025)](https://dkepplinger.org/cherry/chatgpt_entry-2025.html)

### Phenology Research:
- [Advances in Growing Degree Days Models (2024)](https://www.mdpi.com/2311-7524/11/12/1415)
- [Growing degree day models to predict spring phenology](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/1365-2435.70020)
- [Comparison of Phenology Models for Northern Hemisphere](https://pmc.ncbi.nlm.nih.gov/articles/PMC4184861/)
- [Role of phenology in crop yield prediction (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0168192324004532)

### Time Series & ML Methods:
- [Enhanced load forecasting with multi-lag feature engineering (2025)](https://www.sciencedirect.com/science/article/pii/S0360544225036230)
- [Dynamic Lagging for Time-Series Forecasting (2025)](https://arxiv.org/abs/2509.20244)
- [Lag Selection for Time Series Forecasting using Deep Learning (2024)](https://arxiv.org/html/2405.11237v1)
- [Ensemble decomposition for temperature forecasting (2024)](https://link.springer.com/article/10.1007/s44196-024-00667-6)

---

**Next Steps:** Apply Tier 1 improvements immediately while error analysis runs.
