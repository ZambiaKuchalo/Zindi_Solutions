
# IBM SkillsBuild Hydropower Climate Optimisation Challenge

This repository contains the complete pipeline used for the IBM SkillsBuild Hydropower Climate Optimisation Challenge hosted on Zindi. The goal is to forecast energy consumption (kWh) using time-series data from hydropower monitoring systems enriched with climate features.

## 🧠 Approach

The solution is built using a **stacked ensemble** of gradient boosting models: CatBoost, LightGBM, and XGBoost, with a meta-learner (XGBoost) trained on their predictions. It includes:

- Feature engineering with `Polars` for high performance.
- Time-based cross-validation.
- Climate feature extraction with lags, rolling statistics, EMAs, and seasonal/event markers.
- Hyperparameter tuning via `Optuna`.
- Final weighted ensemble with optimization for minimum RMSE.

---

## 📁 Files

- `ibm_skillsbuild_hydropower_climate_optimisation_challenge_polars.py`: Complete notebook/script containing EDA, feature engineering, model training, ensemble logic, and submission generation.

---

## 🛠️ Dependencies

Install dependencies with:

```bash
pip install polars sktime optuna pmdarima openpyxl workalendar catboost lightgbm xgboost pytorch-tabnet scikit-learn fastexcel
```

---

## 📊 Data

- Dataset is downloaded via `kagglehub` from the user: `kafwayakafwaya/africahydropowerclimateoptimisation`.
- Includes:
  - `HydropowerClimateOptimisation.parquet`: Sensor readings.
  - `Kalam Climate Data.xlsx`: Climate measurements.
  - `SampleSubmission.csv`: Prediction template.

---

## ⚙️ Pipeline Overview

### 1. **Data Preprocessing**
- Uses `Polars` to load, clean, and filter sensor and test datasets.
- Extracts date, device ID, and user ID from the `ID` column.

### 2. **Feature Engineering**
- Hydropower data aggregated daily by device and user.
- Climate data transformed with:
  - Daily aggregations
  - Wind speed/direction, heat index
  - Lagged rainfall
  - Rolling stats (mean/sum/max)
  - Event flags (droughts, heatwaves, frost)
  - Cyclical encodings

### 3. **Model Training**
- 3 base models:
  - `CatBoostRegressor`
  - `LightGBM`
  - `XGBoost`
- Hyperparameter tuning via Optuna (with GPU support where applicable).
- TimeSeriesSplit (10-fold) for robust cross-validation.

### 4. **Stacking & Meta Learner**
- Out-of-fold predictions from base models form meta features.
- Meta learner (`XGBoostRegressor`) trained on stacked predictions.
- Further optimized blending via:
  - Grid search on weighted averages
  - Constrained weight optimization (SLSQP)

### 5. **Prediction & Submission**
- Final predictions clipped to non-negative values and rounded.
- Output saved as `tt-model-ensemble_final_test_pred-2-ROUNDED-CLIP-IBM-SHCOC-tt.csv`.

---

## 🧪 Results

- **RMSE (Meta Learner on training data)**: ~3.89
- **Best Weighted Blend (Grid Search)**: Optimal RMSE with weights for XGB, LGB, CAT.

---

## 📈 Feature Importance

Visualizations of feature importances (via `gain`) for XGBoost included for insight into influential predictors.

---

## 🔍 Highlights

- Fully automated pipeline: EDA → Feature Engineering → Training → Ensemble → Submission
- Uses modern `Polars` for ultra-fast processing
- Advanced climate signal engineering to model environmental impact on energy usage
- Modular design suitable for further experimentation and model stacking

---

## 📜 Citation

If you use or reference this solution, please consider citing the original competition page:

> IBM SkillsBuild Hydropower Climate Optimisation Challenge  
> [https://zindi.africa/competitions/ibm-skillsbuild-hydropower-climate-optimisation-challenge](https://zindi.africa/competitions/ibm-skillsbuild-hydropower-climate-optimisation-challenge)

---

