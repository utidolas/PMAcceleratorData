# How to RUN
* Run all the cells in the notebook
# Forecast future weather trends

**dataset avaiable at: [https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code)**

# Data Preprocessing & Cleaning

- pandas library to open, get sample, summary (describe()) and analyze columns of the dataset

## Null values

- checking NULL values with "isnull().sum()" - NONE found

## Outliers & anomaly detection

- IQR did not work (24,163 outliers) 
  - Weather data has some extreme values (typhoons, heatwaves, etc) that are not outliers, and IQR flag them as so. Hence, IQR does not work well here
- Hard limits based on meteorology (upper and lower bounds)
- Clip 'impossible' values - values outside physically recorded world records
- Isolation Forest + Z-score (Per-city)
  - I.F. is Multivariate - catches plausible combinantions of features jointly anomalous, scales well with 130,000 rows
  - Global z-score would not work since some countries are way hotter/colder than others, cause it would flag anomalies incorrectly in extreme temperatures. Per-city makes it geographically aware
  - Combine both to get a high-confidence anomalie detector
- Fill any NaNs in features with median before fitting
  - median instead of mean because weather values are too skewed and mean would pull to extreme values
- z-score of 3.5 to cover higher extremities 
- Visualize anomaly across features & summary table with most extreme anomalities
  - Confirm which variables drive the detection

## Normalization & EDA

- Weather value has different scales for data
  - gotta normalize, so the model does not get biased to the highest magnitude value
- Split by date instead of train_test_split, so we dont allow the model to see the future with random shuffle
  - sort by 'last_updated' and take first 80% as training, 20% as test
- Select features to scale 'temperature_celsius', 'wind_kph', 'pressure_mb', 'precip_mm', 'humidity'
- apply log transformation to precip_mm 
  - it is too skewed. Most of them are 0mm with some large values in rare heavy-rain events; compress into a narrow range.
  - log1p instead of log to handle zero-precipitation readings
- StandardScaler
  - fit on train, transform on both
  
- Visualize Feature Relationship with spearman heatmap 
![Spearman Graph](/SpearmanCorrHeatMap.png)

- Visualize global temp trend (scaled)
![Global Temperature Trend Graph](/GlobalTempTrend.png)

- Visualize ACF / PACF graph
![ACF vs PACF Graph](/ACFvsPACF.png)

- Visualize Global Monthly Temperature and Precipitation 
![Global Monthly Temperature and Precipitation ](/TempVsPrecip.png)
# Model Building

**Architecture chosen: XGBoost + LightGBM + Ensemble of both**

- both models are gradient boosted decision tree frameworks, which is a good approach for structured/tabular forecasting.
- handle many cities X time natively without building a model for each city
- built-in feature importance
- Ensemble them to reduce variance/noise

## XGBoost

- XGBoost cant explot sequential temporal dependencies (treat each row independently with no awareness of which rows preceded it.), need to 'create it'
  - sort by last_update - ensure lag and rolling features are computed in the correct chronological direction
  - cyclical time encodings - encode with sine and cosine to preserve periodicity for the model
  - lag features per city - temperature from previous timestamp in X city may not be informative for Z city. Group location_name befire applying shift() to prevent cross-location leakage
  - rolling stats - rolling mean captures local short-term trend; rolling standard deviation captures recent volatility; use shift(1) before rolling to prevent the current observation from contaminating its own context window.
- separate train and test sets (chronological split)
- drop NaN lag features created by shift() on the first observation of each city
- separate feature columns and target var (temperature_celsius)
- TimeSeriesSplit to split with respect to time series data with cross validation
  - it trains on a strictly earlier period and validates on a later period
  - replicates the real forecasting
  - provides multiple train/validation splits across timeline
- build XGBoost model, apply L1 and L2 regularization 
  - many correlated features; L1 to perform automatic feature selection; L2 to prevent single feature from accumulating an outsized coefficient
  - reduce overfitting, improve generalization
- cross validation "for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):"
  - .iloc[tr_idx] respects temporal order set earilier 
  - XGB regressor is instantiated inside each fold so that no tree carries over between them; each fold's metric is a independent estimate of performance
- inverse temperature so we can read better 
  - model was scaled, numbers would be different if do not inverse
- MAE, RMSE, R², sMAPE, Max Error - stantard metrics for evaluation
  - aggregate metrics computed across the 130k rows spanning every cities
- Visualize predictions by city 
  - capture temporal pattern for individual location
![Tokyo pred XGBoost](/XGBTokyoForecast.png)

## LightGBM & Ensemble

- 5-Fold TimeSeriesSplit CV 
  - same tscv object is used to ensure models are evaluated on identical temporal folds
- LightGBM uses callbacks API for early stopping (stopping_rounds=50)
- joint OOF loop on the same TimeSeriesSplit folds
  - ensure both models see identical training/validation splits
- OPTIMIZE ENSEMBLE WEIGHT ON OOF 
  - optimal bledn weight is found by minimizing `MAE(y_train, w × XGB_OOF + (1−w) × LGB_OOF)`  over `w ∈ [0, 1]` using  `scipy.optimize.minimize_scalar` with the  `bounded` method - Brent's algorithm

## Visualization

- Feature Importance
![Feature Importance Graph](/FeatImportanceBothModel.png)

- Model Performance Comparison
![Feature Importance Graph](/XGBoostVsLightVsEnsemble.png)

- GLobal Monthly Mean Comparison
![Global Monthly Mean Graph](/GlobalMonMeanComp.png)

## Unique Analysis
__"other" group are countries that could not be mapped by the normalization__
### Temperature Distribution & Average Precipitation by continent
* Africa does not have the highest temperature distribution, it has many hotter days than any continent though. Similar to Asia, North America and Europe
* South America is pretty stable
* Precipitation is similar to every continent but Oceania (high rainfall such as seen in Papua New Guinea)
![Temp and Precipt by continent](/GeographPattern.png)

### 15 coldest and hottest countries
* Hottest dominated by Arabian Peninsula and tropical Asia
* Coldest countries are actually __warmer than the true national average__ for large continental countries (like Canada and Russian) because of the dataset where only major urban centers are present
![top 15 countries](/top15.png)

### Seasonal temperature cycle by continent
* Seasonal cycle changes for south and north continents; summer/winter peak according to the hemisphere 
* Africa and Oceania are nearly flat while Europe has high variance
* South America shows a weak inverse cycle than the others due to some countries being close to the equator offset (like Colombia and Brazil)
![season cycle by continent](/seasonCycle.png)

### Climate variability by continent
* Asia and Europe have the greates variability given their hot and cold countries
* Africa and Oceania have the smallest standard deviations (5.0°C and 5.7°C) due their stable warm continents
* South America has a low variability indicates that most cities in the dataset for South America are tropical and subtropical. Patagonian Argetina and high-altitude Andean cities that are colder probably are absent
![Climate Variability](/ClimateVariability.png)

__the dataset is structurally biased toward warm accessible, urban centres.__