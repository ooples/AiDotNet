# Product Demand Forecasting

This sample demonstrates time-aware regression for product demand forecasting, including:

- Multiple regression with regularization (Ridge, Lasso, ElasticNet)
- Time-based features (seasonality, trends, cyclical encoding)
- Automatic feature selection with Lasso
- Prediction intervals for uncertainty quantification

## What You'll Learn

- How to engineer time-based features for forecasting
- How regularization (L1, L2, ElasticNet) affects model performance
- How Lasso regression performs automatic feature selection
- How to calculate and interpret prediction intervals
- How to analyze seasonal patterns in demand data

## The Problem

Forecast daily product demand based on:

- **Calendar features**: Month, day of week, quarter, week of year
- **Cyclical encodings**: Sine/cosine transformations of periodic features
- **Lag features**: Previous demand values (1-day, 7-day, 30-day)
- **Rolling averages**: 7-day and 30-day moving averages
- **External factors**: Temperature, holidays, promotions

## Regularization Methods Compared

| Method | Penalty | Feature Selection | Best For |
|--------|---------|-------------------|----------|
| **Ridge (L2)** | Sum of squared coefficients | No | All features contribute |
| **Lasso (L1)** | Sum of absolute coefficients | Yes | Sparse models, many irrelevant features |
| **Elastic Net** | L1 + L2 combined | Yes | Correlated features, some selection |

## Running the Sample

```bash
cd samples/regression/DemandForecasting
dotnet run
```

## Expected Output

```
=== AiDotNet Product Demand Forecasting ===
Time-aware regression with regularization and prediction intervals

Training set: 730 samples (2 years of daily data)
Test set: 182 samples (6 months of daily data)
Features (19): month, day_of_week, day_of_month, quarter, week_of_year, is_weekend, year, month_sin, month_cos, dow_sin, dow_cos, lag_1d, lag_7d, lag_30d, ma_7d, ma_30d, temperature, is_holiday, is_promotion

Demand Statistics (Training):
  Min: 105 units
  Max: 412 units
  Mean: 267 units
  Std Dev: 48 units

===========================================================================
         REGULARIZATION COMPARISON (Ridge vs Lasso vs ElasticNet)
===========================================================================

Ridge Regression:
  alpha=0.01   R2=0.4523 MAE=32 RMSE=41
  alpha=0.10   R2=0.4567 MAE=31 RMSE=40
  alpha=1.00   R2=0.4534 MAE=32 RMSE=41
  alpha=10.00  R2=0.4234 MAE=34 RMSE=43

Lasso Regression:
  alpha=0.01   R2=0.4512 MAE=32 RMSE=41 Selected=17/19
  alpha=0.10   R2=0.4534 MAE=32 RMSE=41 Selected=14/19
  alpha=1.00   R2=0.4467 MAE=32 RMSE=41 Selected=9/19
  alpha=10.00  R2=0.3234 MAE=38 RMSE=48 Selected=3/19

Elastic Net Regression (L1 ratio = 0.5):
  alpha=0.01   R2=0.4523 MAE=32 RMSE=41 Selected=18/19
  alpha=0.10   R2=0.4556 MAE=31 RMSE=40 Selected=15/19
  alpha=1.00   R2=0.4489 MAE=32 RMSE=41 Selected=11/19
  alpha=10.00  R2=0.3567 MAE=36 RMSE=45 Selected=5/19

===========================================================================
                           BEST MODELS SUMMARY
===========================================================================

--------------------------------------------------------------------------------
Model           Alpha    R2         MAE          RMSE         Features
--------------------------------------------------------------------------------
Ridge           0.10     0.4567     31           40           19
Lasso           0.10     0.4534     32           41           14
ElasticNet      0.10     0.4556     31           40           15

===========================================================================
                    LASSO FEATURE SELECTION ANALYSIS
===========================================================================

Selected Features (non-zero coefficients):
   1. is_promotion         +15.2341
   2. is_holiday           +12.8765
   3. is_weekend           +8.4532
   4. month_sin            +5.2341
   5. dow_sin              +4.8765
   ...

Eliminated Features (zero coefficients):
      day_of_month         (eliminated)
      quarter              (eliminated)
      ...

===========================================================================
                         PREDICTION INTERVALS
===========================================================================

Demand Forecast with Prediction Intervals:
-----------------------------------------------------------------------------------------------
Day    Predicted    80% Lower    80% Upper    95% Lower    95% Upper    Actual       Status
-----------------------------------------------------------------------------------------------
1      287          244          330          214          360          295          In 80%
2      263          220          306          190          336          251          In 80%
3      271          228          314          198          344          289          In 80%
...

Interval Coverage (first 20 days):
  80% interval: 85.0% coverage
  95% interval: 100.0% coverage

===========================================================================
                         SEASONALITY ANALYSIS
===========================================================================

Average Demand by Month:
--------------------------------------------------
  Jan      234 #######################
  Feb      241 ########################
  Mar      256 #########################
  Apr      267 ##########################
  May      278 ###########################
  Jun      295 #############################
  Jul      302 ##############################
  Aug      298 #############################
  Sep      276 ###########################
  Oct      258 #########################
  Nov      245 ########################
  Dec      289 ############################

Average Demand by Day of Week:
--------------------------------------------------
  Mon      248 ###############################
  Tue      256 ################################
  Wed      262 ################################
  Thu      268 #################################
  Fri      275 ##################################
  Sat      298 #####################################
  Sun      285 ###################################

=== Sample Complete ===
```

## Key Concepts

### Time-Based Feature Engineering

```csharp
// Cyclical encoding for periodic features (prevents discontinuity at boundaries)
double monthSin = Math.Sin(2 * Math.PI * month / 12);
double monthCos = Math.Cos(2 * Math.PI * month / 12);
double dowSin = Math.Sin(2 * Math.PI * dayOfWeek / 7);
double dowCos = Math.Cos(2 * Math.PI * dayOfWeek / 7);

// Lag features capture autocorrelation
double lag1 = previousDayDemand;   // 1-day lag
double lag7 = sameDayLastWeek;     // 7-day lag (weekly pattern)
double lag30 = sameDayLastMonth;   // 30-day lag (monthly pattern)

// Rolling averages smooth noise
double ma7 = last7DaysAverage;     // Short-term trend
double ma30 = last30DaysAverage;   // Medium-term trend
```

### Regularization Comparison

```csharp
// Ridge - L2 regularization (shrinks all coefficients)
var ridge = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 0.1 });

// Lasso - L1 regularization (can zero out coefficients)
var lasso = new LassoRegression<double>(new LassoRegressionOptions<double> { Alpha = 0.1 });

// Elastic Net - combines L1 and L2
var elasticNet = new ElasticNetRegression<double>(new ElasticNetRegressionOptions<double>
{
    Alpha = 0.1,
    L1Ratio = 0.5  // 50% L1, 50% L2
});
```

### Prediction Intervals

```csharp
// Calculate residual standard deviation
double residualStd = Math.Sqrt(residuals.Select(r => Math.Pow(r - mean, 2)).Average());

// Z-scores for confidence levels
double z95 = 1.96;  // 95% confidence interval
double z80 = 1.28;  // 80% confidence interval

// Calculate intervals
double predicted = model.Predict(features);
double lower95 = predicted - z95 * residualStd;
double upper95 = predicted + z95 * residualStd;
```

## Feature Importance via Lasso

Lasso regression automatically identifies the most important features:

| Feature | Impact | Description |
|---------|--------|-------------|
| is_promotion | High (+) | Promotions increase demand significantly |
| is_holiday | High (+) | Holidays drive extra demand |
| is_weekend | Medium (+) | Weekend boost in sales |
| month_sin/cos | Medium | Captures seasonal patterns |
| dow_sin/cos | Medium | Captures weekly patterns |
| lag features | Medium | Recent demand correlates with future |

## Interpretation Guidelines

### R2 Score for Time Series

- R2 > 0.5: Good for demand forecasting
- R2 > 0.3: Acceptable (capturing main patterns)
- R2 < 0.3: Need more features or different model

### Prediction Interval Coverage

- 95% intervals should contain ~95% of actual values
- If coverage is lower, intervals are too narrow (overconfident)
- If coverage is higher, intervals may be too wide

### Feature Selection Trade-off

- More features (Ridge): Better fit but harder to interpret
- Fewer features (Lasso): Simpler model, may miss patterns
- Balanced (ElasticNet): Often the best compromise

## Next Steps

- [PricePrediction](../PricePrediction/) - Bayesian hyperparameter optimization
- [BasicRegression](../../getting-started/BasicRegression/) - Simpler regression example
- [AnomalyDetection](../../clustering/AnomalyDetection/) - Detect unusual demand patterns
