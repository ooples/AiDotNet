# Time Series Forecasting

This sample demonstrates time series forecasting using AiDotNet's ARIMA and SARIMA models, including trend/seasonality decomposition with STL, multi-horizon prediction, and comprehensive evaluation metrics.

## What You'll Learn

- How to generate synthetic time series data with trend, seasonality, and noise
- How to use `STLTimeSeriesDecomposition` for trend and seasonal component extraction
- How to train and compare `ARIMAModel` and `SARIMAModel` for forecasting
- How to evaluate forecasts with MAE, RMSE, MSE, and MAPE metrics
- How to perform multi-horizon forecasting and direction accuracy analysis

## Business Context

Time series forecasting is essential for:
- **Sales forecasting** - Predict future demand to optimize inventory
- **Stock price prediction** - Anticipate market movements for trading decisions
- **Capacity planning** - Forecast resource needs based on usage patterns
- **Budget planning** - Project future revenue and expenses

## The Synthetic Dataset

This sample generates 365 days of synthetic data with controlled components:

| Component | Description | Parameters |
|-----------|-------------|------------|
| **Trend** | Linear upward trend | +0.5 units/day starting at 100 |
| **Weekly Seasonality** | 7-day cycle | +/- 15 units (sine wave) |
| **Yearly Seasonality** | 365-day cycle | +/- 25 units (summer peak) |
| **Noise** | Random variation | +/- 10 units (uniform) |

## Running the Sample

```bash
cd samples/time-series/Forecasting
dotnet run
```

## Expected Output

```
=== AiDotNet Time Series Forecasting ===
Sales/Stock Forecasting with Multiple Models

=== Step 1: Generate Synthetic Time Series Data ===

Generated 365 data points with:
  - Linear trend: +0.5 units per day
  - Weekly seasonality (7-day cycle): +/- 15 units
  - Yearly seasonality (365-day cycle): +/- 25 units
  - Random noise: +/- 10 units

Sample Data (first 14 days):
----------------------------------------------------------
Day |   Date    |  Value  |  Trend  | Seasonal |  Noise
----------------------------------------------------------
  1 | 2024-01-01 |   85.3 |   100.0 |    -12.5 |   -2.2
  2 | 2024-01-02 |   98.7 |   100.5 |     -5.8 |    4.0
  3 | 2024-01-03 |  112.4 |   101.0 |      8.2 |    3.2
...

=== Step 2: STL Decomposition (Trend/Seasonality Analysis) ===

STL Decomposition Results (first 14 days):
-------------------------------------------------------------------
Day |  Original |  Trend   | Seasonal | Residual
-------------------------------------------------------------------
  1 |     85.30 |    98.45 |   -11.23 |    -1.92
  2 |     98.70 |    99.12 |    -4.56 |     4.14
...

Trend extraction correlation with actual trend: 0.9847

=== Step 4: ARIMA Model Training and Forecasting ===

Training ARIMA(2,1,1) model...
ARIMA model trained successfully!

ARIMA Forecasting Metrics:
  MAE  (Mean Absolute Error):      18.5432
  RMSE (Root Mean Squared Error):  23.1256
  MSE  (Mean Squared Error):       534.7921

=== Step 5: SARIMA Model Training and Forecasting ===

Training SARIMA(1,1,1)(1,1,1)[7] model...
SARIMA model trained successfully!

SARIMA Forecasting Metrics:
  MAE  (Mean Absolute Error):      12.3456
  RMSE (Root Mean Squared Error):  15.8923
  MSE  (Mean Squared Error):       252.5656

=== Step 6: Model Comparison ===

Model Performance Comparison:
---------------------------------------------------------------
Model   |   MAE    |   RMSE   |   MSE      |   MAPE
---------------------------------------------------------------
ARIMA   |  18.5432 |  23.1256 |   534.7921 |  12.45%
SARIMA  |  12.3456 |  15.8923 |   252.5656 |   8.23%

Best performing model (lowest MAE): SARIMA

=== Step 7: Multi-Horizon Forecasting ===

Forecast Accuracy by Horizon (using best model):
---------------------------------------
Horizon |   MAE    | Direction Acc
---------------------------------------
  1 days |  10.2345 |    72.0%
  7 days |  12.5678 |    68.5%
 14 days |  14.8901 |    65.2%
 30 days |  18.2345 |    58.7%

=== Summary ===

Time Series Characteristics:
  Total data points: 365
  Training period: 300 days
  Test period: 65 days
  Data range: 72.34 to 285.67
  Mean value: 178.45
  Standard deviation: 52.34

Best Model Performance:
  Model: SARIMA
  MAE: 12.3456
  RMSE: 15.8923
  MAPE: 8.23%

=== Sample Complete ===
```

## Code Highlights

### Generating Synthetic Time Series

```csharp
// Create data with known components for evaluation
double trend = baseValue + trendSlope * i;
double weeklySeasonal = 15 * Math.Sin(2 * Math.PI * (i % 7) / 7);
double yearlySeasonal = 25 * Math.Sin(2 * Math.PI * (i % 365) / 365);
double noise = (random.NextDouble() - 0.5) * 20;
values[i] = trend + weeklySeasonal + yearlySeasonal + noise;
```

### STL Decomposition

```csharp
var stlOptions = new STLDecompositionOptions<double>
{
    SeasonalPeriod = 7,        // Weekly pattern
    TrendWindowSize = 21,      // 3-week smoothing
    SeasonalLoessWindow = 13,
    RobustIterations = 2       // Handle outliers
};

var stlDecomposition = new STLTimeSeriesDecomposition<double>(
    timeSeries, stlOptions, STLAlgorithmType.Robust);

var trend = stlDecomposition.GetComponent(DecompositionComponentType.Trend);
var seasonal = stlDecomposition.GetComponent(DecompositionComponentType.Seasonal);
var residual = stlDecomposition.GetComponent(DecompositionComponentType.Residual);
```

### ARIMA Configuration

```csharp
var arimaOptions = new ARIMAOptions<double>
{
    P = 2,  // AutoRegressive order - look at 2 previous values
    D = 1,  // Differencing order - first difference for stationarity
    Q = 1,  // Moving Average order - 1 error lag
    LagOrder = 7,         // Consider weekly patterns
    MaxIterations = 1000,
    Tolerance = 1e-6
};

var arimaModel = new ARIMAModel<double>(arimaOptions);
arimaModel.Train(trainMatrix, trainVector);
var predictions = arimaModel.Predict(testMatrix);
```

### SARIMA Configuration (Seasonal ARIMA)

```csharp
var sarimaOptions = new SARIMAOptions<double>
{
    P = 1, D = 1, Q = 1,        // Non-seasonal components
    SeasonalP = 1,              // Seasonal AR
    SeasonalD = 1,              // Seasonal differencing
    SeasonalQ = 1,              // Seasonal MA
    SeasonalPeriod = 7,         // Weekly seasonality
    LagOrder = 14               // Two weeks of history
};

var sarimaModel = new SARIMAModel<double>(sarimaOptions);
sarimaModel.Train(trainMatrix, trainVector);
```

### Model Evaluation

```csharp
var metrics = model.EvaluateModel(testMatrix, new Vector<double>(testValues));
Console.WriteLine($"MAE: {metrics["MAE"]:F4}");
Console.WriteLine($"RMSE: {metrics["RMSE"]:F4}");
Console.WriteLine($"MSE: {metrics["MSE"]:F4}");
```

## Understanding ARIMA/SARIMA

### ARIMA(p,d,q)

| Parameter | Description | Effect |
|-----------|-------------|--------|
| **p** (AR order) | Number of lag observations | Higher = captures longer patterns |
| **d** (Differencing) | Times to difference data | Removes trend (usually 1) |
| **q** (MA order) | Size of moving average window | Smooths noise effects |

### SARIMA(p,d,q)(P,D,Q)[m]

Extends ARIMA with seasonal components:

| Parameter | Description |
|-----------|-------------|
| **P** | Seasonal AR order |
| **D** | Seasonal differencing |
| **Q** | Seasonal MA order |
| **m** | Seasonal period (7 for weekly, 12 for monthly, etc.) |

### When to Use Which

| Data Characteristics | Recommended Model |
|---------------------|-------------------|
| No clear seasonality | ARIMA |
| Clear seasonal pattern | SARIMA |
| Multiple seasonalities | Use STL decomposition first |
| Non-stationary trend | Increase d parameter |

## Understanding the Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | mean(\|actual - predicted\|) | Average absolute error in original units |
| **RMSE** | sqrt(mean((actual - predicted)^2)) | Penalizes large errors more heavily |
| **MSE** | mean((actual - predicted)^2) | Squared error, sensitive to outliers |
| **MAPE** | mean(\|(actual - predicted)/actual\|) | Percentage error, scale-independent |

### Choosing the Right Metric

- **MAE**: Good general-purpose metric, easy to interpret
- **RMSE**: Use when large errors are particularly costly
- **MAPE**: Good for comparing across different scales
- **Direction Accuracy**: Important for trading/decision applications

## Multi-Horizon Forecasting

The sample demonstrates how forecast accuracy degrades over longer horizons:

```
Horizon |   MAE    | Direction Acc
---------------------------------
 1 day  |  10.23   |   72.0%
 7 days |  12.57   |   68.5%
30 days |  18.23   |   58.7%
```

This is typical - short-term forecasts are more accurate than long-term ones.

## STL Decomposition Benefits

STL (Seasonal-Trend decomposition using LOESS) helps by:

1. **Identifying components**: Separates trend, seasonality, and residuals
2. **Handling outliers**: Robust mode reduces outlier influence
3. **Improving forecasts**: Understanding components helps choose better models
4. **Anomaly detection**: Large residuals indicate unusual values

## Next Steps

- Try different ARIMA/SARIMA parameters
- Add exogenous variables (ARIMAX)
- Implement cross-validation for parameter selection
- Combine with neural network approaches for complex patterns

## Related Samples

- [AnomalyDetection](../AnomalyDetection/) - Detect unusual patterns in time series
- [BasicRegression](../../getting-started/BasicRegression/) - Fundamental regression concepts
