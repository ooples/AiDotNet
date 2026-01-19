using AiDotNet.DecompositionMethods.TimeSeriesDecomposition;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;

Console.WriteLine("=== AiDotNet Time Series Forecasting ===");
Console.WriteLine("Sales/Stock Forecasting with Multiple Models\n");

// =============================================================================
// Generate Synthetic Time Series Data with Trend, Seasonality, and Noise
// =============================================================================

Console.WriteLine("=== Step 1: Generate Synthetic Time Series Data ===\n");

int dataPoints = 365; // One year of daily data
var (timestamps, values, trendComponent, seasonalComponent, noiseComponent) =
    GenerateSyntheticTimeSeries(dataPoints, seed: 42);

Console.WriteLine($"Generated {dataPoints} data points with:");
Console.WriteLine("  - Linear trend: +0.5 units per day");
Console.WriteLine("  - Weekly seasonality (7-day cycle): +/- 15 units");
Console.WriteLine("  - Yearly seasonality (365-day cycle): +/- 25 units");
Console.WriteLine("  - Random noise: +/- 10 units\n");

// Show sample data
Console.WriteLine("Sample Data (first 14 days):");
Console.WriteLine("----------------------------------------------------------");
Console.WriteLine("Day |   Date    |  Value  |  Trend  | Seasonal |  Noise");
Console.WriteLine("----------------------------------------------------------");
for (int i = 0; i < 14; i++)
{
    Console.WriteLine($"{i + 1,3} | {timestamps[i]:yyyy-MM-dd} | {values[i],7:F1} | {trendComponent[i],7:F1} | {seasonalComponent[i],8:F1} | {noiseComponent[i],6:F1}");
}
Console.WriteLine("              ... (351 more days)\n");

// =============================================================================
// STL Decomposition - Trend and Seasonality Analysis
// =============================================================================

Console.WriteLine("=== Step 2: STL Decomposition (Trend/Seasonality Analysis) ===\n");

var timeSeries = new Vector<double>(values);
var stlOptions = new STLDecompositionOptions<double>
{
    SeasonalPeriod = 7, // Weekly seasonality
    TrendWindowSize = 21, // 3 weeks for trend smoothing
    SeasonalLoessWindow = 13,
    RobustIterations = 2
};

var stlDecomposition = new STLTimeSeriesDecomposition<double>(timeSeries, stlOptions, STLAlgorithmType.Robust);

var extractedTrend = stlDecomposition.GetComponent(DecompositionComponentType.Trend);
var extractedSeasonal = stlDecomposition.GetComponent(DecompositionComponentType.Seasonal);
var extractedResidual = stlDecomposition.GetComponent(DecompositionComponentType.Residual);

Console.WriteLine("STL Decomposition Results (first 14 days):");
Console.WriteLine("-------------------------------------------------------------------");
Console.WriteLine("Day |  Original |  Trend   | Seasonal | Residual");
Console.WriteLine("-------------------------------------------------------------------");
for (int i = 0; i < 14; i++)
{
    Console.WriteLine($"{i + 1,3} | {values[i],9:F2} | {extractedTrend[i],8:F2} | {extractedSeasonal[i],8:F2} | {extractedResidual[i],8:F2}");
}
Console.WriteLine();

// Calculate decomposition accuracy
double trendCorrelation = CalculateCorrelation(trendComponent.Take(extractedTrend.Length).ToArray(),
    Enumerable.Range(0, extractedTrend.Length).Select(i => extractedTrend[i]).ToArray());
Console.WriteLine($"Trend extraction correlation with actual trend: {trendCorrelation:F4}");
Console.WriteLine();

// =============================================================================
// Split Data into Training and Test Sets
// =============================================================================

Console.WriteLine("=== Step 3: Train/Test Split ===\n");

int trainSize = 300; // About 10 months for training
int testSize = dataPoints - trainSize; // About 2 months for testing

var trainValues = values.Take(trainSize).ToArray();
var testValues = values.Skip(trainSize).ToArray();
var testTimestamps = timestamps.Skip(trainSize).ToArray();

Console.WriteLine($"Training set: {trainSize} days (Day 1 - {trainSize})");
Console.WriteLine($"Test set: {testSize} days (Day {trainSize + 1} - {dataPoints})");
Console.WriteLine();

// =============================================================================
// Model 1: ARIMA Model
// =============================================================================

Console.WriteLine("=== Step 4: ARIMA Model Training and Forecasting ===\n");

var arimaOptions = new ARIMAOptions<double>
{
    P = 2, // AR order - look at 2 previous values
    D = 1, // Differencing order - first difference
    Q = 1, // MA order - 1 error lag
    LagOrder = 7, // Consider weekly patterns
    MaxIterations = 1000,
    Tolerance = 1e-6
};

var arimaModel = new ARIMAModel<double>(arimaOptions);

// Prepare training data
var trainMatrix = new Matrix<double>(trainSize, 1);
var trainVector = new Vector<double>(trainValues);
for (int i = 0; i < trainSize; i++)
{
    trainMatrix[i, 0] = i;
}

Console.WriteLine($"Training ARIMA({arimaOptions.P},{arimaOptions.D},{arimaOptions.Q}) model...");
arimaModel.Train(trainMatrix, trainVector);
Console.WriteLine("ARIMA model trained successfully!\n");

// Generate forecasts
var testMatrix = new Matrix<double>(testSize, 1);
for (int i = 0; i < testSize; i++)
{
    testMatrix[i, 0] = trainSize + i;
}

var arimaPredictions = arimaModel.Predict(testMatrix);
var arimaMetrics = arimaModel.EvaluateModel(testMatrix, new Vector<double>(testValues));

Console.WriteLine("ARIMA Forecasting Metrics:");
Console.WriteLine($"  MAE  (Mean Absolute Error):      {arimaMetrics["MAE"]:F4}");
Console.WriteLine($"  RMSE (Root Mean Squared Error):  {arimaMetrics["RMSE"]:F4}");
Console.WriteLine($"  MSE  (Mean Squared Error):       {arimaMetrics["MSE"]:F4}");
Console.WriteLine();

// =============================================================================
// Model 2: SARIMA Model (Seasonal ARIMA)
// =============================================================================

Console.WriteLine("=== Step 5: SARIMA Model Training and Forecasting ===\n");

var sarimaOptions = new SARIMAOptions<double>
{
    P = 1, // Non-seasonal AR
    D = 1, // Non-seasonal differencing
    Q = 1, // Non-seasonal MA
    SeasonalP = 1, // Seasonal AR
    SeasonalD = 1, // Seasonal differencing
    SeasonalQ = 1, // Seasonal MA
    SeasonalPeriod = 7, // Weekly seasonality
    LagOrder = 14, // Two weeks of history
    MaxIterations = 1000,
    Tolerance = 1e-6
};

var sarimaModel = new SARIMAModel<double>(sarimaOptions);

Console.WriteLine($"Training SARIMA({sarimaOptions.P},{sarimaOptions.D},{sarimaOptions.Q})({sarimaOptions.SeasonalP},{sarimaOptions.SeasonalD},{sarimaOptions.SeasonalQ})[{sarimaOptions.SeasonalPeriod}] model...");
sarimaModel.Train(trainMatrix, trainVector);
Console.WriteLine("SARIMA model trained successfully!\n");

var sarimaPredictions = sarimaModel.Predict(testMatrix);
var sarimaMetrics = sarimaModel.EvaluateModel(testMatrix, new Vector<double>(testValues));

Console.WriteLine("SARIMA Forecasting Metrics:");
Console.WriteLine($"  MAE  (Mean Absolute Error):      {sarimaMetrics["MAE"]:F4}");
Console.WriteLine($"  RMSE (Root Mean Squared Error):  {sarimaMetrics["RMSE"]:F4}");
Console.WriteLine($"  MSE  (Mean Squared Error):       {sarimaMetrics["MSE"]:F4}");
Console.WriteLine();

// =============================================================================
// Model Comparison
// =============================================================================

Console.WriteLine("=== Step 6: Model Comparison ===\n");

// Calculate MAPE manually for both models
double arimaMape = CalculateMAPE(testValues, Enumerable.Range(0, arimaPredictions.Length).Select(i => arimaPredictions[i]).ToArray());
double sarimaMape = CalculateMAPE(testValues, Enumerable.Range(0, sarimaPredictions.Length).Select(i => sarimaPredictions[i]).ToArray());

Console.WriteLine("Model Performance Comparison:");
Console.WriteLine("---------------------------------------------------------------");
Console.WriteLine("Model   |   MAE    |   RMSE   |   MSE      |   MAPE");
Console.WriteLine("---------------------------------------------------------------");
Console.WriteLine($"ARIMA   | {arimaMetrics["MAE"],8:F4} | {arimaMetrics["RMSE"],8:F4} | {arimaMetrics["MSE"],10:F4} | {arimaMape,6:F2}%");
Console.WriteLine($"SARIMA  | {sarimaMetrics["MAE"],8:F4} | {sarimaMetrics["RMSE"],8:F4} | {sarimaMetrics["MSE"],10:F4} | {sarimaMape,6:F2}%");
Console.WriteLine();

// Determine winner
string winner = arimaMetrics["MAE"] < sarimaMetrics["MAE"] ? "ARIMA" : "SARIMA";
Console.WriteLine($"Best performing model (lowest MAE): {winner}");
Console.WriteLine();

// =============================================================================
// Multi-Horizon Forecasting
// =============================================================================

Console.WriteLine("=== Step 7: Multi-Horizon Forecasting ===\n");

int[] horizons = { 1, 7, 14, 30 };
Console.WriteLine("Forecast Accuracy by Horizon (using best model):");
Console.WriteLine("---------------------------------------");
Console.WriteLine("Horizon |   MAE    | Direction Acc");
Console.WriteLine("---------------------------------------");

var bestModel = arimaMetrics["MAE"] < sarimaMetrics["MAE"] ? arimaModel : sarimaModel;
var bestPredictions = arimaMetrics["MAE"] < sarimaMetrics["MAE"] ? arimaPredictions : sarimaPredictions;

foreach (int horizon in horizons)
{
    if (horizon <= testSize)
    {
        // Calculate MAE for this horizon
        double horizonMae = 0;
        int correctDirection = 0;
        int horizonCount = Math.Min(horizon, testSize);

        for (int i = 0; i < horizonCount; i++)
        {
            horizonMae += Math.Abs(testValues[i] - bestPredictions[i]);

            // Check direction accuracy (is forecast moving in same direction as actual?)
            if (i > 0)
            {
                bool actualUp = testValues[i] > testValues[i - 1];
                bool forecastUp = bestPredictions[i] > bestPredictions[i - 1];
                if (actualUp == forecastUp) correctDirection++;
            }
        }
        horizonMae /= horizonCount;
        double directionAccuracy = horizonCount > 1 ? (double)correctDirection / (horizonCount - 1) * 100 : 100;

        Console.WriteLine($"{horizon,3} days | {horizonMae,8:F4} |   {directionAccuracy,5:F1}%");
    }
}
Console.WriteLine();

// =============================================================================
// Detailed Forecast Comparison
// =============================================================================

Console.WriteLine("=== Step 8: Detailed Forecast Results ===\n");

Console.WriteLine("Forecast vs Actual (first 14 test days):");
Console.WriteLine("-----------------------------------------------------------------------");
Console.WriteLine("Day |    Date    |  Actual  |  ARIMA   |  SARIMA  | Best Err");
Console.WriteLine("-----------------------------------------------------------------------");

for (int i = 0; i < Math.Min(14, testSize); i++)
{
    double arimaErr = Math.Abs(testValues[i] - arimaPredictions[i]);
    double sarimaErr = Math.Abs(testValues[i] - sarimaPredictions[i]);
    double bestErr = Math.Min(arimaErr, sarimaErr);

    Console.WriteLine($"{trainSize + i + 1,3} | {testTimestamps[i]:yyyy-MM-dd} | {testValues[i],8:F2} | {arimaPredictions[i],8:F2} | {sarimaPredictions[i],8:F2} | {bestErr,7:F2}");
}
Console.WriteLine("                   ... (more days)\n");

// =============================================================================
// Model Metadata
// =============================================================================

Console.WriteLine("=== Step 9: Model Metadata ===\n");

var arimaMetadata = arimaModel.GetModelMetadata();
Console.WriteLine("ARIMA Model Configuration:");
Console.WriteLine($"  Model Type: {arimaMetadata.ModelType}");
Console.WriteLine($"  P (AR order): {arimaMetadata.AdditionalInfo["P"]}");
Console.WriteLine($"  D (Differencing): {arimaMetadata.AdditionalInfo["D"]}");
Console.WriteLine($"  Q (MA order): {arimaMetadata.AdditionalInfo["Q"]}");
Console.WriteLine($"  AR Coefficients: {arimaMetadata.AdditionalInfo["ARCoefficientsCount"]}");
Console.WriteLine($"  MA Coefficients: {arimaMetadata.AdditionalInfo["MACoefficientsCount"]}");
Console.WriteLine();

var sarimaMetadata = sarimaModel.GetModelMetadata();
Console.WriteLine("SARIMA Model Configuration:");
Console.WriteLine($"  Model Type: {sarimaMetadata.ModelType}");
Console.WriteLine($"  P (AR order): {sarimaMetadata.AdditionalInfo["P"]}");
Console.WriteLine($"  D (Differencing): {sarimaMetadata.AdditionalInfo["D"]}");
Console.WriteLine($"  Q (MA order): {sarimaMetadata.AdditionalInfo["Q"]}");
Console.WriteLine($"  Seasonal P: {sarimaMetadata.AdditionalInfo["SeasonalP"]}");
Console.WriteLine($"  Seasonal D: {sarimaMetadata.AdditionalInfo["SeasonalD"]}");
Console.WriteLine($"  Seasonal Q: {sarimaMetadata.AdditionalInfo["SeasonalQ"]}");
Console.WriteLine($"  Seasonal Period: {sarimaMetadata.AdditionalInfo["SeasonalPeriod"]}");
Console.WriteLine();

// =============================================================================
// Summary Statistics
// =============================================================================

Console.WriteLine("=== Summary ===\n");

Console.WriteLine("Time Series Characteristics:");
Console.WriteLine($"  Total data points: {dataPoints}");
Console.WriteLine($"  Training period: {trainSize} days");
Console.WriteLine($"  Test period: {testSize} days");
Console.WriteLine($"  Data range: {values.Min():F2} to {values.Max():F2}");
Console.WriteLine($"  Mean value: {values.Average():F2}");
Console.WriteLine($"  Standard deviation: {CalculateStdDev(values):F2}");
Console.WriteLine();

Console.WriteLine("Best Model Performance:");
Console.WriteLine($"  Model: {winner}");
Console.WriteLine($"  MAE: {Math.Min(arimaMetrics["MAE"], sarimaMetrics["MAE"]):F4}");
Console.WriteLine($"  RMSE: {Math.Min(arimaMetrics["RMSE"], sarimaMetrics["RMSE"]):F4}");
Console.WriteLine($"  MAPE: {Math.Min(arimaMape, sarimaMape):F2}%");
Console.WriteLine();

Console.WriteLine("=== Sample Complete ===");

// =============================================================================
// Helper Functions
// =============================================================================

static (DateTime[] timestamps, double[] values, double[] trend, double[] seasonal, double[] noise)
    GenerateSyntheticTimeSeries(int n, int seed)
{
    var random = new Random(seed);
    var timestamps = new DateTime[n];
    var values = new double[n];
    var trend = new double[n];
    var seasonal = new double[n];
    var noise = new double[n];

    DateTime startDate = new DateTime(2024, 1, 1);
    double baseValue = 100.0;
    double trendSlope = 0.5; // Increasing trend

    for (int i = 0; i < n; i++)
    {
        timestamps[i] = startDate.AddDays(i);

        // Trend component (linear growth)
        trend[i] = baseValue + trendSlope * i;

        // Weekly seasonality (stronger on weekends)
        double dayOfWeek = i % 7;
        double weeklySeasonal = 15 * Math.Sin(2 * Math.PI * dayOfWeek / 7);

        // Yearly seasonality (summer peak, winter low)
        double dayOfYear = i % 365;
        double yearlySeasonal = 25 * Math.Sin(2 * Math.PI * (dayOfYear - 80) / 365); // Peak around day 170 (summer)

        seasonal[i] = weeklySeasonal + yearlySeasonal;

        // Random noise
        noise[i] = (random.NextDouble() - 0.5) * 20; // +/- 10 units

        // Combined value
        values[i] = trend[i] + seasonal[i] + noise[i];
    }

    return (timestamps, values, trend, seasonal, noise);
}

static double CalculateMAPE(double[] actual, double[] predicted)
{
    double sumPercentError = 0;
    int count = 0;

    for (int i = 0; i < actual.Length && i < predicted.Length; i++)
    {
        if (Math.Abs(actual[i]) > 0.001) // Avoid division by zero
        {
            sumPercentError += Math.Abs((actual[i] - predicted[i]) / actual[i]) * 100;
            count++;
        }
    }

    return count > 0 ? sumPercentError / count : 0;
}

static double CalculateCorrelation(double[] x, double[] y)
{
    int n = Math.Min(x.Length, y.Length);
    if (n == 0) return 0;

    double meanX = x.Take(n).Average();
    double meanY = y.Take(n).Average();

    double sumXY = 0, sumX2 = 0, sumY2 = 0;

    for (int i = 0; i < n; i++)
    {
        double dx = x[i] - meanX;
        double dy = y[i] - meanY;
        sumXY += dx * dy;
        sumX2 += dx * dx;
        sumY2 += dy * dy;
    }

    double denominator = Math.Sqrt(sumX2 * sumY2);
    return denominator > 0 ? sumXY / denominator : 0;
}

static double CalculateStdDev(double[] values)
{
    double mean = values.Average();
    double sumSquaredDiff = values.Sum(v => (v - mean) * (v - mean));
    return Math.Sqrt(sumSquaredDiff / values.Length);
}
