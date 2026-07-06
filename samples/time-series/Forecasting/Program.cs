// AiDotNet — Time Series Forecasting
//
// One-step-ahead forecasting through the AiModelBuilder facade. The series is
// turned into supervised (lagged-window -> next-value) examples and a gradient
// boosting regressor is trained via ConfigureModel + ConfigureDataLoader +
// BuildAsync; forecasts are read back through result.Predict().

using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;          // GradientBoostingRegressionOptions, ARIMAOptions
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;              // ARIMAModel

Console.WriteLine("=== AiDotNet Time Series Forecasting ===");
Console.WriteLine("Daily sales forecasting with trend + weekly seasonality\n");

// ── 1. Synthetic series: trend + weekly seasonality + noise ────────────────
const int dataPoints = 365;
const int window = 14;          // use the last 14 days to predict the next
var rng = new Random(42);
var values = new double[dataPoints];
for (int t = 0; t < dataPoints; t++)
{
    double trend = 0.5 * t;
    double weekly = 15 * Math.Sin(2 * Math.PI * t / 7.0);
    double yearly = 25 * Math.Sin(2 * Math.PI * t / 365.0);
    double noise = (rng.NextDouble() - 0.5) * 20;
    values[t] = 100 + trend + weekly + yearly + noise;
}
Console.WriteLine($"Generated {dataPoints} daily points; forecasting from a {window}-day window.\n");

// ── 2. Build supervised windows: X = last `window` values, y = next value ──
var features = new List<double[]>();
var targetChanges = new List<double>();   // predict the day-over-day change (stationary)
var actualNext = new List<double>();
for (int i = 0; i + window < dataPoints; i++)
{
    features.Add(values[i..(i + window)]);
    targetChanges.Add(values[i + window] - values[i + window - 1]);
    actualNext.Add(values[i + window]);
}
int trainCount = (int)(features.Count * 0.85);
var trainX = features.Take(trainCount).ToArray();
var trainY = targetChanges.Take(trainCount).ToArray();
var testX = features.Skip(trainCount).ToArray();
var testActual = actualNext.Skip(trainCount).ToArray();
Console.WriteLine($"Training windows: {trainX.Length}, Test windows: {testX.Length}\n");

// ── 3. Train the forecaster through the facade ─────────────────────────────
Console.WriteLine("Training through AiModelBuilder.ConfigureModel ...");
try
{
    var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
        .ConfigureModel(new GradientBoostingRegression<double>(
            new GradientBoostingRegressionOptions { NumberOfTrees = 100, MaxDepth = 4, LearningRate = 0.1 }))
        .ConfigureDataLoader(DataLoaders.FromArrays(trainX, trainY))
        .BuildAsync();

    Console.WriteLine("  Training complete.\n");

    // ── 4. Evaluate one-step-ahead forecasts on the held-out tail ──────────
    var predChanges = result.Predict(ToMatrix(testX));
    var forecasts = new double[testX.Length];
    double mae = 0, sumSq = 0, mean = testActual.Average(), totSq = 0;
    for (int i = 0; i < testX.Length; i++)
    {
        forecasts[i] = testX[i][^1] + predChanges[i];   // reconstruct the level from the last observed value
        double err = forecasts[i] - testActual[i];
        mae += Math.Abs(err);
        sumSq += err * err;
        totSq += Math.Pow(testActual[i] - mean, 2);
    }
    mae /= testX.Length;
    double rmse = Math.Sqrt(sumSq / testX.Length);
    double r2 = 1 - (sumSq / totSq);

    Console.WriteLine("Forecast Accuracy (one-step-ahead):");
    Console.WriteLine("------------------------------------");
    Console.WriteLine($"  R2:   {r2:F4}");
    Console.WriteLine($"  MAE:  {mae:F2}");
    Console.WriteLine($"  RMSE: {rmse:F2}\n");

    Console.WriteLine("Forecast vs Actual (first 7 test days):");
    for (int i = 0; i < Math.Min(7, testX.Length); i++)
        Console.WriteLine($"  Day {trainCount + window + i + 1,3}: forecast={forecasts[i],7:F1}  actual={testActual[i],7:F1}");
}
catch (Exception ex)
{
    // Surface failures so the samples CI catches broken forecasting/facade wiring.
    Console.Error.WriteLine($"  Forecasting sample failed: {ex.Message}");
    throw;
}

// ── 5. The same goal, the unified way: an ARIMA model forecasts through Predict ─
// A time-series model forecasts straight through the facade's result.Predict: ask for N
// rows and you get an N-step-ahead forecast extending the series it was trained on — no
// separate Forecast method, one Predict front for every model.
Console.WriteLine("\nARIMA forecast through the unified result.Predict:");
try
{
    var seriesX = new Matrix<double>(dataPoints, 1);   // placeholder; ARIMA forecasts from the series itself
    for (int t = 0; t < dataPoints; t++) seriesX[t, 0] = t;

    var arimaResult = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
        .ConfigureModel(new ARIMAModel<double>(new ARIMAOptions<double> { P = 2, D = 1, Q = 1 }))
        .ConfigureDataLoader(DataLoaders.FromMatrixVector(seriesX, new Vector<double>(values)))
        .BuildAsync();

    // result.Predict(matrix with N rows) -> N-step-ahead forecast.
    var forecast = arimaResult.Predict(new Matrix<double>(7, 1));
    Console.Write("  Next 7 days:");
    for (int i = 0; i < forecast.Length; i++) Console.Write($" {forecast[i]:F0}");
    Console.WriteLine();
}
catch (Exception ex)
{
    Console.WriteLine($"  ARIMA forecasting reported: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");

// Pack a jagged feature array into the dense Matrix the model's Predict expects.
static Matrix<double> ToMatrix(double[][] rows)
{
    int r = rows.Length, c = rows[0].Length;
    var m = new Matrix<double>(r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            m[i, j] = rows[i][j];
    return m;
}
