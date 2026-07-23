using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using AiDotNet.TimeSeries.AnomalyDetection;

Console.WriteLine("=== AiDotNet Time Series Anomaly Detection ===");
Console.WriteLine("Detecting anomalies using Isolation Forest and ARIMA-based methods\n");

// =============================================================================
// Generate Synthetic Time Series with Known Anomalies
// =============================================================================

Console.WriteLine("=== Step 1: Generate Synthetic Time Series with Anomalies ===\n");

int dataPoints = 500;
var (timestamps, values, anomalyIndices, anomalyTypes) = GenerateTimeSeriesWithAnomalies(dataPoints, seed: 42);

Console.WriteLine($"Generated {dataPoints} data points with {anomalyIndices.Count} injected anomalies:");
Console.WriteLine($"  - Point anomalies (spikes/drops): {anomalyTypes.Count(t => t == "spike" || t == "drop")}");
Console.WriteLine($"  - Level shifts: {anomalyTypes.Count(t => t == "level_shift")}");
Console.WriteLine($"  - Contextual anomalies: {anomalyTypes.Count(t => t == "contextual")}");
Console.WriteLine();

// Show sample data with anomaly markers
Console.WriteLine("Sample Data (points around anomalies marked with *):");
Console.WriteLine("-------------------------------------------------------------");
Console.WriteLine("Index |    Date    |   Value   | Anomaly Type");
Console.WriteLine("-------------------------------------------------------------");

int shownCount = 0;
for (int i = 0; i < Math.Min(dataPoints, 30); i++)
{
    int anomalyIdx = anomalyIndices.IndexOf(i);
    string marker = anomalyIdx >= 0 ? $"  * {anomalyTypes[anomalyIdx]}" : "";
    Console.WriteLine($"{i,5} | {timestamps[i]:yyyy-MM-dd} | {values[i],9:F2} |{marker}");
    shownCount++;
}
Console.WriteLine($"              ... ({dataPoints - shownCount} more points)\n");

// =============================================================================
// Method 1: Time Series Isolation Forest
// =============================================================================

Console.WriteLine("=== Step 2: Isolation Forest Anomaly Detection ===\n");

var isolationForestOptions = new TimeSeriesIsolationForestOptions<double>
{
    NumTrees = 100,              // Number of isolation trees
    SampleSize = 256,            // Samples per tree
    ContaminationRate = 0.05,    // Expected 5% anomalies
    LagFeatures = 10,            // Use 10 lag features
    RollingWindowSize = 20,      // 20-point rolling window
    UseTrendFeatures = true,     // Include derivative features
    UseSeasonalDecomposition = false,
    SeasonalPeriod = 7,          // Weekly pattern
    RandomSeed = 42
};

var isolationForest = new TimeSeriesIsolationForest<double>(isolationForestOptions);

// Prepare training data
var timeSeries = new Vector<double>(values);
var trainMatrix = new Matrix<double>(dataPoints, 1);
for (int i = 0; i < dataPoints; i++)
{
    trainMatrix[i, 0] = values[i];
}

Console.WriteLine("Training Isolation Forest...");
Console.WriteLine($"  NumTrees: {isolationForestOptions.NumTrees}");
Console.WriteLine($"  SampleSize: {isolationForestOptions.SampleSize}");
Console.WriteLine($"  ContaminationRate: {isolationForestOptions.ContaminationRate:P0}");
Console.WriteLine($"  LagFeatures: {isolationForestOptions.LagFeatures}");
Console.WriteLine($"  RollingWindowSize: {isolationForestOptions.RollingWindowSize}");

isolationForest.Train(trainMatrix, timeSeries);
Console.WriteLine("Isolation Forest trained successfully!\n");

// Detect anomalies
var ifScores = isolationForest.DetectAnomalies(timeSeries);
var ifLabels = isolationForest.GetAnomalyLabels(timeSeries);
var ifAnomalyIndices = isolationForest.GetAnomalyIndices(timeSeries);

Console.WriteLine($"Isolation Forest detected {ifAnomalyIndices.Count} anomalies");
Console.WriteLine();

// Show anomaly scores distribution
Console.WriteLine("Anomaly Score Distribution:");
Console.WriteLine("---------------------------");
double[] scoreRanges = { 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
for (int i = 0; i < scoreRanges.Length; i++)
{
    double lower = i == 0 ? 0 : scoreRanges[i - 1];
    double upper = scoreRanges[i];
    int count = 0;
    for (int j = 0; j < ifScores.Length; j++)
    {
        if (ifScores[j] > lower && ifScores[j] <= upper)
            count++;
    }
    string bar = new string('#', Math.Min(count / 5, 40));
    Console.WriteLine($"  {lower:F1}-{upper:F1}: {count,4} |{bar}");
}
Console.WriteLine();

// =============================================================================
// Method 2: ARIMA-based Anomaly Detection
// =============================================================================

Console.WriteLine("=== Step 3: ARIMA-based Anomaly Detection ===\n");

var arimaOptions = new ARIMAOptions<double>
{
    P = 2,
    D = 1,
    Q = 1,
    LagOrder = 7,
    EnableAnomalyDetection = true,   // Enable anomaly detection
    AnomalyThresholdSigma = 2.5,     // 2.5 standard deviations
    MaxIterations = 1000,
    Tolerance = 1e-6
};

var arimaModel = new ARIMAModel<double>(arimaOptions);

Console.WriteLine("Training ARIMA model with anomaly detection...");
Console.WriteLine($"  P (AR order): {arimaOptions.P}");
Console.WriteLine($"  D (Differencing): {arimaOptions.D}");
Console.WriteLine($"  Q (MA order): {arimaOptions.Q}");
Console.WriteLine($"  Anomaly Threshold: {arimaOptions.AnomalyThresholdSigma} sigma");

arimaModel.Train(trainMatrix, timeSeries);
Console.WriteLine("ARIMA model trained successfully!\n");

// Detect anomalies using ARIMA
var arimaAnomalies = arimaModel.DetectAnomalies(timeSeries);
var arimaScores = arimaModel.ComputeAnomalyScores(timeSeries);
var arimaDetailedAnomalies = arimaModel.DetectAnomaliesDetailed(timeSeries);

int arimaAnomalyCount = arimaAnomalies.Count(a => a);
Console.WriteLine($"ARIMA detected {arimaAnomalyCount} anomalies");
Console.WriteLine($"Anomaly threshold: {arimaModel.GetAnomalyThreshold():F4}");
Console.WriteLine();

// =============================================================================
// Compare Detection Methods
// =============================================================================

Console.WriteLine("=== Step 4: Detection Method Comparison ===\n");

// Calculate detection metrics
var ifDetectedSet = new HashSet<int>(ifAnomalyIndices);
var arimaDetectedSet = new HashSet<int>();
for (int i = 0; i < arimaAnomalies.Length; i++)
{
    if (arimaAnomalies[i]) arimaDetectedSet.Add(i);
}
var actualAnomalySet = new HashSet<int>(anomalyIndices);

// Calculate TP, FP, FN for each method
var ifMetrics = CalculateDetectionMetrics(actualAnomalySet, ifDetectedSet, dataPoints);
var arimaMetrics = CalculateDetectionMetrics(actualAnomalySet, arimaDetectedSet, dataPoints);

Console.WriteLine("Detection Performance Comparison:");
Console.WriteLine("------------------------------------------------------------------");
Console.WriteLine("Method          | Detected | TP  | FP  | FN  | Precision | Recall | F1");
Console.WriteLine("------------------------------------------------------------------");
Console.WriteLine($"Isolation Forest | {ifAnomalyIndices.Count,7}  | {ifMetrics.tp,3} | {ifMetrics.fp,3} | {ifMetrics.fn,3} |   {ifMetrics.precision,6:F3}  | {ifMetrics.recall,6:F3} | {ifMetrics.f1,6:F3}");
Console.WriteLine($"ARIMA            | {arimaAnomalyCount,7}  | {arimaMetrics.tp,3} | {arimaMetrics.fp,3} | {arimaMetrics.fn,3} |   {arimaMetrics.precision,6:F3}  | {arimaMetrics.recall,6:F3} | {arimaMetrics.f1,6:F3}");
Console.WriteLine();

// Ensemble: detected by both methods
var ensembleDetected = new HashSet<int>(ifDetectedSet.Intersect(arimaDetectedSet));
var ensembleMetrics = CalculateDetectionMetrics(actualAnomalySet, ensembleDetected, dataPoints);

// Union: detected by either method
var unionDetected = new HashSet<int>(ifDetectedSet.Union(arimaDetectedSet));
var unionMetrics = CalculateDetectionMetrics(actualAnomalySet, unionDetected, dataPoints);

Console.WriteLine($"Ensemble (both)  | {ensembleDetected.Count,7}  | {ensembleMetrics.tp,3} | {ensembleMetrics.fp,3} | {ensembleMetrics.fn,3} |   {ensembleMetrics.precision,6:F3}  | {ensembleMetrics.recall,6:F3} | {ensembleMetrics.f1,6:F3}");
Console.WriteLine($"Union (either)   | {unionDetected.Count,7}  | {unionMetrics.tp,3} | {unionMetrics.fp,3} | {unionMetrics.fn,3} |   {unionMetrics.precision,6:F3}  | {unionMetrics.recall,6:F3} | {unionMetrics.f1,6:F3}");
Console.WriteLine();

// =============================================================================
// Detailed Anomaly Analysis
// =============================================================================

Console.WriteLine("=== Step 5: Detailed Anomaly Analysis ===\n");

Console.WriteLine("Detected Anomalies (Top 15 by Isolation Forest score):");
Console.WriteLine("-------------------------------------------------------------------------");
Console.WriteLine("Index |    Date    |  Value   | IF Score | ARIMA Score | Actual Type");
Console.WriteLine("-------------------------------------------------------------------------");

// Sort by IF score and show top anomalies
var sortedByScore = ifAnomalyIndices
    .Select(idx => new
    {
        Index = idx,
        Value = values[idx],
        IFScore = ifScores[idx],
        ARIMAScore = idx < arimaScores.Length ? arimaScores[idx] : 0.0,
        IsActual = actualAnomalySet.Contains(idx),
        ActualType = anomalyIndices.Contains(idx)
            ? anomalyTypes[anomalyIndices.IndexOf(idx)]
            : "-"
    })
    .OrderByDescending(x => x.IFScore)
    .Take(15);

foreach (var anomaly in sortedByScore)
{
    string statusMarker = anomaly.IsActual ? "TRUE" : "FP";
    Console.WriteLine($"{anomaly.Index,5} | {timestamps[anomaly.Index]:yyyy-MM-dd} | {anomaly.Value,8:F2} | {anomaly.IFScore,8:F4} | {anomaly.ARIMAScore,11:F4} | {anomaly.ActualType,-12} ({statusMarker})");
}
Console.WriteLine();

// =============================================================================
// Anomaly Visualization (ASCII)
// =============================================================================

Console.WriteLine("=== Step 6: Anomaly Visualization ===\n");

// Show ASCII visualization of a portion of the time series
int vizStart = 50;
int vizEnd = 150;
double minVal = values.Skip(vizStart).Take(vizEnd - vizStart).Min();
double maxVal = values.Skip(vizStart).Take(vizEnd - vizStart).Max();
int chartHeight = 15;
int chartWidth = vizEnd - vizStart;

Console.WriteLine($"Time Series Visualization (points {vizStart}-{vizEnd}):");
Console.WriteLine($"Value range: {minVal:F1} to {maxVal:F1}");
Console.WriteLine();

// Create ASCII chart
for (int row = chartHeight - 1; row >= 0; row--)
{
    double rowValue = minVal + (maxVal - minVal) * row / (chartHeight - 1);
    Console.Write($"{rowValue,7:F1} |");

    for (int col = 0; col < chartWidth; col++)
    {
        int idx = vizStart + col;
        double normalizedValue = (values[idx] - minVal) / (maxVal - minVal) * (chartHeight - 1);
        int valueRow = (int)Math.Round(normalizedValue);

        if (valueRow == row)
        {
            if (ifDetectedSet.Contains(idx) || arimaDetectedSet.Contains(idx))
                Console.Write("*"); // Detected anomaly
            else if (actualAnomalySet.Contains(idx))
                Console.Write("!"); // Missed anomaly
            else
                Console.Write("."); // Normal point
        }
        else
        {
            Console.Write(" ");
        }
    }
    Console.WriteLine();
}

Console.Write("        +");
Console.WriteLine(new string('-', chartWidth));
Console.Write("         ");
for (int i = 0; i < chartWidth; i += 20)
{
    Console.Write($"{vizStart + i,-20}");
}
Console.WriteLine();
Console.WriteLine();
Console.WriteLine("Legend: . = normal, * = detected anomaly, ! = missed anomaly");
Console.WriteLine();

// =============================================================================
// Threshold Analysis
// =============================================================================

Console.WriteLine("=== Step 7: Threshold Sensitivity Analysis ===\n");

double[] thresholds = { 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85 };

Console.WriteLine("Isolation Forest - Threshold vs Detection Performance:");
Console.WriteLine("----------------------------------------------------------");
Console.WriteLine("Threshold | Detected | TP  | FP  | Precision | Recall | F1");
Console.WriteLine("----------------------------------------------------------");

foreach (double threshold in thresholds)
{
    var thresholdDetected = new HashSet<int>();
    for (int i = 0; i < ifScores.Length; i++)
    {
        if (ifScores[i] > threshold)
            thresholdDetected.Add(i);
    }

    var metrics = CalculateDetectionMetrics(actualAnomalySet, thresholdDetected, dataPoints);
    Console.WriteLine($"   {threshold:F2}    |   {thresholdDetected.Count,4}   | {metrics.tp,3} | {metrics.fp,3} |   {metrics.precision,6:F3}  | {metrics.recall,6:F3} | {metrics.f1,6:F3}");
}
Console.WriteLine();

// =============================================================================
// Anomaly Score Statistics
// =============================================================================

Console.WriteLine("=== Step 8: Anomaly Score Statistics ===\n");

// Calculate statistics for normal vs anomaly points
var normalScores = new List<double>();
var anomalyScoresActual = new List<double>();

for (int i = 0; i < ifScores.Length; i++)
{
    if (actualAnomalySet.Contains(i))
        anomalyScoresActual.Add(ifScores[i]);
    else
        normalScores.Add(ifScores[i]);
}

Console.WriteLine("Isolation Forest Score Statistics:");
Console.WriteLine($"  Normal points ({normalScores.Count}):");
Console.WriteLine($"    Mean:   {normalScores.Average():F4}");
Console.WriteLine($"    StdDev: {CalculateStdDev(normalScores.ToArray()):F4}");
Console.WriteLine($"    Min:    {normalScores.Min():F4}");
Console.WriteLine($"    Max:    {normalScores.Max():F4}");
Console.WriteLine();
Console.WriteLine($"  Actual anomaly points ({anomalyScoresActual.Count}):");
if (anomalyScoresActual.Count > 0)
{
    Console.WriteLine($"    Mean:   {anomalyScoresActual.Average():F4}");
    Console.WriteLine($"    StdDev: {CalculateStdDev(anomalyScoresActual.ToArray()):F4}");
    Console.WriteLine($"    Min:    {anomalyScoresActual.Min():F4}");
    Console.WriteLine($"    Max:    {anomalyScoresActual.Max():F4}");
}
Console.WriteLine();

// Score separation
if (anomalyScoresActual.Count > 0 && normalScores.Count > 0)
{
    double separation = (anomalyScoresActual.Average() - normalScores.Average()) /
                       Math.Sqrt((CalculateVariance(anomalyScoresActual.ToArray()) +
                                 CalculateVariance(normalScores.ToArray())) / 2);
    Console.WriteLine($"Score separation (Cohen's d): {separation:F4}");
    Console.WriteLine(separation > 0.8 ? "  (Good separation between normal and anomaly scores)" :
                     separation > 0.5 ? "  (Moderate separation)" : "  (Poor separation)");
}
Console.WriteLine();

// =============================================================================
// Summary
// =============================================================================

Console.WriteLine("=== Summary ===\n");

Console.WriteLine("Dataset Characteristics:");
Console.WriteLine($"  Total data points: {dataPoints}");
Console.WriteLine($"  Actual anomalies: {anomalyIndices.Count} ({anomalyIndices.Count * 100.0 / dataPoints:F1}%)");
Console.WriteLine($"  Data range: {values.Min():F2} to {values.Max():F2}");
Console.WriteLine();

string bestMethod = ifMetrics.f1 > arimaMetrics.f1 ? "Isolation Forest" : "ARIMA";
var bestMetrics = ifMetrics.f1 > arimaMetrics.f1 ? ifMetrics : arimaMetrics;

Console.WriteLine("Best Detection Method:");
Console.WriteLine($"  Method: {bestMethod}");
Console.WriteLine($"  F1-Score: {bestMetrics.f1:F4}");
Console.WriteLine($"  Precision: {bestMetrics.precision:F4}");
Console.WriteLine($"  Recall: {bestMetrics.recall:F4}");
Console.WriteLine();

Console.WriteLine("Recommendations:");
if (bestMetrics.precision < 0.7)
    Console.WriteLine("  - Consider increasing the anomaly threshold to reduce false positives");
if (bestMetrics.recall < 0.7)
    Console.WriteLine("  - Consider lowering the anomaly threshold to catch more anomalies");
if (ensembleMetrics.precision > bestMetrics.precision)
    Console.WriteLine("  - Using ensemble (both methods agree) improves precision");
if (unionMetrics.recall > bestMetrics.recall)
    Console.WriteLine("  - Using union (either method detects) improves recall");
Console.WriteLine();

Console.WriteLine("=== Sample Complete ===");

// =============================================================================
// Helper Functions
// =============================================================================

static (DateTime[] timestamps, double[] values, List<int> anomalyIndices, List<string> anomalyTypes)
    GenerateTimeSeriesWithAnomalies(int n, int seed)
{
    var random = new Random(seed);
    var timestamps = new DateTime[n];
    var values = new double[n];
    var anomalyIndices = new List<int>();
    var anomalyTypes = new List<string>();

    DateTime startDate = new DateTime(2024, 1, 1);
    double baseValue = 100.0;

    // Generate base time series with trend and seasonality
    for (int i = 0; i < n; i++)
    {
        timestamps[i] = startDate.AddDays(i);

        // Linear trend
        double trend = baseValue + 0.1 * i;

        // Weekly seasonality
        double weekly = 10 * Math.Sin(2 * Math.PI * (i % 7) / 7);

        // Monthly seasonality (30-day cycle)
        double monthly = 15 * Math.Sin(2 * Math.PI * (i % 30) / 30);

        // Random noise
        double noise = (random.NextDouble() - 0.5) * 8;

        values[i] = trend + weekly + monthly + noise;
    }

    // Inject anomalies of different types
    int numAnomalies = n / 20; // About 5% anomalies

    for (int a = 0; a < numAnomalies; a++)
    {
        int idx = random.Next(30, n - 10); // Avoid edges

        // Skip if already an anomaly nearby
        if (anomalyIndices.Any(ai => Math.Abs(ai - idx) < 5))
            continue;

        int anomalyType = random.Next(4);

        switch (anomalyType)
        {
            case 0: // Spike (sudden increase)
                values[idx] += 40 + random.NextDouble() * 30;
                anomalyIndices.Add(idx);
                anomalyTypes.Add("spike");
                break;

            case 1: // Drop (sudden decrease)
                values[idx] -= 40 + random.NextDouble() * 30;
                anomalyIndices.Add(idx);
                anomalyTypes.Add("drop");
                break;

            case 2: // Level shift (sustained change)
                double shift = (random.NextDouble() > 0.5 ? 1 : -1) * (25 + random.NextDouble() * 15);
                for (int j = idx; j < Math.Min(idx + 5, n); j++)
                {
                    values[j] += shift;
                }
                anomalyIndices.Add(idx);
                anomalyTypes.Add("level_shift");
                break;

            case 3: // Contextual anomaly (normal value but wrong context)
                // Make a weekday value look like a weekend value or vice versa
                double contextShift = 25 * (random.NextDouble() > 0.5 ? 1 : -1);
                values[idx] += contextShift;
                anomalyIndices.Add(idx);
                anomalyTypes.Add("contextual");
                break;
        }
    }

    return (timestamps, values, anomalyIndices, anomalyTypes);
}

static (int tp, int fp, int fn, double precision, double recall, double f1)
    CalculateDetectionMetrics(HashSet<int> actual, HashSet<int> detected, int totalPoints)
{
    int tp = actual.Intersect(detected).Count();
    int fp = detected.Except(actual).Count();
    int fn = actual.Except(detected).Count();

    double precision = detected.Count > 0 ? (double)tp / detected.Count : 0;
    double recall = actual.Count > 0 ? (double)tp / actual.Count : 0;
    double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

    return (tp, fp, fn, precision, recall, f1);
}

static double CalculateStdDev(double[] values)
{
    if (values.Length == 0) return 0;
    double mean = values.Average();
    double sumSquaredDiff = values.Sum(v => (v - mean) * (v - mean));
    return Math.Sqrt(sumSquaredDiff / values.Length);
}

static double CalculateVariance(double[] values)
{
    if (values.Length == 0) return 0;
    double mean = values.Average();
    return values.Sum(v => (v - mean) * (v - mean)) / values.Length;
}
