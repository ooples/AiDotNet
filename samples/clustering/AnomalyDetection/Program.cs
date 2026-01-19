using AiDotNet.Clustering.Density;
using AiDotNet.Clustering.Evaluation;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Anomaly Detection ===");
Console.WriteLine("Detecting fraudulent transactions using DBSCAN clustering\n");

// Generate realistic transaction data with known anomalies
var (transactions, trueLabels, transactionIds) = GenerateTransactionData(1000, anomalyRate: 0.05);

Console.WriteLine($"Generated {transactions.Rows} transactions");
Console.WriteLine($"True anomaly rate: {trueLabels.Count(l => l) * 100.0 / trueLabels.Length:F1}%\n");

// Display sample data
Console.WriteLine("Sample Transaction Data:");
Console.WriteLine("----------------------------------------------------------------");
Console.WriteLine("Transaction ID | Amount ($) | Hour | Day | Distance (mi) | Anomaly");
Console.WriteLine("----------------------------------------------------------------");
for (int i = 0; i < 10; i++)
{
    string anomalyFlag = trueLabels[i] ? "  *" : "";
    Console.WriteLine($"   {transactionIds[i]}   | {transactions[i, 0],9:F2} | {transactions[i, 1],4:F0} | {transactions[i, 2],3:F0} |    {transactions[i, 3],6:F1}     |{anomalyFlag}");
}
Console.WriteLine("              ...    (990 more transactions)");
Console.WriteLine("\n* = Known anomaly in ground truth\n");

// Normalize features for better clustering
var normalizedData = NormalizeFeatures(transactions);

// =====================================================
// Method 1: DBSCAN for Density-Based Anomaly Detection
// =====================================================
Console.WriteLine("=== Method 1: DBSCAN Anomaly Detection ===\n");

// Test different epsilon values to find optimal parameters
Console.WriteLine("Testing DBSCAN parameters...\n");
var dbscanResults = new List<(double eps, int minPts, int clusters, int noise, double precision, double recall, double f1)>();

double[] epsilonValues = { 0.3, 0.5, 0.7, 1.0, 1.5 };
int[] minPointsValues = { 3, 5, 10 };

foreach (double eps in epsilonValues)
{
    foreach (int minPts in minPointsValues)
    {
        var dbscan = new DBSCAN<double>(new DBSCANOptions<double>
        {
            Epsilon = eps,
            MinPoints = minPts,
            Algorithm = NeighborAlgorithm.Auto
        });

        dbscan.Train(normalizedData);
        var labels = dbscan.Labels!;

        // Points labeled as -1 are noise (potential anomalies)
        var predictedAnomalies = new bool[labels.Length];
        for (int i = 0; i < labels.Length; i++)
        {
            predictedAnomalies[i] = (int)labels[i] == DBSCAN<double>.NoiseLabel;
        }

        var (precision, recall, f1) = CalculateMetrics(trueLabels, predictedAnomalies);
        int numClusters = dbscan.NumClusters;
        int noiseCount = dbscan.GetNoiseCount();

        dbscanResults.Add((eps, minPts, numClusters, noiseCount, precision, recall, f1));
    }
}

// Display results table
Console.WriteLine("Epsilon | MinPts | Clusters | Noise | Precision | Recall | F1-Score");
Console.WriteLine("--------|--------|----------|-------|-----------|--------|----------");
foreach (var r in dbscanResults.OrderByDescending(x => x.f1))
{
    Console.WriteLine($"  {r.eps,4:F1}  |   {r.minPts,3}  |    {r.clusters,3}   |  {r.noise,4} |   {r.precision,6:F3}  | {r.recall,6:F3} |  {r.f1,6:F3}");
}

// Use the best parameters
var bestDbscan = dbscanResults.OrderByDescending(x => x.f1).First();
Console.WriteLine($"\nBest DBSCAN parameters: Epsilon={bestDbscan.eps}, MinPoints={bestDbscan.minPts}");
Console.WriteLine($"Best F1-Score: {bestDbscan.f1:F4}");

// Train final DBSCAN model
var finalDbscan = new DBSCAN<double>(new DBSCANOptions<double>
{
    Epsilon = bestDbscan.eps,
    MinPoints = bestDbscan.minPts,
    Algorithm = NeighborAlgorithm.Auto
});

finalDbscan.Train(normalizedData);
var dbscanLabels = finalDbscan.Labels!;

// =====================================================
// Contamination Rate Analysis
// =====================================================
Console.WriteLine("\n=== Contamination Rate Analysis ===\n");

// Analyze how detection rate changes with different thresholds
var contamRates = new double[] { 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15 };
Console.WriteLine("Contam. Rate | Flagged | True Pos | False Pos | Precision | Recall | F1-Score");
Console.WriteLine("-------------|---------|----------|-----------|-----------|--------|----------");

foreach (double contamRate in contamRates)
{
    // Using DBSCAN noise as anomalies, we can simulate different thresholds
    // by looking at core point distances
    int numToFlag = (int)(transactions.Rows * contamRate);

    // Calculate distance to nearest core point for each sample
    var distancesToCore = CalculateDistancesToCorePoints(normalizedData, finalDbscan);
    var sortedIndices = new HashSet<int>(distancesToCore
        .Select((dist, idx) => (dist, idx))
        .OrderByDescending(x => x.dist)
        .Take(numToFlag)
        .Select(x => x.idx));

    var predictedAnomalies = new bool[transactions.Rows];
    foreach (int idx in sortedIndices)
    {
        predictedAnomalies[idx] = true;
    }

    var (precision, recall, f1) = CalculateMetrics(trueLabels, predictedAnomalies);
    int truePos = trueLabels.Zip(predictedAnomalies, (t, p) => t && p).Count(x => x);
    int falsePos = trueLabels.Zip(predictedAnomalies, (t, p) => !t && p).Count(x => x);

    Console.WriteLine($"    {contamRate,5:F2}    |   {numToFlag,4}  |    {truePos,4}   |     {falsePos,4}  |   {precision,6:F3}  | {recall,6:F3} |  {f1,6:F3}");
}

// =====================================================
// Detailed Analysis of Detected Anomalies
// =====================================================
Console.WriteLine("\n=== Detected Anomalies Analysis ===\n");

var detectedAnomalies = new List<int>();
for (int i = 0; i < dbscanLabels.Length; i++)
{
    if ((int)dbscanLabels[i] == DBSCAN<double>.NoiseLabel)
    {
        detectedAnomalies.Add(i);
    }
}

Console.WriteLine($"Total transactions flagged as anomalies: {detectedAnomalies.Count}");

// Categorize the detected anomalies
int truePositives = 0;
int falsePositives = 0;

foreach (int idx in detectedAnomalies)
{
    if (trueLabels[idx])
        truePositives++;
    else
        falsePositives++;
}

Console.WriteLine($"  - True Positives (correctly identified fraud): {truePositives}");
Console.WriteLine($"  - False Positives (normal flagged as fraud): {falsePositives}");

// Show some example anomalies
Console.WriteLine("\nSample Detected Anomalies:");
Console.WriteLine("----------------------------------------------------------------");
Console.WriteLine("Transaction ID | Amount ($) | Hour | Day | Distance | Actual");
Console.WriteLine("----------------------------------------------------------------");

int shown = 0;
foreach (int idx in detectedAnomalies.Take(10))
{
    string actual = trueLabels[idx] ? "FRAUD" : "Normal";
    Console.WriteLine($"   {transactionIds[idx]}   | {transactions[idx, 0],9:F2} | {transactions[idx, 1],4:F0} | {transactions[idx, 2],3:F0} |   {transactions[idx, 3],6:F1}  | {actual}");
    shown++;
}

if (detectedAnomalies.Count > 10)
{
    Console.WriteLine($"              ...    ({detectedAnomalies.Count - 10} more anomalies)");
}

// =====================================================
// Cluster Quality Metrics
// =====================================================
Console.WriteLine("\n=== Clustering Quality Metrics ===\n");

var evaluator = new ClusteringEvaluator<double>();

// Filter out noise points for silhouette calculation (DBSCAN returns -1 for noise)
var validIndices = new List<int>();
for (int i = 0; i < dbscanLabels.Length; i++)
{
    if ((int)dbscanLabels[i] >= 0)
        validIndices.Add(i);
}

if (validIndices.Count > 0 && finalDbscan.NumClusters >= 2)
{
    var validData = ExtractRows(normalizedData, validIndices);
    var validLabels = new Vector<double>(validIndices.Count);
    for (int i = 0; i < validIndices.Count; i++)
    {
        validLabels[i] = dbscanLabels[validIndices[i]];
    }

    var silhouetteCalculator = new SilhouetteScore<double>();
    double silhouette = silhouetteCalculator.Compute(validData, validLabels);

    Console.WriteLine($"Number of Clusters (excluding noise): {finalDbscan.NumClusters}");
    Console.WriteLine($"Core Points: {finalDbscan.GetCoreSampleIndices().Length}");
    Console.WriteLine($"Noise Points (potential anomalies): {finalDbscan.GetNoiseCount()}");
    Console.WriteLine($"Silhouette Score (clustered points only): {silhouette:F4}");
}
else
{
    Console.WriteLine($"Number of Clusters: {finalDbscan.NumClusters}");
    Console.WriteLine($"Core Points: {finalDbscan.GetCoreSampleIndices().Length}");
    Console.WriteLine($"Noise Points (potential anomalies): {finalDbscan.GetNoiseCount()}");
    Console.WriteLine("Note: Insufficient clusters for silhouette calculation");
}

// =====================================================
// Final Summary
// =====================================================
Console.WriteLine("\n=== Summary ===\n");

var finalPredictions = new bool[transactions.Rows];
for (int i = 0; i < dbscanLabels.Length; i++)
{
    finalPredictions[i] = (int)dbscanLabels[i] == DBSCAN<double>.NoiseLabel;
}

var (finalPrecision, finalRecall, finalF1) = CalculateMetrics(trueLabels, finalPredictions);

Console.WriteLine("Detection Performance:");
Console.WriteLine($"  Precision: {finalPrecision:F4} ({finalPrecision * 100:F1}% of flagged transactions are actual fraud)");
Console.WriteLine($"  Recall:    {finalRecall:F4} ({finalRecall * 100:F1}% of actual frauds were detected)");
Console.WriteLine($"  F1-Score:  {finalF1:F4} (harmonic mean of precision and recall)");

int totalActualFrauds = trueLabels.Count(l => l);
int detectedFrauds = trueLabels.Zip(finalPredictions, (t, p) => t && p).Count(x => x);
int missedFrauds = totalActualFrauds - detectedFrauds;

Console.WriteLine($"\nFraud Detection:");
Console.WriteLine($"  Total actual fraudulent transactions: {totalActualFrauds}");
Console.WriteLine($"  Frauds detected: {detectedFrauds}");
Console.WriteLine($"  Frauds missed: {missedFrauds}");

Console.WriteLine("\n=== Sample Complete ===");

// =====================================================
// Helper Functions
// =====================================================

static (Matrix<double> features, bool[] trueLabels, string[] ids) GenerateTransactionData(int numTransactions, double anomalyRate)
{
    var random = new Random(42);
    var features = new Matrix<double>(numTransactions, 4);
    var trueLabels = new bool[numTransactions];
    var ids = new string[numTransactions];

    // Features: Amount, Hour of Day (0-23), Day of Week (0-6), Distance from Home (miles)

    for (int i = 0; i < numTransactions; i++)
    {
        ids[i] = $"TXN{i + 1:D5}";
        bool isAnomaly = random.NextDouble() < anomalyRate;
        trueLabels[i] = isAnomaly;

        if (isAnomaly)
        {
            // Generate anomalous transaction
            int anomalyType = random.Next(4);

            switch (anomalyType)
            {
                case 0: // Unusually large amount
                    features[i, 0] = 500 + random.NextDouble() * 9500; // $500-$10000
                    features[i, 1] = random.Next(24);
                    features[i, 2] = random.Next(7);
                    features[i, 3] = random.NextDouble() * 50;
                    break;

                case 1: // Unusual time (2-5 AM)
                    features[i, 0] = 20 + random.NextDouble() * 200;
                    features[i, 1] = 2 + random.Next(4); // 2-5 AM
                    features[i, 2] = random.Next(7);
                    features[i, 3] = 100 + random.NextDouble() * 400; // Far from home
                    break;

                case 2: // Very far from home
                    features[i, 0] = 50 + random.NextDouble() * 300;
                    features[i, 1] = random.Next(24);
                    features[i, 2] = random.Next(7);
                    features[i, 3] = 200 + random.NextDouble() * 800; // Very far (200-1000 mi)
                    break;

                case 3: // Multiple red flags
                    features[i, 0] = 300 + random.NextDouble() * 2000;
                    features[i, 1] = random.Next(4) + 1; // Late night (1-4 AM)
                    features[i, 2] = random.Next(7);
                    features[i, 3] = 150 + random.NextDouble() * 350;
                    break;
            }
        }
        else
        {
            // Generate normal transaction
            // Most transactions during business/evening hours
            double hourWeight = random.NextDouble();
            int hour;
            if (hourWeight < 0.3)
                hour = 9 + random.Next(4); // Morning (9 AM - 12 PM)
            else if (hourWeight < 0.7)
                hour = 12 + random.Next(6); // Afternoon (12 PM - 6 PM)
            else
                hour = 18 + random.Next(4); // Evening (6 PM - 10 PM)

            // Normal amounts follow log-normal distribution
            double amount = Math.Exp(3.0 + random.NextGaussian() * 0.8); // Most $20-$100
            amount = Math.Min(Math.Max(amount, 5), 400); // Clamp to $5-$400

            features[i, 0] = amount;
            features[i, 1] = hour;
            features[i, 2] = random.Next(7);
            features[i, 3] = Math.Abs(random.NextGaussian() * 15); // Usually near home
        }
    }

    return (features, trueLabels, ids);
}

static Matrix<double> NormalizeFeatures(Matrix<double> features)
{
    var normalized = new Matrix<double>(features.Rows, features.Columns);

    for (int j = 0; j < features.Columns; j++)
    {
        double mean = 0;
        for (int i = 0; i < features.Rows; i++)
            mean += features[i, j];
        mean /= features.Rows;

        double variance = 0;
        for (int i = 0; i < features.Rows; i++)
            variance += Math.Pow(features[i, j] - mean, 2);
        double std = Math.Sqrt(variance / features.Rows);

        for (int i = 0; i < features.Rows; i++)
        {
            normalized[i, j] = std > 0 ? (features[i, j] - mean) / std : 0;
        }
    }

    return normalized;
}

static (double precision, double recall, double f1) CalculateMetrics(bool[] trueLabels, bool[] predictions)
{
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;

    for (int i = 0; i < trueLabels.Length; i++)
    {
        if (predictions[i] && trueLabels[i])
            truePositives++;
        else if (predictions[i] && !trueLabels[i])
            falsePositives++;
        else if (!predictions[i] && trueLabels[i])
            falseNegatives++;
    }

    double precision = truePositives + falsePositives > 0
        ? (double)truePositives / (truePositives + falsePositives)
        : 0;

    double recall = truePositives + falseNegatives > 0
        ? (double)truePositives / (truePositives + falseNegatives)
        : 0;

    double f1 = precision + recall > 0
        ? 2 * precision * recall / (precision + recall)
        : 0;

    return (precision, recall, f1);
}

static double[] CalculateDistancesToCorePoints(Matrix<double> data, DBSCAN<double> dbscan)
{
    var distances = new double[data.Rows];
    var coreIndices = dbscan.GetCoreSampleIndices();

    if (coreIndices.Length == 0)
    {
        // If no core points, use distance to centroid
        for (int i = 0; i < data.Rows; i++)
        {
            distances[i] = 1.0;
        }
        return distances;
    }

    for (int i = 0; i < data.Rows; i++)
    {
        double minDist = double.MaxValue;

        foreach (int coreIdx in coreIndices)
        {
            double dist = 0;
            for (int j = 0; j < data.Columns; j++)
            {
                double diff = data[i, j] - data[coreIdx, j];
                dist += diff * diff;
            }
            dist = Math.Sqrt(dist);

            if (dist < minDist)
                minDist = dist;
        }

        distances[i] = minDist;
    }

    return distances;
}

static Matrix<double> ExtractRows(Matrix<double> data, List<int> indices)
{
    var result = new Matrix<double>(indices.Count, data.Columns);
    for (int i = 0; i < indices.Count; i++)
    {
        for (int j = 0; j < data.Columns; j++)
        {
            result[i, j] = data[indices[i], j];
        }
    }
    return result;
}

// Extension for Gaussian random numbers
public static class RandomExtensions
{
    public static double NextGaussian(this Random random)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
