using AiDotNet.Clustering.Evaluation;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Customer Segmentation ===");
Console.WriteLine("Customer segmentation using K-Means clustering\n");

// Generate realistic customer data
var (features, customerIds) = GenerateCustomerData(500);

Console.WriteLine($"Generated {features.Rows} customer records");
Console.WriteLine($"Features: Age, Annual Income ($K), Spending Score (1-100)\n");

// Display sample data
Console.WriteLine("Sample Customer Data:");
Console.WriteLine("----------------------------------------------");
Console.WriteLine("Customer ID | Age | Income ($K) | Spending Score");
Console.WriteLine("----------------------------------------------");
for (int i = 0; i < 5; i++)
{
    Console.WriteLine($"   {customerIds[i],6}   | {features[i, 0],3:F0} |    {features[i, 1],6:F1}    |      {features[i, 2],3:F0}");
}
Console.WriteLine("            ...    (495 more customers)\n");

// Normalize features for better clustering
var normalizedFeatures = NormalizeFeatures(features);

// Find optimal number of clusters using the elbow method
Console.WriteLine("Finding optimal number of clusters...\n");
var clusterResults = new List<(int k, double inertia, double silhouette)>();

for (int k = 2; k <= 8; k++)
{
    var kmeans = new KMeans<double>(new KMeansOptions<double>
    {
        NumClusters = k,
        MaxIterations = 300,
        Tolerance = 1e-4,
        RandomState = 42,
        NumInitializations = 10,
        InitMethod = KMeansInitMethod.KMeansPlusPlus
    });

    kmeans.Train(normalizedFeatures);
    var labels = kmeans.Labels!;

    var silhouetteCalculator = new SilhouetteScore<double>();
    double silhouette = silhouetteCalculator.Compute(normalizedFeatures, labels);
    double inertia = Convert.ToDouble(kmeans.Inertia);

    clusterResults.Add((k, inertia, silhouette));
    Console.WriteLine($"  K={k}: Inertia={inertia,8:F2}, Silhouette Score={silhouette:F4}");
}

// Find the best K based on silhouette score
var bestResult = clusterResults.OrderByDescending(r => r.silhouette).First();
int optimalK = bestResult.k;
Console.WriteLine($"\nOptimal K based on Silhouette Score: {optimalK}");

// Train final model with optimal K
Console.WriteLine($"\nTraining final K-Means model with K={optimalK}...");
var finalKMeans = new KMeans<double>(new KMeansOptions<double>
{
    NumClusters = optimalK,
    MaxIterations = 300,
    Tolerance = 1e-4,
    RandomState = 42,
    NumInitializations = 10,
    InitMethod = KMeansInitMethod.KMeansPlusPlus
});

finalKMeans.Train(normalizedFeatures);
var finalLabels = finalKMeans.Labels!;

Console.WriteLine($"Converged in {finalKMeans.NumIterations} iterations\n");

// Analyze cluster characteristics
Console.WriteLine("=== Cluster Analysis ===\n");
AnalyzeClusters(features, finalLabels, optimalK);

// Compute and display comprehensive metrics
Console.WriteLine("\n=== Clustering Quality Metrics ===\n");
var evaluator = new ClusteringEvaluator<double>();
var evalResult = evaluator.EvaluateAll(normalizedFeatures, finalLabels);

Console.WriteLine($"Number of Clusters: {evalResult.NumClusters}");
Console.WriteLine($"Total Customers: {evalResult.NumPoints}");
Console.WriteLine("\nCluster Sizes:");
for (int i = 0; i < evalResult.ClusterSizes.Length; i++)
{
    double percentage = 100.0 * evalResult.ClusterSizes[i] / evalResult.NumPoints;
    Console.WriteLine($"  Cluster {i}: {evalResult.ClusterSizes[i]} customers ({percentage:F1}%)");
}

Console.WriteLine("\nInternal Validity Metrics:");
Console.WriteLine("----------------------------------------------");
foreach (var metric in evalResult.InternalMetrics)
{
    string interpretation = InterpretMetric(metric.Key, metric.Value);
    Console.WriteLine($"  {metric.Key,-28}: {metric.Value,8:F4}  {interpretation}");
}

// Display cluster centers in original scale
Console.WriteLine("\n=== Cluster Centers (Original Scale) ===\n");
DisplayClusterCenters(features, finalLabels, optimalK);

// Generate marketing insights
Console.WriteLine("\n=== Marketing Insights ===\n");
GenerateMarketingInsights(features, finalLabels, optimalK);

Console.WriteLine("\n=== Sample Complete ===");

// =====================================================
// Helper Functions
// =====================================================

static (Matrix<double> features, string[] customerIds) GenerateCustomerData(int numCustomers)
{
    var random = new Random(42);
    var features = new Matrix<double>(numCustomers, 3);
    var customerIds = new string[numCustomers];

    // Define customer segments with realistic characteristics
    // Segment 1: Young, moderate income, high spenders (trend-conscious)
    // Segment 2: Middle-aged, high income, moderate spenders (established)
    // Segment 3: Older, moderate income, low spenders (conservative)
    // Segment 4: Young professionals, high income, high spenders (affluent millennials)
    // Segment 5: Budget-conscious across ages

    for (int i = 0; i < numCustomers; i++)
    {
        customerIds[i] = $"CUST{i + 1:D4}";

        int segment = random.Next(5);
        switch (segment)
        {
            case 0: // Young, moderate income, high spenders
                features[i, 0] = 20 + random.NextDouble() * 15; // Age: 20-35
                features[i, 1] = 30 + random.NextDouble() * 40; // Income: 30K-70K
                features[i, 2] = 60 + random.NextDouble() * 40; // Spending: 60-100
                break;

            case 1: // Middle-aged, high income, moderate spenders
                features[i, 0] = 40 + random.NextDouble() * 20; // Age: 40-60
                features[i, 1] = 80 + random.NextDouble() * 70; // Income: 80K-150K
                features[i, 2] = 40 + random.NextDouble() * 30; // Spending: 40-70
                break;

            case 2: // Older, moderate income, low spenders
                features[i, 0] = 55 + random.NextDouble() * 20; // Age: 55-75
                features[i, 1] = 40 + random.NextDouble() * 40; // Income: 40K-80K
                features[i, 2] = 10 + random.NextDouble() * 30; // Spending: 10-40
                break;

            case 3: // Young professionals, high income, high spenders
                features[i, 0] = 25 + random.NextDouble() * 15; // Age: 25-40
                features[i, 1] = 100 + random.NextDouble() * 80; // Income: 100K-180K
                features[i, 2] = 70 + random.NextDouble() * 30; // Spending: 70-100
                break;

            case 4: // Budget-conscious across ages
                features[i, 0] = 25 + random.NextDouble() * 45; // Age: 25-70
                features[i, 1] = 20 + random.NextDouble() * 50; // Income: 20K-70K
                features[i, 2] = 5 + random.NextDouble() * 25;  // Spending: 5-30
                break;
        }

        // Add some noise
        features[i, 0] += (random.NextDouble() - 0.5) * 5;
        features[i, 1] += (random.NextDouble() - 0.5) * 10;
        features[i, 2] += (random.NextDouble() - 0.5) * 8;

        // Ensure values are within reasonable bounds
        features[i, 0] = Math.Max(18, Math.Min(80, features[i, 0]));
        features[i, 1] = Math.Max(10, Math.Min(200, features[i, 1]));
        features[i, 2] = Math.Max(1, Math.Min(100, features[i, 2]));
    }

    return (features, customerIds);
}

static Matrix<double> NormalizeFeatures(Matrix<double> features)
{
    var normalized = new Matrix<double>(features.Rows, features.Columns);

    // Z-score normalization for each feature
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

static void AnalyzeClusters(Matrix<double> features, Vector<double> labels, int numClusters)
{
    for (int k = 0; k < numClusters; k++)
    {
        var clusterIndices = new List<int>();
        for (int i = 0; i < labels.Length; i++)
        {
            if ((int)labels[i] == k)
                clusterIndices.Add(i);
        }

        if (clusterIndices.Count == 0) continue;

        // Calculate statistics for each feature
        double avgAge = clusterIndices.Average(i => features[i, 0]);
        double avgIncome = clusterIndices.Average(i => features[i, 1]);
        double avgSpending = clusterIndices.Average(i => features[i, 2]);

        double minAge = clusterIndices.Min(i => features[i, 0]);
        double maxAge = clusterIndices.Max(i => features[i, 0]);
        double minIncome = clusterIndices.Min(i => features[i, 1]);
        double maxIncome = clusterIndices.Max(i => features[i, 1]);
        double minSpending = clusterIndices.Min(i => features[i, 2]);
        double maxSpending = clusterIndices.Max(i => features[i, 2]);

        string segmentName = GetSegmentName(avgAge, avgIncome, avgSpending);

        Console.WriteLine($"Cluster {k}: \"{segmentName}\"");
        Console.WriteLine($"  Size: {clusterIndices.Count} customers");
        Console.WriteLine($"  Age:            Avg={avgAge,5:F1}, Range=[{minAge,4:F0}-{maxAge,4:F0}]");
        Console.WriteLine($"  Income ($K):    Avg={avgIncome,5:F1}, Range=[{minIncome,4:F0}-{maxIncome,4:F0}]");
        Console.WriteLine($"  Spending Score: Avg={avgSpending,5:F1}, Range=[{minSpending,4:F0}-{maxSpending,4:F0}]");
        Console.WriteLine();
    }
}

static string GetSegmentName(double avgAge, double avgIncome, double avgSpending)
{
    if (avgIncome > 90 && avgSpending > 70)
        return "Premium Spenders";
    if (avgIncome > 80 && avgSpending >= 40 && avgSpending <= 70)
        return "Established Moderates";
    if (avgAge > 50 && avgSpending < 35)
        return "Conservative Seniors";
    if (avgAge < 35 && avgSpending > 60)
        return "Young Trendsetters";
    if (avgSpending < 30)
        return "Budget Conscious";
    return "General Consumers";
}

static void DisplayClusterCenters(Matrix<double> features, Vector<double> labels, int numClusters)
{
    Console.WriteLine("Cluster |   Age   | Income ($K) | Spending Score");
    Console.WriteLine("--------|---------|-------------|---------------");

    for (int k = 0; k < numClusters; k++)
    {
        double sumAge = 0, sumIncome = 0, sumSpending = 0;
        int count = 0;

        for (int i = 0; i < labels.Length; i++)
        {
            if ((int)labels[i] == k)
            {
                sumAge += features[i, 0];
                sumIncome += features[i, 1];
                sumSpending += features[i, 2];
                count++;
            }
        }

        if (count > 0)
        {
            Console.WriteLine($"   {k}    |  {sumAge / count,5:F1}  |    {sumIncome / count,6:F1}   |     {sumSpending / count,5:F1}");
        }
    }
}

static void GenerateMarketingInsights(Matrix<double> features, Vector<double> labels, int numClusters)
{
    for (int k = 0; k < numClusters; k++)
    {
        var clusterIndices = new List<int>();
        for (int i = 0; i < labels.Length; i++)
        {
            if ((int)labels[i] == k)
                clusterIndices.Add(i);
        }

        if (clusterIndices.Count == 0) continue;

        double avgAge = clusterIndices.Average(i => features[i, 0]);
        double avgIncome = clusterIndices.Average(i => features[i, 1]);
        double avgSpending = clusterIndices.Average(i => features[i, 2]);

        string segmentName = GetSegmentName(avgAge, avgIncome, avgSpending);
        Console.WriteLine($"Cluster {k} - {segmentName}:");

        // Generate targeted recommendations
        if (avgIncome > 90 && avgSpending > 70)
        {
            Console.WriteLine("  -> Target with premium product launches and exclusive offers");
            Console.WriteLine("  -> High-value customers - prioritize retention programs");
            Console.WriteLine("  -> Offer VIP membership and early access privileges");
        }
        else if (avgIncome > 80 && avgSpending >= 40 && avgSpending <= 70)
        {
            Console.WriteLine("  -> Opportunity to increase spending through personalized recommendations");
            Console.WriteLine("  -> Focus on quality and value propositions");
            Console.WriteLine("  -> Cross-sell complementary premium products");
        }
        else if (avgAge > 50 && avgSpending < 35)
        {
            Console.WriteLine("  -> Emphasize reliability, quality, and value");
            Console.WriteLine("  -> Simple, clear communication preferred");
            Console.WriteLine("  -> Loyalty discounts may increase engagement");
        }
        else if (avgAge < 35 && avgSpending > 60)
        {
            Console.WriteLine("  -> Leverage social media and influencer marketing");
            Console.WriteLine("  -> Trendy products and limited editions appeal to this group");
            Console.WriteLine("  -> Mobile-first engagement strategy recommended");
        }
        else if (avgSpending < 30)
        {
            Console.WriteLine("  -> Price-sensitive - emphasize discounts and promotions");
            Console.WriteLine("  -> Bundle deals and loyalty rewards effective");
            Console.WriteLine("  -> Email marketing with sale notifications");
        }
        else
        {
            Console.WriteLine("  -> General marketing campaigns appropriate");
            Console.WriteLine("  -> A/B testing recommended to find optimal messaging");
        }
        Console.WriteLine();
    }
}

static string InterpretMetric(string metricName, double value)
{
    return metricName switch
    {
        "Silhouette Score" => value > 0.5 ? "(Good)" : value > 0.25 ? "(Fair)" : "(Poor)",
        "Davies-Bouldin Index" => value < 1.0 ? "(Good)" : value < 2.0 ? "(Fair)" : "(Poor)",
        "Calinski-Harabasz Index" => value > 100 ? "(Good)" : value > 50 ? "(Fair)" : "(Low)",
        "Dunn Index" => value > 0.5 ? "(Good)" : value > 0.1 ? "(Fair)" : "(Poor)",
        "Connectivity Index" => value < 50 ? "(Good)" : value < 100 ? "(Fair)" : "(High)",
        _ => ""
    };
}
