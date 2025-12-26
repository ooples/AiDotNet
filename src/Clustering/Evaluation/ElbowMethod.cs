using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Elbow Method for determining the optimal number of clusters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Elbow Method plots the within-cluster sum of squares (WCSS) against
/// the number of clusters. The optimal K is at the "elbow" of the curve where
/// adding more clusters provides diminishing returns.
/// </para>
/// <para><b>For Beginners:</b> The Elbow Method finds where adding clusters stops helping.
///
/// As you increase K:
/// - K=1: All data in one cluster (high WCSS)
/// - K=2: Two clusters (lower WCSS)
/// - K=3: Three clusters (even lower WCSS)
/// - Eventually, improvements become tiny
///
/// The "elbow" is where the curve bends:
/// - Before elbow: Big improvements per cluster
/// - After elbow: Tiny improvements per cluster
///
/// The elbow point is often the best K!
/// </para>
/// </remarks>
public class ElbowMethod<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int? _randomState;

    /// <summary>
    /// Initializes a new ElbowMethod instance.
    /// </summary>
    /// <param name="randomState">Random seed for reproducibility.</param>
    public ElbowMethod(int? randomState = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _randomState = randomState;
    }

    /// <summary>
    /// Computes WCSS for a range of cluster counts.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="minClusters">Minimum number of clusters to try.</param>
    /// <param name="maxClusters">Maximum number of clusters to try.</param>
    /// <returns>Elbow method results including detected elbow point.</returns>
    public ElbowMethodResult Compute(Matrix<T> data, int minClusters = 1, int maxClusters = 10)
    {
        int n = data.Rows;

        if (maxClusters > n)
        {
            maxClusters = n;
        }

        var kValues = new List<int>();
        var wcssValues = new List<double>();

        for (int k = minClusters; k <= maxClusters; k++)
        {
            double wcss = ComputeWCSS(data, k);
            kValues.Add(k);
            wcssValues.Add(wcss);
        }

        // Detect elbow using the Kneedle algorithm
        int elbowK = DetectElbow(kValues.ToArray(), wcssValues.ToArray());

        // Compute improvement rates
        var improvementRates = new double[wcssValues.Count];
        for (int i = 1; i < wcssValues.Count; i++)
        {
            if (wcssValues[i - 1] > 0)
            {
                improvementRates[i] = (wcssValues[i - 1] - wcssValues[i]) / wcssValues[i - 1];
            }
        }

        return new ElbowMethodResult
        {
            ElbowK = elbowK,
            KValues = kValues.ToArray(),
            WCSSValues = wcssValues.ToArray(),
            ImprovementRates = improvementRates
        };
    }

    private double ComputeWCSS(Matrix<T> data, int k)
    {
        int n = data.Rows;

        if (k >= n)
        {
            return 0;
        }

        var kmeans = new KMeans<T>(new KMeansOptions<T>
        {
            NumClusters = k,
            MaxIterations = 100,
            NumInitializations = 3,
            RandomState = _randomState
        });

        try
        {
            kmeans.Train(data);
            return kmeans.Inertia is not null ? _numOps.ToDouble(kmeans.Inertia) : double.MaxValue;
        }
        catch
        {
            return double.MaxValue;
        }
    }

    /// <summary>
    /// Detects the elbow point using the Kneedle algorithm.
    /// </summary>
    private int DetectElbow(int[] kValues, double[] wcssValues)
    {
        if (kValues.Length < 3)
        {
            return kValues[0];
        }

        // Normalize values to [0, 1] range
        double maxK = kValues.Max();
        double minK = kValues.Min();
        double rangeK = maxK - minK;

        double maxWcss = wcssValues.Max();
        double minWcss = wcssValues.Min();
        double rangeWcss = maxWcss - minWcss;

        if (rangeK == 0 || rangeWcss == 0)
        {
            return kValues[0];
        }

        var normalizedK = new double[kValues.Length];
        var normalizedWcss = new double[wcssValues.Length];

        for (int i = 0; i < kValues.Length; i++)
        {
            normalizedK[i] = (kValues[i] - minK) / rangeK;
            normalizedWcss[i] = (wcssValues[i] - minWcss) / rangeWcss;
        }

        // Compute distance from each point to the line from first to last point
        double x1 = normalizedK[0];
        double y1 = normalizedWcss[0];
        double x2 = normalizedK[kValues.Length - 1];
        double y2 = normalizedWcss[wcssValues.Length - 1];

        // Line equation: ax + by + c = 0
        double a = y2 - y1;
        double b = x1 - x2;
        double c = x2 * y1 - x1 * y2;
        double denom = Math.Sqrt(a * a + b * b);

        if (denom == 0)
        {
            return kValues[0];
        }

        // Find point with maximum distance (the elbow)
        double maxDistance = 0;
        int elbowIdx = 0;

        for (int i = 0; i < kValues.Length; i++)
        {
            double distance = Math.Abs(a * normalizedK[i] + b * normalizedWcss[i] + c) / denom;

            // For decreasing curve, the elbow is where the curve bends most
            if (distance > maxDistance)
            {
                maxDistance = distance;
                elbowIdx = i;
            }
        }

        return kValues[elbowIdx];
    }
}

/// <summary>
/// Results from Elbow Method analysis.
/// </summary>
public class ElbowMethodResult
{
    /// <summary>
    /// The detected elbow point (optimal K).
    /// </summary>
    public int ElbowK { get; set; }

    /// <summary>
    /// The K values tested.
    /// </summary>
    public int[] KValues { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Within-cluster sum of squares for each K.
    /// </summary>
    public double[] WCSSValues { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Improvement rate from K-1 to K for each K.
    /// </summary>
    public double[] ImprovementRates { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets a summary of the Elbow analysis.
    /// </summary>
    public string GetSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Elbow Method Analysis - Detected Elbow: K={ElbowK}");
        sb.AppendLine();
        sb.AppendLine("K\tWCSS\t\tImprovement");
        sb.AppendLine("---\t----\t\t-----------");

        for (int i = 0; i < KValues.Length; i++)
        {
            string marker = KValues[i] == ElbowK ? " *" : "";
            string improvement = i == 0 ? "-" : $"{ImprovementRates[i]:P1}";
            sb.AppendLine($"{KValues[i]}{marker}\t{WCSSValues[i]:F2}\t\t{improvement}");
        }

        return sb.ToString();
    }
}
