using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Gap Statistic for determining the optimal number of clusters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Gap Statistic compares the within-cluster dispersion of the data to that
/// expected under a null reference distribution (uniform random). The optimal K
/// is where the gap between observed and expected is largest.
/// </para>
/// <para>
/// Gap(k) = E*[log(W_k)] - log(W_k)
/// Where:
/// - W_k = within-cluster sum of squares for k clusters
/// - E*[log(W_k)] = expected value under null reference distribution
/// </para>
/// <para><b>For Beginners:</b> Gap Statistic finds the "right" number of clusters.
///
/// The idea:
/// 1. Cluster the data with k=1, 2, 3, ... clusters
/// 2. For each k, measure how "compact" the clusters are (WCSS)
/// 3. Compare this to what you'd expect from random data
/// 4. The best k is where real data is MUCH better than random
///
/// Think of it like:
/// - "At k=3, my clustering is 10x better than random"
/// - "At k=4, it's only 2x better"
/// - "So k=3 is probably right!"
/// </para>
/// </remarks>
public class GapStatistic<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numReferences;
    private readonly int? _randomState;

    /// <summary>
    /// Initializes a new GapStatistic instance.
    /// </summary>
    /// <param name="numReferences">Number of reference datasets to generate. Default is 10.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    public GapStatistic(int numReferences = 10, int? randomState = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numReferences = numReferences;
        _randomState = randomState;
    }

    /// <summary>
    /// Computes the Gap Statistic for a range of cluster counts.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="minClusters">Minimum number of clusters to try.</param>
    /// <param name="maxClusters">Maximum number of clusters to try.</param>
    /// <returns>Gap statistic results including optimal K.</returns>
    public GapStatisticResult Compute(Matrix<T> data, int minClusters = 1, int maxClusters = 10)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (maxClusters > n)
        {
            maxClusters = n;
        }

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Compute data bounds for uniform reference generation
        var minBounds = new double[d];
        var maxBounds = new double[d];

        for (int j = 0; j < d; j++)
        {
            minBounds[j] = double.MaxValue;
            maxBounds[j] = double.MinValue;

            for (int i = 0; i < n; i++)
            {
                double val = _numOps.ToDouble(data[i, j]);
                minBounds[j] = Math.Min(minBounds[j], val);
                maxBounds[j] = Math.Max(maxBounds[j], val);
            }
        }

        var kValues = new List<int>();
        var gapValues = new List<double>();
        var sdValues = new List<double>();
        var wkValues = new List<double>();
        var wkbValues = new List<double>();

        for (int k = minClusters; k <= maxClusters; k++)
        {
            // Compute W_k for real data
            double wk = ComputeWCSS(data, k);

            // Compute E*[log(W_k)] from reference datasets
            var refLogWk = new double[_numReferences];

            for (int b = 0; b < _numReferences; b++)
            {
                var refData = GenerateUniformReference(n, d, minBounds, maxBounds, rand);
                double refWk = ComputeWCSS(refData, k);
                refLogWk[b] = Math.Log(Math.Max(refWk, 1e-10));
            }

            // Compute mean and standard deviation
            double meanLogWk = refLogWk.Average();
            double sdLogWk = ComputeStd(refLogWk);

            // Gap = E*[log(W_k)] - log(W_k)
            double logWk = Math.Log(Math.Max(wk, 1e-10));
            double gap = meanLogWk - logWk;

            // Adjust sd for sampling (multiply by sqrt(1 + 1/B))
            double sk = sdLogWk * Math.Sqrt(1 + 1.0 / _numReferences);

            kValues.Add(k);
            gapValues.Add(gap);
            sdValues.Add(sk);
            wkValues.Add(wk);
            wkbValues.Add(Math.Exp(meanLogWk));
        }

        // Find optimal K using Gap criterion: Gap(k) >= Gap(k+1) - s(k+1)
        int optimalK = kValues[0];
        for (int i = 0; i < kValues.Count - 1; i++)
        {
            if (gapValues[i] >= gapValues[i + 1] - sdValues[i + 1])
            {
                optimalK = kValues[i];
                break;
            }
        }

        // If no optimal found with criterion, use max gap
        if (optimalK == kValues[0] && kValues.Count > 1)
        {
            int maxIdx = 0;
            for (int i = 1; i < gapValues.Count; i++)
            {
                if (gapValues[i] > gapValues[maxIdx])
                {
                    maxIdx = i;
                }
            }
            optimalK = kValues[maxIdx];
        }

        return new GapStatisticResult
        {
            OptimalK = optimalK,
            KValues = kValues.ToArray(),
            GapValues = gapValues.ToArray(),
            StandardErrors = sdValues.ToArray(),
            WCSSValues = wkValues.ToArray(),
            ReferenceWCSSValues = wkbValues.ToArray()
        };
    }

    private double ComputeWCSS(Matrix<T> data, int k)
    {
        int n = data.Rows;

        if (k >= n)
        {
            return 0; // Each point is its own cluster
        }

        // Use K-Means clustering
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

    private Matrix<T> GenerateUniformReference(int n, int d, double[] minBounds, double[] maxBounds, Random rand)
    {
        var refData = new Matrix<T>(n, d);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double val = minBounds[j] + rand.NextDouble() * (maxBounds[j] - minBounds[j]);
                refData[i, j] = _numOps.FromDouble(val);
            }
        }

        return refData;
    }

    private double ComputeStd(double[] values)
    {
        if (values.Length < 2) return 0;

        double mean = values.Average();
        double sumSq = values.Sum(v => (v - mean) * (v - mean));
        return Math.Sqrt(sumSq / (values.Length - 1));
    }
}

/// <summary>
/// Results from Gap Statistic analysis.
/// </summary>
public class GapStatisticResult
{
    /// <summary>
    /// The optimal number of clusters based on the Gap criterion.
    /// </summary>
    public int OptimalK { get; set; }

    /// <summary>
    /// The K values tested.
    /// </summary>
    public int[] KValues { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gap values for each K.
    /// </summary>
    public double[] GapValues { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Standard errors for each Gap value.
    /// </summary>
    public double[] StandardErrors { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Within-cluster sum of squares for each K.
    /// </summary>
    public double[] WCSSValues { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Reference (expected) WCSS values for each K.
    /// </summary>
    public double[] ReferenceWCSSValues { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets a summary of the Gap analysis.
    /// </summary>
    public string GetSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Gap Statistic Analysis - Optimal K: {OptimalK}");
        sb.AppendLine();
        sb.AppendLine("K\tGap\tStd Error\tWCSS");
        sb.AppendLine("---\t---\t---------\t----");

        for (int i = 0; i < KValues.Length; i++)
        {
            string marker = KValues[i] == OptimalK ? " *" : "";
            sb.AppendLine($"{KValues[i]}{marker}\t{GapValues[i]:F4}\t{StandardErrors[i]:F4}\t\t{WCSSValues[i]:F2}");
        }

        return sb.ToString();
    }
}
