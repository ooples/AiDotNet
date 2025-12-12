using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Analysis results for model weights to guide compression decisions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class WeightAnalysisResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the total number of weights analyzed.
    /// </summary>
    public long TotalWeights { get; set; }

    /// <summary>
    /// Gets or sets the number of weights with near-zero magnitude.
    /// </summary>
    public long NearZeroWeights { get; set; }

    /// <summary>
    /// Gets or sets the fraction of weights that are near-zero (pruning potential).
    /// </summary>
    public T PruningPotential { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the mean weight magnitude.
    /// </summary>
    public T MeanMagnitude { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the standard deviation of weight magnitudes.
    /// </summary>
    public T StdDevMagnitude { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the maximum weight magnitude.
    /// </summary>
    public T MaxMagnitude { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the minimum weight magnitude.
    /// </summary>
    public T MinMagnitude { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the number of unique weight values (for clustering potential).
    /// </summary>
    public long UniqueValues { get; set; }

    /// <summary>
    /// Gets or sets the clustering potential (how much quantization will help).
    /// </summary>
    public T ClusteringPotential { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the entropy of weight distribution (for Huffman coding potential).
    /// </summary>
    public T Entropy { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the recommended compression technique based on analysis.
    /// </summary>
    public CompressionType RecommendedTechnique { get; set; }

    /// <summary>
    /// Gets or sets the reasoning for the recommendation.
    /// </summary>
    public string RecommendationReasoning { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the estimated compression ratio achievable.
    /// </summary>
    public T EstimatedCompressionRatio { get; set; } = NumOps.Zero;

    /// <summary>
    /// Gets or sets the recommended hyperparameters for the compression technique.
    /// </summary>
    public Dictionary<string, object> RecommendedParameters { get; set; } = new();
}

/// <summary>
/// Analyzes model weights to determine optimal compression strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// CompressionAnalyzer examines model weight distributions to recommend the best compression
/// technique and hyperparameters. It analyzes properties like weight sparsity, magnitude
/// distribution, and redundancy to make informed recommendations.
/// </para>
/// <para><b>For Beginners:</b> Before compressing a model, it helps to understand its weights.
///
/// This analyzer looks at your model's weights and answers questions like:
/// - Are many weights already close to zero? (Good for pruning)
/// - Are weights clustered around certain values? (Good for quantization)
/// - What's the distribution of weight values? (Affects all techniques)
///
/// Based on this analysis, it recommends:
/// - Which compression technique to use
/// - What settings (hyperparameters) to use
/// - What compression ratio to expect
///
/// This helps you make informed decisions without trial-and-error.
/// </para>
/// </remarks>
public class CompressionAnalyzer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _nearZeroThreshold;
    private readonly int _histogramBins;

    /// <summary>
    /// Initializes a new instance of the CompressionAnalyzer class.
    /// </summary>
    /// <param name="nearZeroThreshold">Threshold for considering a weight as "near zero" (default: 0.01).</param>
    /// <param name="histogramBins">Number of bins for histogram analysis (default: 256).</param>
    public CompressionAnalyzer(double nearZeroThreshold = 0.01, int histogramBins = 256)
    {
        _nearZeroThreshold = nearZeroThreshold;
        _histogramBins = histogramBins;
    }

    /// <summary>
    /// Analyzes model weights and returns compression recommendations.
    /// </summary>
    /// <param name="weights">The model weights to analyze.</param>
    /// <param name="isConvolutional">Whether the weights are from convolutional layers (default: false).</param>
    /// <returns>Analysis results with compression recommendations.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pass your model weights to get compression recommendations.
    ///
    /// The isConvolutional parameter helps optimize recommendations:
    /// - Convolutional layers: typically use lower pruning rates (60-70%)
    /// - Fully-connected layers: can use higher pruning rates (90-95%)
    ///
    /// This is because convolutional layers have more structured, important patterns.
    /// </para>
    /// </remarks>
    public WeightAnalysisResult<T> Analyze(Vector<T> weights, bool isConvolutional = false)
    {
        if (weights == null || weights.Length == 0)
        {
            throw new ArgumentException("Weights cannot be null or empty.", nameof(weights));
        }

        var result = new WeightAnalysisResult<T>
        {
            TotalWeights = weights.Length
        };

        // Calculate magnitude statistics
        var magnitudes = new double[weights.Length];
        double sum = 0;
        double maxMag = double.MinValue;
        double minMag = double.MaxValue;
        long nearZeroCount = 0;

        for (int i = 0; i < weights.Length; i++)
        {
            var mag = Math.Abs(NumOps.ToDouble(weights[i]));
            magnitudes[i] = mag;
            sum += mag;

            if (mag > maxMag) maxMag = mag;
            if (mag < minMag) minMag = mag;
            if (mag < _nearZeroThreshold) nearZeroCount++;
        }

        var mean = sum / weights.Length;
        result.MeanMagnitude = NumOps.FromDouble(mean);
        result.MaxMagnitude = NumOps.FromDouble(maxMag);
        result.MinMagnitude = NumOps.FromDouble(minMag);
        result.NearZeroWeights = nearZeroCount;
        result.PruningPotential = NumOps.FromDouble((double)nearZeroCount / weights.Length);

        // Calculate standard deviation
        double variance = 0;
        for (int i = 0; i < magnitudes.Length; i++)
        {
            variance += (magnitudes[i] - mean) * (magnitudes[i] - mean);
        }
        variance /= weights.Length;
        result.StdDevMagnitude = NumOps.FromDouble(Math.Sqrt(variance));

        // Estimate unique values (sample-based for efficiency)
        var uniqueValues = EstimateUniqueValues(weights);
        result.UniqueValues = uniqueValues;

        // Calculate clustering potential (lower unique values = better clustering)
        var clusteringPotential = 1.0 - Math.Min(1.0, (double)uniqueValues / weights.Length);
        result.ClusteringPotential = NumOps.FromDouble(clusteringPotential);

        // Calculate entropy for Huffman coding potential
        var entropy = CalculateEntropy(magnitudes);
        result.Entropy = NumOps.FromDouble(entropy);

        // Make recommendation based on analysis
        MakeRecommendation(result, isConvolutional);

        return result;
    }

    /// <summary>
    /// Estimates the number of unique values using sampling for large arrays.
    /// </summary>
    private long EstimateUniqueValues(Vector<T> weights)
    {
        const int maxSamples = 10000;
        var sampleSize = Math.Min(weights.Length, maxSamples);

        var uniqueSet = new HashSet<double>();
        var step = weights.Length / sampleSize;

        for (int i = 0; i < weights.Length && uniqueSet.Count < maxSamples; i += Math.Max(1, step))
        {
            // Round to 4 decimal places for practical uniqueness
            var rounded = Math.Round(NumOps.ToDouble(weights[i]), 4);
            uniqueSet.Add(rounded);
        }

        // Scale up if we sampled
        if (weights.Length > maxSamples)
        {
            return (long)(uniqueSet.Count * ((double)weights.Length / sampleSize));
        }

        return uniqueSet.Count;
    }

    /// <summary>
    /// Calculates the entropy of the weight distribution.
    /// </summary>
    private double CalculateEntropy(double[] magnitudes)
    {
        // Create histogram
        var max = magnitudes.Max();
        var min = magnitudes.Min();
        var range = max - min;

        if (range < 1e-10) return 0; // All values are the same

        var histogram = new int[_histogramBins];
        var binSize = range / _histogramBins;

        foreach (var mag in magnitudes)
        {
            var bin = (int)((mag - min) / binSize);
            bin = Math.Min(bin, _histogramBins - 1);
            histogram[bin]++;
        }

        // Calculate entropy using LINQ for cleaner filtering
        double entropy = 0;
        var total = (double)magnitudes.Length;

        foreach (var count in histogram.Where(c => c > 0))
        {
            var p = count / total;
            entropy -= p * (Math.Log(p) / Math.Log(2));
        }

        return entropy;
    }

    /// <summary>
    /// Makes compression recommendations based on analysis results.
    /// </summary>
    private void MakeRecommendation(WeightAnalysisResult<T> result, bool isConvolutional)
    {
        var pruningPotential = NumOps.ToDouble(result.PruningPotential);
        var clusteringPotential = NumOps.ToDouble(result.ClusteringPotential);
        var entropy = NumOps.ToDouble(result.Entropy);

        var reasoning = new List<string>();
        var parameters = new Dictionary<string, object>();

        // Determine recommended technique based on analysis
        if (pruningPotential > 0.7 && clusteringPotential > 0.5)
        {
            // High pruning and clustering potential - use Deep Compression
            result.RecommendedTechnique = CompressionType.HybridHuffmanClustering;

            var targetSparsity = isConvolutional ? 0.65 : 0.92;
            var numClusters = clusteringPotential > 0.7 ? 32 : 64;

            parameters["pruningSparsity"] = targetSparsity;
            parameters["numClusters"] = numClusters;
            parameters["huffmanPrecision"] = 4;

            reasoning.Add($"High pruning potential ({pruningPotential:P0} of weights near zero)");
            reasoning.Add($"Good clustering potential ({clusteringPotential:P0})");
            reasoning.Add("Recommending Deep Compression (pruning + quantization + Huffman)");
            reasoning.Add($"Using {(isConvolutional ? "conservative" : "aggressive")} pruning for {(isConvolutional ? "convolutional" : "fully-connected")} layers");

            // Estimate compression ratio
            var pruningRatio = 1.0 / (1.0 - targetSparsity);
            var quantizationRatio = 32.0 / (Math.Log(numClusters) / Math.Log(2));
            var estimatedRatio = pruningRatio * quantizationRatio * 0.7; // 0.7 factor for overhead
            result.EstimatedCompressionRatio = NumOps.FromDouble(estimatedRatio);
        }
        else if (pruningPotential > 0.5)
        {
            // Good pruning potential
            result.RecommendedTechnique = CompressionType.SparsePruning;

            var targetSparsity = isConvolutional
                ? Math.Min(0.7, pruningPotential)
                : Math.Min(0.9, pruningPotential + 0.1);

            parameters["sparsityTarget"] = targetSparsity;

            reasoning.Add($"Moderate to high pruning potential ({pruningPotential:P0})");
            reasoning.Add("Recommending Sparse Pruning");
            reasoning.Add($"Target sparsity: {targetSparsity:P0}");

            result.EstimatedCompressionRatio = NumOps.FromDouble(1.0 / (1.0 - targetSparsity));
        }
        else if (clusteringPotential > 0.4)
        {
            // Good clustering potential
            result.RecommendedTechnique = CompressionType.WeightClustering;

            var numClusters = clusteringPotential > 0.7 ? 32 :
                             clusteringPotential > 0.5 ? 64 : 128;

            parameters["numClusters"] = numClusters;

            reasoning.Add($"Good clustering potential ({clusteringPotential:P0})");
            reasoning.Add($"Weight distribution suggests {numClusters} clusters");
            reasoning.Add("Recommending Weight Clustering (k-means quantization)");

            result.EstimatedCompressionRatio = NumOps.FromDouble(32.0 / (Math.Log(numClusters) / Math.Log(2)));
        }
        else if (entropy < 6.0)
        {
            // Low entropy - Huffman coding will be effective
            result.RecommendedTechnique = CompressionType.HuffmanEncoding;

            var precision = entropy < 4.0 ? 4 : 6;
            parameters["precision"] = precision;

            reasoning.Add($"Low entropy ({entropy:F2} bits) indicates good encoding potential");
            reasoning.Add("Recommending Huffman Encoding");

            result.EstimatedCompressionRatio = NumOps.FromDouble(32.0 / entropy);
        }
        else
        {
            // Default to weight clustering with moderate settings
            result.RecommendedTechnique = CompressionType.WeightClustering;
            parameters["numClusters"] = 256;

            reasoning.Add("Weight distribution doesn't strongly favor any technique");
            reasoning.Add("Recommending conservative Weight Clustering as baseline");
            reasoning.Add("Consider trying multiple techniques with CompressionOptimizer");

            result.EstimatedCompressionRatio = NumOps.FromDouble(4.0);
        }

        result.RecommendationReasoning = string.Join("\n", reasoning);
        result.RecommendedParameters = parameters;
    }

    /// <summary>
    /// Generates a detailed analysis report.
    /// </summary>
    /// <param name="result">The analysis result to report on.</param>
    /// <returns>A formatted string containing the analysis report.</returns>
    public string GenerateReport(WeightAnalysisResult<T> result)
    {
        return $@"Model Weight Analysis Report
============================

Weight Statistics:
  Total Weights: {result.TotalWeights:N0}
  Near-Zero Weights: {result.NearZeroWeights:N0} ({NumOps.ToDouble(result.PruningPotential):P1})
  Mean Magnitude: {NumOps.ToDouble(result.MeanMagnitude):F6}
  Std Dev Magnitude: {NumOps.ToDouble(result.StdDevMagnitude):F6}
  Min/Max Magnitude: {NumOps.ToDouble(result.MinMagnitude):F6} / {NumOps.ToDouble(result.MaxMagnitude):F6}

Compression Potential:
  Pruning Potential: {NumOps.ToDouble(result.PruningPotential):P1}
  Clustering Potential: {NumOps.ToDouble(result.ClusteringPotential):P1}
  Entropy: {NumOps.ToDouble(result.Entropy):F2} bits
  Unique Values (est.): {result.UniqueValues:N0}

Recommendation:
  Technique: {result.RecommendedTechnique}
  Estimated Compression: {NumOps.ToDouble(result.EstimatedCompressionRatio):F1}x

Reasoning:
{result.RecommendationReasoning}

Recommended Parameters:
{string.Join("\n", result.RecommendedParameters.Select(p => $"  {p.Key}: {p.Value}"))}
";
    }
}
