using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using AiDotNet.Models;

namespace AiDotNet.AutoML;

/// <summary>
/// Represents a compression configuration to be evaluated.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CompressionTrial<T>
{
    /// <summary>
    /// Gets or sets the compression technique used.
    /// </summary>
    public CompressionType Technique { get; set; }

    /// <summary>
    /// Gets or sets the hyperparameters for the compression technique.
    /// </summary>
    public Dictionary<string, object> Hyperparameters { get; set; } = new();

    /// <summary>
    /// Gets or sets the resulting compression metrics.
    /// </summary>
    public CompressionMetrics<T>? Metrics { get; set; }

    /// <summary>
    /// Gets or sets the fitness score for this trial.
    /// </summary>
    public T FitnessScore { get; set; } = default!;

    /// <summary>
    /// Gets or sets whether this trial completed successfully.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets any error message from the trial.
    /// </summary>
    public string? ErrorMessage { get; set; }
}

/// <summary>
/// Configuration options for the compression optimizer.
/// </summary>
public class CompressionOptimizerOptions
{
    /// <summary>
    /// Gets or sets the maximum number of trials to run (default: 20).
    /// </summary>
    public int MaxTrials { get; set; } = 20;

    /// <summary>
    /// Gets or sets the maximum acceptable accuracy loss as a fraction (default: 0.02 = 2%).
    /// </summary>
    public double MaxAccuracyLoss { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the minimum acceptable compression ratio (default: 2.0).
    /// </summary>
    public double MinCompressionRatio { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the weight for accuracy in fitness calculation (default: 0.5).
    /// </summary>
    public double AccuracyWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the weight for compression ratio in fitness calculation (default: 0.3).
    /// </summary>
    public double CompressionWeight { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the weight for inference speed in fitness calculation (default: 0.2).
    /// </summary>
    public double SpeedWeight { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets whether to include pruning techniques (default: true).
    /// </summary>
    public bool IncludePruning { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include quantization techniques (default: true).
    /// </summary>
    public bool IncludeQuantization { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include encoding techniques (default: true).
    /// </summary>
    public bool IncludeEncoding { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include hybrid techniques like Deep Compression (default: true).
    /// </summary>
    public bool IncludeHybrid { get; set; } = true;

    /// <summary>
    /// Gets or sets the random seed for reproducibility (default: null for random).
    /// </summary>
    public int? RandomSeed { get; set; }
}

/// <summary>
/// Automatically finds the best compression configuration for a model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// CompressionOptimizer uses automated machine learning techniques to find the optimal
/// compression configuration for a neural network model. It evaluates different compression
/// techniques and hyperparameters, tracking metrics like compression ratio, accuracy loss,
/// and inference speed.
/// </para>
/// <para><b>For Beginners:</b> Think of this as an automated assistant that tries different
/// ways to compress your model and finds the best one for your needs.
///
/// Instead of manually trying:
/// - Different pruning levels (50%, 70%, 90% of weights removed)
/// - Different quantization settings (8-bit, 5-bit, etc.)
/// - Different compression techniques (pruning, clustering, Huffman)
///
/// The optimizer automatically:
/// 1. Generates compression configurations to try
/// 2. Applies each configuration and measures results
/// 3. Tracks which configurations work best
/// 4. Returns the best compression settings found
///
/// Example usage:
/// <code>
/// var optimizer = new CompressionOptimizer&lt;double&gt;(options);
/// var bestCompression = await optimizer.OptimizeAsync(modelWeights, evaluator);
/// Console.WriteLine($"Best technique: {bestCompression.Technique}");
/// Console.WriteLine($"Compression ratio: {bestCompression.Metrics.CompressionRatio}x");
/// </code>
/// </para>
/// </remarks>
public class CompressionOptimizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly CompressionOptimizerOptions _options;
    private readonly Random _random;
    private readonly List<CompressionTrial<T>> _trialHistory;

    private CompressionTrial<T>? _bestTrial;
    private T _bestFitness;

    /// <summary>
    /// Initializes a new instance of the CompressionOptimizer class.
    /// </summary>
    /// <param name="options">Configuration options for the optimizer.</param>
    public CompressionOptimizer(CompressionOptimizerOptions? options = null)
    {
        _options = options ?? new CompressionOptimizerOptions();
        _random = _options.RandomSeed.HasValue
            ? new Random(_options.RandomSeed.Value)
            : new Random();
        _trialHistory = new List<CompressionTrial<T>>();
        _bestFitness = NumOps.FromDouble(double.NegativeInfinity);
    }

    /// <summary>
    /// Gets the history of all compression trials.
    /// </summary>
    public IReadOnlyList<CompressionTrial<T>> TrialHistory => _trialHistory.AsReadOnly();

    /// <summary>
    /// Gets the best compression trial found so far.
    /// </summary>
    public CompressionTrial<T>? BestTrial => _bestTrial;

    /// <summary>
    /// Runs the compression optimization process.
    /// </summary>
    /// <param name="weights">The model weights to compress.</param>
    /// <param name="evaluateAccuracy">A function that evaluates model accuracy given compressed weights.</param>
    /// <returns>The best compression trial found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method tries different compression settings and returns the best one.
    ///
    /// The evaluateAccuracy function should:
    /// 1. Take compressed weights
    /// 2. Apply them to your model
    /// 3. Run inference on validation data
    /// 4. Return the accuracy (0.0 to 1.0)
    ///
    /// The optimizer will call this function many times with different compression settings.
    /// </para>
    /// </remarks>
    public CompressionTrial<T> Optimize(
        Vector<T> weights,
        Func<Vector<T>, T> evaluateAccuracy)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        if (evaluateAccuracy == null)
        {
            throw new ArgumentNullException(nameof(evaluateAccuracy));
        }

        // Calculate original accuracy
        var originalAccuracy = evaluateAccuracy(weights);
        var originalSize = weights.Length * GetElementSize();

        // Generate and evaluate trials
        var trials = GenerateTrials();

        foreach (var trial in trials.Take(_options.MaxTrials))
        {
            try
            {
                var compressor = CreateCompressor(trial.Technique, trial.Hyperparameters);

                // Time compression
                var compressionStart = DateTime.UtcNow;
                var (compressedWeights, metadata) = compressor.Compress(weights);
                var compressionTime = (DateTime.UtcNow - compressionStart).TotalMilliseconds;

                // Time decompression
                var decompressionStart = DateTime.UtcNow;
                var decompressedWeights = compressor.Decompress(compressedWeights, metadata);
                var decompressionTime = (DateTime.UtcNow - decompressionStart).TotalMilliseconds;

                // Evaluate accuracy on decompressed weights
                var compressedAccuracy = evaluateAccuracy(decompressedWeights);

                // Calculate compression metrics
                var compressedSize = compressor.GetCompressedSize(compressedWeights, metadata);

                trial.Metrics = new CompressionMetrics<T>
                {
                    OriginalSize = originalSize,
                    CompressedSize = compressedSize,
                    OriginalAccuracy = originalAccuracy,
                    CompressedAccuracy = compressedAccuracy,
                    CompressionTimeMs = NumOps.FromDouble(compressionTime),
                    DecompressionTimeMs = NumOps.FromDouble(decompressionTime),
                    CompressionTechnique = trial.Technique.ToString()
                };
                trial.Metrics.CalculateDerivedMetrics();

                // Calculate fitness
                trial.FitnessScore = trial.Metrics.CalculateCompositeFitness(
                    _options.AccuracyWeight,
                    _options.CompressionWeight,
                    _options.SpeedWeight);

                trial.Success = true;

                // Check if this is the best trial that meets quality thresholds
                if (trial.Metrics.MeetsQualityThreshold(_options.MaxAccuracyLoss * 100, _options.MinCompressionRatio))
                {
                    if (_bestTrial == null || NumOps.ToDouble(trial.FitnessScore) > NumOps.ToDouble(_bestFitness))
                    {
                        _bestTrial = trial;
                        _bestFitness = trial.FitnessScore;
                    }
                }

                _trialHistory.Add(trial);
            }
            catch (Exception ex)
            {
                trial.Success = false;
                trial.ErrorMessage = ex.Message;
                _trialHistory.Add(trial);
            }
        }

        // If no trial met quality thresholds, return the best one anyway
        if (_bestTrial == null && _trialHistory.Any(t => t.Success))
        {
            _bestTrial = _trialHistory
                .Where(t => t.Success)
                .OrderByDescending(t => NumOps.ToDouble(t.FitnessScore))
                .First();
            _bestFitness = _bestTrial.FitnessScore;
        }

        return _bestTrial ?? throw new InvalidOperationException("No successful compression trials completed.");
    }

    /// <summary>
    /// Generates a list of compression trials to evaluate.
    /// </summary>
    private IEnumerable<CompressionTrial<T>> GenerateTrials()
    {
        var trials = new List<CompressionTrial<T>>();

        // Pruning trials
        if (_options.IncludePruning)
        {
            foreach (var sparsity in new[] { 0.5, 0.7, 0.8, 0.9, 0.95 })
            {
                trials.Add(new CompressionTrial<T>
                {
                    Technique = CompressionType.SparsePruning,
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["sparsityTarget"] = sparsity
                    }
                });
            }
        }

        // Quantization trials (weight clustering)
        if (_options.IncludeQuantization)
        {
            foreach (var numClusters in new[] { 16, 32, 64, 128, 256 })
            {
                trials.Add(new CompressionTrial<T>
                {
                    Technique = CompressionType.WeightClustering,
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["numClusters"] = numClusters
                    }
                });
            }
        }

        // Encoding trials
        if (_options.IncludeEncoding)
        {
            foreach (var precision in new[] { 4, 6, 8 })
            {
                trials.Add(new CompressionTrial<T>
                {
                    Technique = CompressionType.HuffmanEncoding,
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["precision"] = precision
                    }
                });
            }
        }

        // Hybrid/Deep Compression trials
        if (_options.IncludeHybrid)
        {
            foreach (var sparsity in new[] { 0.7, 0.85, 0.92 })
            {
                foreach (var numClusters in new[] { 32, 64, 128 })
                {
                    trials.Add(new CompressionTrial<T>
                    {
                        Technique = CompressionType.HybridHuffmanClustering,
                        Hyperparameters = new Dictionary<string, object>
                        {
                            ["pruningSparsity"] = sparsity,
                            ["numClusters"] = numClusters
                        }
                    });
                }
            }
        }

        // Shuffle trials for random exploration
        return trials.OrderBy(_ => _random.Next());
    }

    /// <summary>
    /// Creates a compression algorithm instance based on the technique and hyperparameters.
    /// </summary>
    private IModelCompressionStrategy<T> CreateCompressor(
        CompressionType technique,
        Dictionary<string, object> hyperparameters)
    {
        return technique switch
        {
            CompressionType.SparsePruning => new SparsePruningCompression<T>(
                sparsityTarget: Convert.ToDouble(hyperparameters.GetValueOrDefault("sparsityTarget", 0.9))),

            CompressionType.WeightClustering => new WeightClusteringCompression<T>(
                numClusters: Convert.ToInt32(hyperparameters.GetValueOrDefault("numClusters", 32)),
                randomSeed: _options.RandomSeed),

            CompressionType.HuffmanEncoding => new HuffmanEncodingCompression<T>(
                precision: Convert.ToInt32(hyperparameters.GetValueOrDefault("precision", 6))),

            CompressionType.HybridHuffmanClustering => new DeepCompression<T>(
                pruningSparsity: Convert.ToDouble(hyperparameters.GetValueOrDefault("pruningSparsity", 0.9)),
                numClusters: Convert.ToInt32(hyperparameters.GetValueOrDefault("numClusters", 32)),
                randomSeed: _options.RandomSeed),

            _ => throw new NotSupportedException($"Compression type {technique} is not supported.")
        };
    }

    /// <summary>
    /// Gets the size in bytes of one element of type T.
    /// </summary>
    private static long GetElementSize()
    {
        if (typeof(T) == typeof(float)) return 4;
        if (typeof(T) == typeof(double)) return 8;
        if (typeof(T) == typeof(Half)) return 2;
        if (typeof(T) == typeof(decimal)) return 16;
        return 8; // Default assumption
    }

    /// <summary>
    /// Gets a summary of the optimization results.
    /// </summary>
    /// <returns>A formatted string containing optimization results.</returns>
    public string GetSummary()
    {
        var successful = _trialHistory.Count(t => t.Success);
        var failed = _trialHistory.Count - successful;

        var summary = $@"Compression Optimization Summary
================================
Trials: {_trialHistory.Count} ({successful} successful, {failed} failed)

";

        if (_bestTrial != null && _bestTrial.Metrics != null)
        {
            summary += $@"Best Configuration:
  Technique: {_bestTrial.Technique}
  Fitness Score: {NumOps.ToDouble(_bestTrial.FitnessScore):F4}
  Compression Ratio: {NumOps.ToDouble(_bestTrial.Metrics.CompressionRatio):F2}x
  Accuracy Loss: {NumOps.ToDouble(_bestTrial.Metrics.AccuracyLoss) * 100:F2}%
  Size Reduction: {NumOps.ToDouble(_bestTrial.Metrics.SizeReductionPercentage):F2}%

Hyperparameters:
";
            foreach (var param in _bestTrial.Hyperparameters)
            {
                summary += $"  {param.Key}: {param.Value}\n";
            }
        }
        else
        {
            summary += "No successful trials completed.\n";
        }

        return summary;
    }
}
