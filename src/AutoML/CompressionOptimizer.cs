using System.Text;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.AutoML;

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
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();
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
                if (trial.Metrics.MeetsQualityThreshold(_options.MaxAccuracyLoss * 100, _options.MinCompressionRatio)
                    && (_bestTrial == null || NumOps.ToDouble(trial.FitnessScore) > NumOps.ToDouble(_bestFitness)))
                {
                    _bestTrial = trial;
                    _bestFitness = trial.FitnessScore;
                }

                _trialHistory.Add(trial);
            }
            catch (InvalidOperationException ex)
            {
                trial.Success = false;
                trial.ErrorMessage = ex.Message;
                _trialHistory.Add(trial);
            }
            catch (ArgumentException ex)
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

        var summary = new StringBuilder();
        summary.AppendLine("Compression Optimization Summary");
        summary.AppendLine("================================");
        summary.AppendLine($"Trials: {_trialHistory.Count} ({successful} successful, {failed} failed)");
        summary.AppendLine();

        if (_bestTrial != null && _bestTrial.Metrics != null)
        {
            summary.AppendLine("Best Configuration:");
            summary.AppendLine($"  Technique: {_bestTrial.Technique}");
            summary.AppendLine($"  Fitness Score: {NumOps.ToDouble(_bestTrial.FitnessScore):F4}");
            summary.AppendLine($"  Compression Ratio: {NumOps.ToDouble(_bestTrial.Metrics.CompressionRatio):F2}x");
            summary.AppendLine($"  Accuracy Loss: {NumOps.ToDouble(_bestTrial.Metrics.AccuracyLoss) * 100:F2}%");
            summary.AppendLine($"  Size Reduction: {NumOps.ToDouble(_bestTrial.Metrics.SizeReductionPercentage):F2}%");
            summary.AppendLine();
            summary.AppendLine("Hyperparameters:");
            foreach (var param in _bestTrial.Hyperparameters)
            {
                summary.AppendLine($"  {param.Key}: {param.Value}");
            }
        }
        else
        {
            summary.AppendLine("No successful trials completed.");
        }

        return summary.ToString();
    }
}
