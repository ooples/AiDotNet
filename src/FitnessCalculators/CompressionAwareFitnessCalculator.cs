using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.ModelCompression;
using AiDotNet.Models;

namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that considers both model accuracy and compression effectiveness.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output predictions.</typeparam>
/// <remarks>
/// <para>
/// This fitness calculator implements multi-objective optimization for model compression.
/// It balances accuracy preservation against compression benefits (size reduction, speed improvement).
/// </para>
/// <para><b>For Beginners:</b> When compressing a model, you have competing goals:
///
/// 1. **Keep accuracy high** - You don't want the compressed model to make worse predictions
/// 2. **Make the model smaller** - Smaller models use less memory and storage
/// 3. **Make inference faster** - Smaller models often run faster
///
/// This fitness calculator combines all three goals into a single score that optimization
/// algorithms (like AutoML or genetic algorithms) can use to find the best compression settings.
///
/// The weights control how much each goal matters:
/// - High accuracyWeight = prioritize keeping predictions accurate
/// - High compressionWeight = prioritize making the model smaller
/// - High speedWeight = prioritize making inference faster
///
/// Example:
/// - For edge devices with limited memory: increase compressionWeight
/// - For real-time applications: increase speedWeight
/// - For medical/financial applications: increase accuracyWeight
/// </para>
/// </remarks>
public class CompressionAwareFitnessCalculator<T, TInput, TOutput> : IFitnessCalculator<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _accuracyWeight;
    private readonly double _compressionWeight;
    private readonly double _speedWeight;
    private readonly IFitnessCalculator<T, TInput, TOutput> _baseFitnessCalculator;
    private CompressionMetrics<T>? _currentCompressionMetrics;

    /// <summary>
    /// Initializes a new instance of the CompressionAwareFitnessCalculator class.
    /// </summary>
    /// <param name="baseFitnessCalculator">The base fitness calculator for accuracy metrics.</param>
    /// <param name="accuracyWeight">Weight for accuracy preservation (default: 0.5).</param>
    /// <param name="compressionWeight">Weight for compression ratio (default: 0.3).</param>
    /// <param name="speedWeight">Weight for inference speedup (default: 0.2).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The base fitness calculator handles accuracy measurement.
    /// This class wraps it to add compression considerations.
    ///
    /// Default weights (0.5, 0.3, 0.2) prioritize accuracy while still rewarding compression.
    /// Adjust based on your deployment constraints:
    /// - Mobile deployment: (0.4, 0.4, 0.2) - balance accuracy and size
    /// - Real-time inference: (0.4, 0.2, 0.4) - prioritize speed
    /// - High-stakes predictions: (0.7, 0.2, 0.1) - prioritize accuracy
    /// </para>
    /// </remarks>
    public CompressionAwareFitnessCalculator(
        IFitnessCalculator<T, TInput, TOutput> baseFitnessCalculator,
        double accuracyWeight = 0.5,
        double compressionWeight = 0.3,
        double speedWeight = 0.2)
    {
        _baseFitnessCalculator = baseFitnessCalculator ?? throw new ArgumentNullException(nameof(baseFitnessCalculator));

        if (accuracyWeight < 0 || compressionWeight < 0 || speedWeight < 0)
        {
            throw new ArgumentException("Weights must be non-negative.");
        }

        var totalWeight = accuracyWeight + compressionWeight + speedWeight;
        if (totalWeight <= 0)
        {
            throw new ArgumentException("At least one weight must be positive.");
        }

        // Normalize weights to sum to 1
        _accuracyWeight = accuracyWeight / totalWeight;
        _compressionWeight = compressionWeight / totalWeight;
        _speedWeight = speedWeight / totalWeight;
    }

    /// <summary>
    /// Gets a value indicating whether higher fitness scores are better.
    /// </summary>
    public bool IsHigherScoreBetter => true;

    /// <summary>
    /// Gets or sets the current compression metrics to use in fitness calculation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before calculating fitness, set this property with the
    /// compression results. The fitness calculation will then consider both model accuracy
    /// and compression effectiveness.
    /// </para>
    /// </remarks>
    public CompressionMetrics<T>? CompressionMetrics
    {
        get => _currentCompressionMetrics;
        set => _currentCompressionMetrics = value;
    }

    /// <summary>
    /// Calculates a composite fitness score considering both accuracy and compression.
    /// </summary>
    /// <param name="evaluationData">Model evaluation data containing accuracy metrics.</param>
    /// <returns>A fitness score where higher values indicate better overall performance.</returns>
    public T CalculateFitnessScore(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        // Get base accuracy fitness (normalize to [0, 1])
        var baseFitness = _baseFitnessCalculator.CalculateFitnessScore(evaluationData);
        var normalizedAccuracy = NormalizeBaseFitness(baseFitness);

        // If no compression metrics, return accuracy-only fitness
        if (_currentCompressionMetrics == null)
        {
            return NumOps.FromDouble(normalizedAccuracy);
        }

        // Calculate compression score (from compression metrics)
        var compressionFitness = _currentCompressionMetrics.CalculateCompositeFitness(
            _accuracyWeight, _compressionWeight, _speedWeight);

        // Weight the accuracy component and compression component
        var accuracyComponent = normalizedAccuracy * _accuracyWeight;
        var compressionComponent = NumOps.ToDouble(compressionFitness) * (1 - _accuracyWeight);

        return NumOps.FromDouble(accuracyComponent + compressionComponent);
    }

    /// <summary>
    /// Calculates a fitness score from dataset statistics.
    /// </summary>
    /// <param name="dataSet">Dataset statistics.</param>
    /// <returns>A fitness score.</returns>
    public T CalculateFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        var baseFitness = _baseFitnessCalculator.CalculateFitnessScore(dataSet);
        var normalizedAccuracy = NormalizeBaseFitness(baseFitness);

        if (_currentCompressionMetrics == null)
        {
            return NumOps.FromDouble(normalizedAccuracy);
        }

        var compressionFitness = _currentCompressionMetrics.CalculateCompositeFitness(
            _accuracyWeight, _compressionWeight, _speedWeight);

        var accuracyComponent = normalizedAccuracy * _accuracyWeight;
        var compressionComponent = NumOps.ToDouble(compressionFitness) * (1 - _accuracyWeight);

        return NumOps.FromDouble(accuracyComponent + compressionComponent);
    }

    /// <summary>
    /// Compares two fitness scores and determines if the current is better than the best.
    /// </summary>
    /// <param name="currentFitness">The current fitness score.</param>
    /// <param name="bestFitness">The best fitness score so far.</param>
    /// <returns>True if current fitness is better than best fitness.</returns>
    public bool IsBetterFitness(T currentFitness, T bestFitness)
    {
        return NumOps.ToDouble(currentFitness) > NumOps.ToDouble(bestFitness);
    }

    /// <summary>
    /// Normalizes the base fitness score to a [0, 1] range.
    /// </summary>
    private double NormalizeBaseFitness(T fitness)
    {
        var fitnessValue = NumOps.ToDouble(fitness);

        // If base calculator prefers higher scores, assume it's already [0, 1] or close
        if (_baseFitnessCalculator.IsHigherScoreBetter)
        {
            return Math.Max(0, Math.Min(1, fitnessValue));
        }
        else
        {
            // For error-based metrics (lower is better), convert to [0, 1] where 1 is best
            // Using exponential decay: 1 - e^(-1/error) for error > 0
            if (fitnessValue <= 0)
            {
                return 1.0; // Perfect score (zero error)
            }
            return Math.Exp(-fitnessValue);
        }
    }
}
