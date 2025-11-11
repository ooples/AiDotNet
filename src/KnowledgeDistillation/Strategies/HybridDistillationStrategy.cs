using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Hybrid distillation strategy that combines multiple distillation strategies with configurable weights.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Production Use:</b> This strategy allows you to combine multiple distillation approaches
/// (response-based, feature-based, attention-based, etc.) in a single training run. Each strategy
/// contributes to the total loss based on its configured weight.</para>
///
/// <para><b>Example Use Case:</b>
/// For transformer distillation, combine:
/// - 40% Response-based (output matching)
/// - 30% Attention-based (attention pattern matching)
/// - 30% Feature-based (intermediate layer matching)
/// This gives you comprehensive knowledge transfer at multiple levels.</para>
///
/// <para><b>Benefits:</b>
/// - Leverages multiple knowledge transfer mechanisms simultaneously
/// - Weights can be tuned based on validation performance
/// - More robust than single-strategy distillation
/// - Commonly used in SOTA models like TinyBERT, MobileBERT</para>
/// </remarks>
public class HybridDistillationStrategy<T> : DistillationStrategyBase<Vector<T>, T>
{
    private readonly (IDistillationStrategy<Vector<T>, T> Strategy, double Weight)[] _strategies;

    /// <summary>
    /// Initializes a new instance of the HybridDistillationStrategy class.
    /// </summary>
    /// <param name="strategies">Array of (strategy, weight) tuples. Weights should sum to 1.0.</param>
    /// <param name="temperature">Temperature for strategies that don't specify their own.</param>
    /// <param name="alpha">Alpha for strategies that don't specify their own.</param>
    /// <exception cref="ArgumentException">Thrown if weights don't sum to approximately 1.0 or strategies is empty.</exception>
    /// <remarks>
    /// <para>Example:
    /// <code>
    /// var hybrid = new HybridDistillationStrategy&lt;double&gt;(
    ///     new[] {
    ///         (new DistillationLoss&lt;double&gt;(3.0, 0.3), 0.4),        // 40% response
    ///         (new AttentionDistillationStrategy&lt;double&gt;(...), 0.3),  // 30% attention
    ///         (new FeatureDistillationStrategy&lt;double&gt;(...), 0.3)     // 30% features
    ///     }
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public HybridDistillationStrategy(
        (IDistillationStrategy<Vector<T>, T> Strategy, double Weight)[] strategies,
        double temperature = 3.0,
        double alpha = 0.3)
        : base(temperature, alpha)
    {
        if (strategies == null || strategies.Length == 0)
            throw new ArgumentException("At least one strategy must be provided", nameof(strategies));

        double weightSum = strategies.Sum(s => s.Weight);
        if (Math.Abs(weightSum - 1.0) > 1e-6)
            throw new ArgumentException($"Strategy weights must sum to 1.0, got {weightSum}", nameof(strategies));

        foreach (var (_, weight) in strategies)
        {
            if (weight < 0 || weight > 1)
                throw new ArgumentException($"Weights must be between 0 and 1, got {weight}", nameof(strategies));
        }

        _strategies = strategies;
    }

    /// <summary>
    /// Computes combined loss from all strategies.
    /// </summary>
    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        T totalLoss = NumOps.Zero;

        foreach (var (strategy, weight) in _strategies)
        {
            T strategyLoss = strategy.ComputeLoss(studentOutput, teacherOutput, trueLabels);
            T weightedLoss = NumOps.Multiply(strategyLoss, NumOps.FromDouble(weight));
            totalLoss = NumOps.Add(totalLoss, weightedLoss);
        }

        return totalLoss;
    }

    /// <summary>
    /// Computes combined gradient from all strategies.
    /// </summary>
    public override Vector<T> ComputeGradient(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        int n = studentOutput.Length;
        var totalGradient = new Vector<T>(n);

        // Initialize to zero
        for (int i = 0; i < n; i++)
        {
            totalGradient[i] = NumOps.Zero;
        }

        foreach (var (strategy, weight) in _strategies)
        {
            var strategyGradient = strategy.ComputeGradient(studentOutput, teacherOutput, trueLabels);

            for (int i = 0; i < n; i++)
            {
                T weightedGrad = NumOps.Multiply(strategyGradient[i], NumOps.FromDouble(weight));
                totalGradient[i] = NumOps.Add(totalGradient[i], weightedGrad);
            }
        }

        return totalGradient;
    }

    /// <summary>
    /// Gets the individual strategies and their weights.
    /// </summary>
    public (IDistillationStrategy<Vector<T>, T> Strategy, double Weight)[] GetStrategies() => _strategies;
}
