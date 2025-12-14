using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Adaptive distillation strategy that adjusts temperature based on student confidence.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This strategy adapts temperature based on how confident
/// the student is in its predictions. When the student is confident (high max probability),
/// we use lower temperature (harder distillation). When uncertain (low max probability),
/// we use higher temperature (easier distillation with softer targets).</para>
///
/// <para><b>Intuition:</b>
/// - **High Confidence** → Student understands this sample → Lower temp (sharpen targets)
/// - **Low Confidence** → Student struggles with this sample → Higher temp (soften targets)</para>
///
/// <para><b>Example:</b>
/// Student predicts [0.95, 0.03, 0.02] → High confidence (0.95) → Low temperature
/// Student predicts [0.40, 0.35, 0.25] → Low confidence (0.40) → High temperature</para>
///
/// <para><b>Best For:</b>
/// - General-purpose adaptive distillation
/// - When you want automatic difficulty adjustment
/// - Datasets with varying sample complexity</para>
///
/// <para><b>Temperature Mapping:</b>
/// Confidence = max(probabilities)
/// Difficulty = 1 - Confidence
/// Temperature = MinTemp + Difficulty * (MaxTemp - MinTemp)</para>
/// </remarks>
public class ConfidenceBasedAdaptiveStrategy<T> : AdaptiveDistillationStrategyBase<T>
{
    /// <summary>
    /// Initializes a new instance of the ConfidenceBasedAdaptiveStrategy class.
    /// </summary>
    /// <param name="baseTemperature">Base temperature for distillation (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="minTemperature">Minimum temperature (for high confidence samples, default: 1.0).</param>
    /// <param name="maxTemperature">Maximum temperature (for low confidence samples, default: 5.0).</param>
    /// <param name="adaptationRate">EMA rate for performance tracking (default: 0.1).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The default parameters work well for most cases.
    /// Adjust minTemperature and maxTemperature to control the adaptation range.</para>
    ///
    /// <para>Example:
    /// <code>
    /// // Conservative adaptation (narrow temperature range)
    /// var strategy1 = new ConfidenceBasedAdaptiveStrategy&lt;double&gt;(
    ///     minTemperature: 2.0,
    ///     maxTemperature: 4.0
    /// );
    ///
    /// // Aggressive adaptation (wide temperature range)
    /// var strategy2 = new ConfidenceBasedAdaptiveStrategy&lt;double&gt;(
    ///     minTemperature: 1.0,
    ///     maxTemperature: 8.0
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public ConfidenceBasedAdaptiveStrategy(
        double baseTemperature = 3.0,
        double alpha = 0.3,
        double minTemperature = 1.0,
        double maxTemperature = 5.0,
        double adaptationRate = 0.1)
        : base(baseTemperature, alpha, minTemperature, maxTemperature, adaptationRate)
    {
    }

    /// <summary>
    /// Computes adaptive temperature based on student confidence.
    /// </summary>
    /// <param name="studentOutput">Student's output logits.</param>
    /// <param name="teacherOutput">Teacher's output logits (not used in confidence-based).</param>
    /// <returns>Adapted temperature based on student confidence.</returns>
    /// <remarks>
    /// <para><b>Algorithm:</b>
    /// 1. Convert logits to probabilities (softmax)
    /// 2. Find maximum probability (confidence)
    /// 3. Compute difficulty = 1 - confidence
    /// 4. Map to temperature range: temp = min + difficulty * (max - min)</para>
    ///
    /// <para>This creates an inverse relationship:
    /// - High confidence (0.9) → Low difficulty (0.1) → Low temperature (~1.4 with defaults)
    /// - Low confidence (0.4) → High difficulty (0.6) → High temperature (~3.4 with defaults)</para>
    /// </remarks>
    public override double ComputeAdaptiveTemperature(Vector<T> studentOutput, Vector<T> teacherOutput)
    {
        // Convert to probabilities
        var probs = DistillationHelper<T>.Softmax(studentOutput, temperature: 1.0);

        // Get maximum probability (student's confidence)
        double confidence = GetMaxConfidence(probs);

        // Lower confidence = harder sample = higher temperature needed
        // Higher confidence = easier sample = lower temperature needed
        double difficulty = 1.0 - confidence;

        // Map difficulty [0, 1] to temperature [MinTemperature, MaxTemperature]
        double adaptiveTemp = MinTemperature + difficulty * (MaxTemperature - MinTemperature);

        return ClampTemperature(adaptiveTemp);
    }
}
