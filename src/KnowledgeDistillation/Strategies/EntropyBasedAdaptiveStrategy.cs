using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Adaptive distillation strategy that adjusts temperature based on prediction entropy.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Entropy measures how uncertain or "spread out" a probability
/// distribution is. High entropy means the student is uncertain (probabilities are similar
/// across classes). Low entropy means the student is certain (one class has high probability).</para>
///
/// <para><b>Entropy Examples:</b>
/// - **Low Entropy** [0.95, 0.03, 0.02]: Student is certain → Class 0 dominates
/// - **High Entropy** [0.35, 0.33, 0.32]: Student is uncertain → All classes similar</para>
///
/// <para><b>Intuition:</b>
/// - **High Entropy** (uncertain) → Student struggling → Lower temp (focus learning)
/// - **Low Entropy** (certain) → Student confident → Higher temp (explore more)</para>
///
/// <para><b>Why Lower Temp for High Entropy?</b>
/// When student is uncertain, we want to provide sharper (lower temp) targets to focus
/// learning on the most important features, rather than soft targets that might reinforce
/// uncertainty.</para>
///
/// <para><b>Best For:</b>
/// - Detecting student uncertainty
/// - Calibrating overconfident students
/// - Datasets where uncertainty patterns are meaningful</para>
///
/// <para><b>Entropy Range:</b>
/// - Minimum: 0.0 (completely certain, one class = 1.0)
/// - Maximum: 1.0 (normalized, completely uncertain, uniform distribution)
/// - Normalized by log(num_classes) to get [0, 1] range</para>
///
/// <para><b>Temperature Mapping:</b>
/// High entropy → high difficulty → lower temperature (sharpen)
/// Low entropy → low difficulty → higher temperature (soften)</para>
/// </remarks>
public class EntropyBasedAdaptiveStrategy<T> : AdaptiveDistillationStrategyBase<T>
{
    /// <summary>
    /// Initializes a new instance of the EntropyBasedAdaptiveStrategy class.
    /// </summary>
    /// <param name="baseTemperature">Base temperature for distillation (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="minTemperature">Minimum temperature (for high entropy/uncertain, default: 1.0).</param>
    /// <param name="maxTemperature">Maximum temperature (for low entropy/certain, default: 5.0).</param>
    /// <param name="adaptationRate">EMA rate for performance tracking (default: 0.1).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This strategy automatically adapts based on how
    /// uncertain the student's predictions are. No labels required!</para>
    ///
    /// <para>Example:
    /// <code>
    /// var strategy = new EntropyBasedAdaptiveStrategy&lt;double&gt;(
    ///     minTemperature: 1.5,  // For uncertain predictions (high entropy)
    ///     maxTemperature: 4.0,  // For confident predictions (low entropy)
    ///     adaptationRate: 0.15  // Moderate adaptation speed
    /// );
    ///
    /// for (int i = 0; i &lt; samples.Length; i++)
    /// {
    ///     var teacherLogits = teacher.GetLogits(samples[i]);
    ///     var studentLogits = student.Predict(samples[i]);
    ///
    ///     // Automatically adapts based on entropy
    ///     var loss = strategy.ComputeLoss(studentLogits, teacherLogits);
    ///     strategy.UpdatePerformance(i, studentLogits);
    /// }
    /// </code>
    /// </para>
    ///
    /// <para><b>Comparison with Confidence-Based:</b>
    /// - **Confidence**: max(probabilities) - focuses on highest class
    /// - **Entropy**: considers full distribution - more holistic uncertainty measure</para>
    /// </remarks>
    public EntropyBasedAdaptiveStrategy(
        double baseTemperature = 3.0,
        double alpha = 0.3,
        double minTemperature = 1.0,
        double maxTemperature = 5.0,
        double adaptationRate = 0.1)
        : base(baseTemperature, alpha, minTemperature, maxTemperature, adaptationRate)
    {
    }

    /// <summary>
    /// Computes performance based on entropy (inverse relationship).
    /// </summary>
    /// <remarks>
    /// <para>Returns 1 - entropy. Low entropy (certain) = high performance score.
    /// High entropy (uncertain) = low performance score.</para>
    /// </remarks>
    protected override double ComputePerformance(Vector<T> studentOutput, Vector<T>? trueLabel)
    {
        var probs = DistillationHelper<T>.Softmax(studentOutput, 1.0);
        double entropy = ComputeEntropy(probs);

        // Convert entropy to performance score
        // Low entropy (certain) = high performance
        // High entropy (uncertain) = low performance
        return 1.0 - entropy;
    }

    /// <summary>
    /// Computes adaptive temperature based on prediction entropy.
    /// </summary>
    /// <param name="studentOutput">Student's output logits.</param>
    /// <param name="teacherOutput">Teacher's output logits (not used in entropy-based).</param>
    /// <returns>Adapted temperature based on student entropy.</returns>
    /// <remarks>
    /// <para><b>Algorithm:</b>
    /// 1. Convert logits to probabilities (softmax)
    /// 2. Compute normalized entropy H = -Σ(p * log(p)) / log(n)
    /// 3. Map entropy to temperature (inverted):
    ///    - High entropy → Low temp (sharpen to reduce uncertainty)
    ///    - Low entropy → High temp (soften to explore more)</para>
    ///
    /// <para><b>Examples with 3 classes:</b>
    /// - [0.95, 0.03, 0.02]: Entropy ≈ 0.2 (low) → Higher temperature
    /// - [0.35, 0.33, 0.32]: Entropy ≈ 1.0 (high) → Lower temperature</para>
    ///
    /// <para><b>Why Invert?</b>
    /// High uncertainty needs focused (sharp) targets to learn clear boundaries.
    /// Low uncertainty can benefit from softer targets to learn class relationships.</para>
    /// </remarks>
    public override double ComputeAdaptiveTemperature(Vector<T> studentOutput, Vector<T> teacherOutput)
    {
        // Convert to probabilities
        var probs = DistillationHelper<T>.Softmax(studentOutput, temperature: 1.0);

        // Compute normalized entropy [0, 1]
        double entropy = ComputeEntropy(probs);

        // INVERTED MAPPING: Higher entropy = harder = LOWER temperature
        // This helps focus uncertain predictions
        // difficulty = entropy (high entropy = high difficulty)
        double difficulty = entropy;

        // Invert: low temp for high entropy, high temp for low entropy
        double adaptiveTemp = MaxTemperature - difficulty * (MaxTemperature - MinTemperature);

        return ClampTemperature(adaptiveTemp);
    }
}
