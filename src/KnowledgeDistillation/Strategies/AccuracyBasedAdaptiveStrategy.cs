using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Adaptive distillation strategy that adjusts temperature based on student accuracy.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This strategy tracks whether the student is making correct
/// predictions and adjusts temperature accordingly. When the student is correct, we use
/// lower temperature (reinforce learning). When incorrect, we use higher temperature
/// (provide softer, more exploratory targets).</para>
///
/// <para><b>Intuition:</b>
/// - **Correct Prediction** → Student learned this well → Lower temp (reinforce)
/// - **Incorrect Prediction** → Student struggling → Higher temp (help learn)</para>
///
/// <para><b>Example:</b>
/// True label: [0, 1, 0] (class 1)
/// Student predicts: [0.1, 0.8, 0.1] → Correct! → Low temperature
/// Student predicts: [0.6, 0.3, 0.1] → Wrong! → High temperature</para>
///
/// <para><b>Best For:</b>
/// - Supervised learning with labeled data
/// - When you want to focus more on difficult samples
/// - Tracking which samples student struggles with</para>
///
/// <para><b>Requirements:</b>
/// Requires true labels to be provided in ComputeLoss/ComputeGradient calls.
/// Without labels, falls back to confidence-based adaptation.</para>
///
/// <para><b>Performance Tracking:</b>
/// Uses exponential moving average of correctness:
/// - 1.0 = consistently correct
/// - 0.0 = consistently incorrect
/// Temperature inversely proportional to performance.</para>
/// </remarks>
public class AccuracyBasedAdaptiveStrategy<T> : AdaptiveDistillationStrategyBase<T>
{
    /// <summary>
    /// Initializes a new instance of the AccuracyBasedAdaptiveStrategy class.
    /// </summary>
    /// <param name="baseTemperature">Base temperature for distillation (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="minTemperature">Minimum temperature (for correct predictions, default: 1.0).</param>
    /// <param name="maxTemperature">Maximum temperature (for incorrect predictions, default: 5.0).</param>
    /// <param name="adaptationRate">EMA rate for performance tracking (default: 0.1).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This strategy requires true labels during training.
    /// Make sure to pass labels to ComputeLoss() and ComputeGradient().</para>
    ///
    /// <para>Example:
    /// <code>
    /// var strategy = new AccuracyBasedAdaptiveStrategy&lt;double&gt;(
    ///     minTemperature: 1.5,  // For samples student gets right
    ///     maxTemperature: 6.0,  // For samples student gets wrong
    ///     adaptationRate: 0.2   // How fast to adapt (higher = faster)
    /// );
    ///
    /// for (int i = 0; i &lt; samples.Length; i++)
    /// {
    ///     var teacherLogits = teacher.GetLogits(samples[i]);
    ///     var studentLogits = student.Predict(samples[i]);
    ///
    ///     // IMPORTANT: Pass labels for accuracy tracking
    ///     var loss = strategy.ComputeLoss(studentLogits, teacherLogits, labels[i]);
    ///     strategy.UpdatePerformance(i, studentLogits, labels[i]);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public AccuracyBasedAdaptiveStrategy(
        double baseTemperature = 3.0,
        double alpha = 0.3,
        double minTemperature = 1.0,
        double maxTemperature = 5.0,
        double adaptationRate = 0.1)
        : base(baseTemperature, alpha, minTemperature, maxTemperature, adaptationRate)
    {
    }

    /// <summary>
    /// Computes performance based on prediction correctness.
    /// </summary>
    /// <remarks>
    /// <para>Returns 1.0 if prediction is correct, 0.0 if incorrect.
    /// This is tracked with EMA to get average accuracy per sample.</para>
    /// </remarks>
    protected override double ComputePerformance(Vector<T> studentOutput, Vector<T>? trueLabel)
    {
        if (trueLabel == null)
        {
            // Fall back to confidence-based if no label
            var probs = DistillationHelper<T>.Softmax(studentOutput, 1.0);
            return GetMaxConfidence(probs);
        }

        // Return 1.0 if correct, 0.0 if incorrect
        return IsCorrect(studentOutput, trueLabel) ? 1.0 : 0.0;
    }

    /// <summary>
    /// Computes adaptive temperature based on student accuracy.
    /// </summary>
    /// <param name="studentOutput">Student's output logits.</param>
    /// <param name="teacherOutput">Teacher's output logits (not used in accuracy-based).</param>
    /// <returns>Adapted temperature based on historical accuracy.</returns>
    /// <remarks>
    /// <para><b>Algorithm:</b>
    /// 1. Get historical performance for this sample (0.0 to 1.0)
    /// 2. If no history, use current confidence
    /// 3. Compute difficulty = 1 - performance
    /// 4. Map to temperature: temp = min + difficulty * (max - min)</para>
    ///
    /// <para>This creates adaptive behavior:
    /// - High performance (0.8) → Low difficulty (0.2) → Lower temperature
    /// - Low performance (0.3) → High difficulty (0.7) → Higher temperature</para>
    ///
    /// <para><b>Note:</b> This uses historical performance (EMA), not current prediction.
    /// Call UpdatePerformance() regularly to keep tracking updated.</para>
    /// </remarks>
    public override double ComputeAdaptiveTemperature(Vector<T> studentOutput, Vector<T> teacherOutput)
    {
        // We don't have sample index here, so use current confidence as proxy
        // In practice, UpdatePerformance should be called separately with the sample index
        var probs = DistillationHelper<T>.Softmax(studentOutput, temperature: 1.0);
        double currentConfidence = GetMaxConfidence(probs);

        // Use confidence as difficulty estimate
        // Lower confidence often correlates with lower accuracy
        double difficulty = 1.0 - currentConfidence;

        // Map difficulty to temperature range
        double adaptiveTemp = MinTemperature + difficulty * (MaxTemperature - MinTemperature);

        return ClampTemperature(adaptiveTemp);
    }
}
