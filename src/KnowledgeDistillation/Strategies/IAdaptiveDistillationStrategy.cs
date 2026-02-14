using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Interface for adaptive distillation strategies that adjust temperature based on student performance.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Adaptive strategies dynamically adjust the temperature parameter
/// during training based on how well the student is learning. This allows for more flexible
/// knowledge transfer compared to fixed-temperature distillation.</para>
///
/// <para><b>Key Concepts:</b>
/// - **Performance Tracking**: Monitor student learning progress
/// - **Temperature Adaptation**: Adjust temperature based on sample difficulty or student confidence
/// - **Per-Sample Adjustment**: Different temperatures for different training samples</para>
///
/// <para><b>When to Use:</b>
/// - Training data has varying difficulty levels
/// - Student performance is uneven across samples
/// - You want automatic temperature tuning instead of manual selection</para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("AdaptiveDistillationStrategy")]
public interface IAdaptiveDistillationStrategy<T>
{
    /// <summary>
    /// Gets the minimum temperature value used for adaptation.
    /// </summary>
    /// <remarks>
    /// <para>Lower temperatures produce sharper probability distributions,
    /// making the distillation more challenging for the student.</para>
    /// </remarks>
    double MinTemperature { get; }

    /// <summary>
    /// Gets the maximum temperature value used for adaptation.
    /// </summary>
    /// <remarks>
    /// <para>Higher temperatures produce softer probability distributions,
    /// making the distillation easier for the student.</para>
    /// </remarks>
    double MaxTemperature { get; }

    /// <summary>
    /// Gets the adaptation rate for performance tracking.
    /// </summary>
    /// <remarks>
    /// <para>Controls how quickly the strategy adapts to new performance metrics.
    /// Lower values = slower adaptation, Higher values = faster adaptation.</para>
    /// </remarks>
    double AdaptationRate { get; }

    /// <summary>
    /// Updates the performance metric for a specific training sample.
    /// </summary>
    /// <param name="sampleIndex">Index of the training sample.</param>
    /// <param name="studentOutput">Student's output (logits) for this sample.</param>
    /// <param name="trueLabel">Optional true label for accuracy-based tracking.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this method after each training step to update
    /// the strategy's understanding of how well the student is performing on each sample.</para>
    ///
    /// <para>The strategy uses this information to adjust temperature in future iterations.</para>
    /// </remarks>
    void UpdatePerformance(int sampleIndex, Vector<T> studentOutput, Vector<T>? trueLabel = null);

    /// <summary>
    /// Computes the adaptive temperature for a specific sample.
    /// </summary>
    /// <param name="studentOutput">Student's current output (logits).</param>
    /// <param name="teacherOutput">Teacher's output (logits).</param>
    /// <returns>Adapted temperature value for this sample.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This method should analyze the student's output
    /// and return an appropriate temperature within [MinTemperature, MaxTemperature].</para>
    ///
    /// <para>Common approaches:
    /// - High confidence → lower temperature (student is doing well)
    /// - Low confidence → higher temperature (student needs help)
    /// - High entropy → lower temperature (focus learning)
    /// - Low entropy → higher temperature (explore more)</para>
    /// </remarks>
    double ComputeAdaptiveTemperature(Vector<T> studentOutput, Vector<T> teacherOutput);

    /// <summary>
    /// Gets the current performance metric for a specific sample.
    /// </summary>
    /// <param name="sampleIndex">Index of the sample.</param>
    /// <returns>Performance metric (interpretation depends on strategy).</returns>
    /// <remarks>
    /// <para>Returns the tracked performance for a sample, or 0.5 (neutral) if not yet tracked.</para>
    /// </remarks>
    double GetPerformance(int sampleIndex);
}
