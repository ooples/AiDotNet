using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Curriculum distillation strategy that progresses from easy to hard samples.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Easy-to-hard curriculum learning mimics how humans learn best:
/// start with simple concepts and gradually introduce more complex ones. This strategy
/// filters training samples and adjusts temperature based on difficulty and training progress.</para>
///
/// <para><b>How It Works:</b>
/// 1. **Early Training** (progress 0.0-0.3):
///    - Include only easy samples (difficulty ≤ 0.3)
///    - Use high temperature (soft targets, gentle learning)
/// 2. **Mid Training** (progress 0.3-0.7):
///    - Include easy and medium samples (difficulty ≤ 0.7)
///    - Gradually decrease temperature
/// 3. **Late Training** (progress 0.7-1.0):
///    - Include all samples (even hard ones)
///    - Use low temperature (sharp targets, challenging)</para>
///
/// <para><b>Temperature Progression:</b>
/// Starts at MaxTemperature (e.g., 5.0) and linearly decreases to MinTemperature (e.g., 2.0)
/// as training progresses. This makes distillation progressively more challenging.</para>
///
/// <para><b>Sample Filtering:</b>
/// At progress P, only include samples with difficulty ≤ P.
/// Example: At 50% progress, only samples with difficulty ≤ 0.5 are included.</para>
///
/// <para><b>Real-World Analogy:</b>
/// Learning mathematics: Start with addition (easy), then multiplication (medium),
/// then algebra (hard). Don't try to teach calculus to someone who hasn't learned addition!</para>
///
/// <para><b>Best For:</b>
/// - Training from scratch
/// - Datasets with clear difficulty levels
/// - Preventing student from being overwhelmed early
/// - Improving convergence speed and final performance</para>
///
/// <para><b>References:</b>
/// - Bengio et al. (2009). Curriculum Learning. ICML.
/// - Kumar et al. (2010). Self-paced Learning for Latent Variable Models.</para>
/// </remarks>
public class EasyToHardCurriculumStrategy<T> : CurriculumDistillationStrategyBase<T>
{
    /// <summary>
    /// Initializes a new instance of the EasyToHardCurriculumStrategy class.
    /// </summary>
    /// <param name="baseTemperature">Base temperature for distillation (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="minTemperature">Ending temperature for hard samples (default: 2.0).</param>
    /// <param name="maxTemperature">Starting temperature for easy samples (default: 5.0).</param>
    /// <param name="totalSteps">Total training steps/epochs (default: 100).</param>
    /// <param name="sampleDifficulties">Optional pre-defined difficulty scores (0.0=easy, 1.0=hard).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set totalSteps to your number of training epochs.
    /// Optionally provide sampleDifficulties to control which samples appear when.</para>
    ///
    /// <para>Example:
    /// <code>
    /// // Define sample difficulties (optional, can be computed automatically)
    /// var difficulties = new Dictionary&lt;int, double&gt;
    /// {
    ///     { 0, 0.1 },  // Very easy sample
    ///     { 1, 0.3 },  // Easy sample
    ///     { 2, 0.5 },  // Medium sample
    ///     { 3, 0.8 },  // Hard sample
    ///     { 4, 0.95 }  // Very hard sample
    /// };
    ///
    /// var strategy = new EasyToHardCurriculumStrategy&lt;double&gt;(
    ///     minTemperature: 2.0,   // Final temperature (hard phase)
    ///     maxTemperature: 5.0,   // Initial temperature (easy phase)
    ///     totalSteps: 100,       // 100 epochs
    ///     sampleDifficulties: difficulties
    /// );
    ///
    /// // Training loop
    /// for (int epoch = 0; epoch &lt; 100; epoch++)
    /// {
    ///     strategy.UpdateProgress(epoch);
    ///
    ///     foreach (var (sample, index) in trainingSamples.WithIndex())
    ///     {
    ///         // Filter samples by curriculum
    ///         if (!strategy.ShouldIncludeSample(index))
    ///             continue; // Too hard for current stage
    ///
    ///         // Train on this sample...
    ///     }
    /// }
    /// </code>
    /// </para>
    ///
    /// <para><b>Automatic Difficulty Scoring:</b>
    /// If you don't provide difficulties, you can estimate them:
    /// - Teacher confidence (lower = harder)
    /// - Validation loss (higher = harder)
    /// - Number of similar samples (fewer = harder, rare edge case)
    /// - Expert annotation</para>
    /// </remarks>
    public EasyToHardCurriculumStrategy(
        double baseTemperature = 3.0,
        double alpha = 0.3,
        double minTemperature = 2.0,
        double maxTemperature = 5.0,
        int totalSteps = 100,
        Dictionary<int, double>? sampleDifficulties = null)
        : base(baseTemperature, alpha, minTemperature, maxTemperature, totalSteps, sampleDifficulties)
    {
    }

    /// <summary>
    /// Determines if a sample should be included based on curriculum progress.
    /// </summary>
    /// <param name="sampleIndex">Index of the sample.</param>
    /// <returns>True if sample difficulty is within current curriculum stage.</returns>
    /// <remarks>
    /// <para><b>Algorithm:</b>
    /// - Get sample difficulty (or assume 0.5 if not set)
    /// - Include if difficulty ≤ current progress
    /// - Example: At 60% progress, include samples with difficulty ≤ 0.6</para>
    ///
    /// <para><b>Progression Example (100 epochs):</b>
    /// - Epoch 0 (0% progress): Only difficulty ≤ 0.0 (almost nothing, need some easy samples!)
    /// - Epoch 25 (25%): difficulty ≤ 0.25 (easy samples)
    /// - Epoch 50 (50%): difficulty ≤ 0.50 (easy + medium)
    /// - Epoch 75 (75%): difficulty ≤ 0.75 (easy + medium + some hard)
    /// - Epoch 99 (99%): difficulty ≤ 0.99 (all samples)</para>
    ///
    /// <para><b>Note:</b> Samples without difficulty scores are always included
    /// (treated as appropriate for all stages).</para>
    /// </remarks>
    public override bool ShouldIncludeSample(int sampleIndex)
    {
        double? difficulty = GetSampleDifficulty(sampleIndex);

        // If no difficulty set, include by default
        if (difficulty == null)
            return true;

        // Easy-to-Hard: Include samples with difficulty ≤ current progress
        // Progress 0.0 → only easiest samples
        // Progress 1.0 → all samples (including hardest)
        return difficulty.Value <= CurriculumProgress;
    }

    /// <summary>
    /// Computes curriculum temperature that decreases over time (easy to hard).
    /// </summary>
    /// <returns>Current temperature based on curriculum progress.</returns>
    /// <remarks>
    /// <para><b>Algorithm:</b>
    /// Temperature decreases linearly from MaxTemperature to MinTemperature.
    /// temp = MaxTemp - progress * (MaxTemp - MinTemp)</para>
    ///
    /// <para><b>Example with defaults (min=2.0, max=5.0):</b>
    /// - Progress 0.0 (start): Temp = 5.0 (very soft, easy)
    /// - Progress 0.25: Temp = 4.25 (softer)
    /// - Progress 0.50: Temp = 3.5 (medium)
    /// - Progress 0.75: Temp = 2.75 (harder)
    /// - Progress 1.0 (end): Temp = 2.0 (sharp, hard)</para>
    ///
    /// <para><b>Intuition:</b>
    /// - **High temp early**: Soft targets help student learn basic patterns gently
    /// - **Low temp late**: Sharp targets force student to learn precise boundaries</para>
    /// </remarks>
    public override double ComputeCurriculumTemperature()
    {
        // Start with max temperature (easy/soft), decrease to min (hard/sharp)
        // High temperature early = softer targets (easier to learn)
        // Low temperature late = sharper targets (more challenging)
        double temperature = MaxTemperature - CurriculumProgress * (MaxTemperature - MinTemperature);

        return ClampTemperature(temperature);
    }
}
