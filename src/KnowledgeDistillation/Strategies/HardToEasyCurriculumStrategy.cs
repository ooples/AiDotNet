using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Curriculum distillation strategy that progresses from hard to easy samples.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Hard-to-easy curriculum is the opposite of traditional
/// curriculum learning. It starts with challenging samples and gradually includes easier ones.
/// This approach is useful for fine-tuning already-trained models or when the student
/// has prior knowledge.</para>
///
/// <para><b>How It Works:</b>
/// 1. **Early Training** (progress 0.0-0.3):
///    - Include only hard samples (difficulty ≥ 0.7)
///    - Use low temperature (sharp targets, challenging)
/// 2. **Mid Training** (progress 0.3-0.7):
///    - Include hard and medium samples (difficulty ≥ 0.3)
///    - Gradually increase temperature
/// 3. **Late Training** (progress 0.7-1.0):
///    - Include all samples (even easy ones)
///    - Use high temperature (soft targets, exploratory)</para>
///
/// <para><b>Temperature Progression:</b>
/// Starts at MinTemperature (e.g., 2.0) and linearly increases to MaxTemperature (e.g., 5.0)
/// as training progresses. This makes distillation progressively easier.</para>
///
/// <para><b>Sample Filtering:</b>
/// At progress P, only include samples with difficulty ≥ (1 - P).
/// Example: At 50% progress, only samples with difficulty ≥ 0.5 are included.</para>
///
/// <para><b>Real-World Analogy:</b>
/// Training an advanced student: Start with challenging problems to identify gaps in knowledge,
/// then fill in easier concepts they might have missed. Like a PhD student reviewing
/// undergraduate material to strengthen foundations.</para>
///
/// <para><b>When to Use Hard-to-Easy:</b>
/// - **Fine-tuning**: Student already has base knowledge
/// - **Transfer Learning**: Adapting pre-trained model to new domain
/// - **Anti-forgetting**: Prevent model from forgetting hard concepts
/// - **Expert Refinement**: Polish already-good student model
/// - **Debugging**: Identify which hard samples student struggles with</para>
///
/// <para><b>Advantages:</b>
/// - Forces student to tackle challenges early
/// - Identifies weaknesses quickly
/// - Can improve performance on difficult edge cases
/// - Prevents overfitting to easy samples</para>
///
/// <para><b>Disadvantages:</b>
/// - Can be unstable if student has no prior knowledge
/// - May not converge well from random initialization
/// - Harder to tune than easy-to-hard</para>
///
/// <para><b>References:</b>
/// - Krueger &amp; Dayan (2009). Flexible shaping: How learning in small steps helps.
/// - Kumar et al. (2010). Self-paced curriculum learning (discusses anti-curriculum).</para>
/// </remarks>
public class HardToEasyCurriculumStrategy<T> : CurriculumDistillationStrategyBase<T>
{
    /// <summary>
    /// Initializes a new instance of the HardToEasyCurriculumStrategy class.
    /// </summary>
    /// <param name="baseTemperature">Base temperature for distillation (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="minTemperature">Starting temperature for hard samples (default: 2.0).</param>
    /// <param name="maxTemperature">Ending temperature for easy samples (default: 5.0).</param>
    /// <param name="totalSteps">Total training steps/epochs (default: 100).</param>
    /// <param name="sampleDifficulties">Optional pre-defined difficulty scores (0.0=easy, 1.0=hard).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this strategy when fine-tuning an already-trained
    /// student model. Not recommended for training from scratch.</para>
    ///
    /// <para>Example:
    /// <code>
    /// // Fine-tuning a pre-trained student model
    /// var difficulties = ComputeSampleDifficulties(validationSet);
    ///
    /// var strategy = new HardToEasyCurriculumStrategy&lt;double&gt;(
    ///     minTemperature: 1.5,   // Initial temperature (hard phase) - challenging!
    ///     maxTemperature: 4.0,   // Final temperature (easy phase) - gentle
    ///     totalSteps: 50,        // Shorter curriculum for fine-tuning
    ///     sampleDifficulties: difficulties
    /// );
    ///
    /// // Fine-tuning loop
    /// for (int epoch = 0; epoch &lt; 50; epoch++)
    /// {
    ///     strategy.UpdateProgress(epoch);
    ///
    ///     foreach (var (sample, index) in trainingSamples.WithIndex())
    ///     {
    ///         // Filter: Only hard samples early, all samples later
    ///         if (!strategy.ShouldIncludeSample(index))
    ///             continue; // Too easy for current stage
    ///
    ///         // Fine-tune on this sample...
    ///     }
    /// }
    /// </code>
    /// </para>
    ///
    /// <para><b>Difficulty Scoring for Fine-Tuning:</b>
    /// - Student's current loss on sample (higher = harder)
    /// - Disagreement with teacher (higher = harder)
    /// - Sample rarity in training set (rarer = harder)
    /// - Domain distance from pre-training data</para>
    /// </remarks>
    public HardToEasyCurriculumStrategy(
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
    /// Determines if a sample should be included based on curriculum progress (inverted).
    /// </summary>
    /// <param name="sampleIndex">Index of the sample.</param>
    /// <returns>True if sample difficulty is within current curriculum stage.</returns>
    /// <remarks>
    /// <para><b>Algorithm:</b>
    /// - Get sample difficulty (or assume 0.5 if not set)
    /// - Include if difficulty ≥ (1 - current progress)
    /// - Example: At 60% progress, include samples with difficulty ≥ 0.4</para>
    ///
    /// <para><b>Progression Example (100 epochs):</b>
    /// - Epoch 0 (0% progress): Only difficulty ≥ 1.0 (hardest samples only)
    /// - Epoch 25 (25%): difficulty ≥ 0.75 (hard samples)
    /// - Epoch 50 (50%): difficulty ≥ 0.50 (hard + medium)
    /// - Epoch 75 (75%): difficulty ≥ 0.25 (hard + medium + some easy)
    /// - Epoch 99 (99%): difficulty ≥ 0.01 (all samples)</para>
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

        // Hard-to-Easy: Include samples with difficulty >= (1 - current progress)
        // Progress 0.0 → only hardest samples (difficulty >= 1.0, threshold = 1.0)
        // Progress 0.5 → medium samples (difficulty >= 0.5, threshold = 0.5)  
        // Progress 1.0 → all samples (difficulty >= 0.0, threshold = 0.0)
        // With progress now able to reach exactly 1.0, threshold can reach 0.0
        double threshold = 1.0 - CurriculumProgress;
        return difficulty.Value >= threshold;
    }

    /// <summary>
    /// Computes curriculum temperature that increases over time (hard to easy).
    /// </summary>
    /// <returns>Current temperature based on curriculum progress.</returns>
    /// <remarks>
    /// <para><b>Algorithm:</b>
    /// Temperature increases linearly from MinTemperature to MaxTemperature.
    /// temp = MinTemp + progress * (MaxTemp - MinTemp)</para>
    ///
    /// <para><b>Example with defaults (min=2.0, max=5.0):</b>
    /// - Progress 0.0 (start): Temp = 2.0 (sharp, hard, challenging)
    /// - Progress 0.25: Temp = 2.75 (harder)
    /// - Progress 0.50: Temp = 3.5 (medium)
    /// - Progress 0.75: Temp = 4.25 (softer)
    /// - Progress 1.0 (end): Temp = 5.0 (very soft, easy, exploratory)</para>
    ///
    /// <para><b>Intuition:</b>
    /// - **Low temp early**: Sharp targets force student to learn hard patterns precisely
    /// - **High temp late**: Soft targets help student generalize to easier patterns</para>
    ///
    /// <para><b>Use Case:</b>
    /// When fine-tuning, start with sharp targets to fix errors on hard samples,
    /// then soften to improve overall generalization.</para>
    /// </remarks>
    public override double ComputeCurriculumTemperature()
    {
        // Start with min temperature (hard/sharp), increase to max (easy/soft)
        // Low temperature early = sharper targets (more challenging)
        // High temperature late = softer targets (easier)
        double temperature = MinTemperature + CurriculumProgress * (MaxTemperature - MinTemperature);

        return ClampTemperature(temperature);
    }
}
