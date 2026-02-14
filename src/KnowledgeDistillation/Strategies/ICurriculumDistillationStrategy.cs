using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Interface for curriculum distillation strategies that progressively adjust training difficulty.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Curriculum learning is inspired by how humans learn - starting
/// with easy concepts and gradually increasing difficulty. This interface defines strategies
/// that control this progression during knowledge distillation.</para>
///
/// <para><b>Key Concepts:</b>
/// - **Progressive Difficulty**: Training difficulty increases (or decreases) over time
/// - **Sample Filtering**: Only include samples appropriate for current curriculum stage
/// - **Temperature Progression**: Temperature adjusts based on training progress</para>
///
/// <para><b>Common Curriculum Strategies:</b>
/// - **Easy-to-Hard**: Start with simple samples, gradually add harder ones
/// - **Hard-to-Easy**: Start with challenging samples, then easier ones (for fine-tuning)
/// - **Paced Learning**: Combine difficulty-based and time-based progression</para>
///
/// <para><b>When to Use:</b>
/// - Training data has clear difficulty levels
/// - Student model benefits from structured learning progression
/// - You want to prevent overwhelming the student early in training</para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("CurriculumDistillationStrategy")]
public interface ICurriculumDistillationStrategy<T>
{
    /// <summary>
    /// Gets the total number of steps/epochs in the curriculum.
    /// </summary>
    /// <remarks>
    /// <para>Defines the duration of the curriculum progression.
    /// After this many steps, all samples (regardless of difficulty) are included.</para>
    /// </remarks>
    int TotalSteps { get; }

    /// <summary>
    /// Gets the current curriculum progress as a ratio (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <para>0.0 = Beginning of curriculum, 1.0 = End of curriculum.
    /// Progress determines which samples are included and what temperature is used.</para>
    /// </remarks>
    double CurriculumProgress { get; }

    /// <summary>
    /// Gets the minimum temperature for the curriculum range.
    /// </summary>
    /// <remarks>
    /// <para>For Easy-to-Hard: This is the ending temperature (harder samples).
    /// For Hard-to-Easy: This is the starting temperature (harder samples).</para>
    /// </remarks>
    double MinTemperature { get; }

    /// <summary>
    /// Gets the maximum temperature for the curriculum range.
    /// </summary>
    /// <remarks>
    /// <para>For Easy-to-Hard: This is the starting temperature (easier samples).
    /// For Hard-to-Easy: This is the ending temperature (easier samples).</para>
    /// </remarks>
    double MaxTemperature { get; }

    /// <summary>
    /// Updates the current curriculum progress.
    /// </summary>
    /// <param name="step">Current training step/epoch (0 to TotalSteps-1).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this at the beginning of each epoch or training iteration
    /// to advance the curriculum. The strategy will adjust temperature and sample filtering accordingly.</para>
    ///
    /// <para>Example:
    /// <code>
    /// for (int epoch = 0; epoch &lt; 100; epoch++)
    /// {
    ///     curriculumStrategy.UpdateProgress(epoch);
    ///     // ... training loop
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    void UpdateProgress(int step);

    /// <summary>
    /// Sets the difficulty score for a specific training sample.
    /// </summary>
    /// <param name="sampleIndex">Index of the sample.</param>
    /// <param name="difficulty">Difficulty score (0.0 = easy, 1.0 = hard).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Assign difficulty scores before training to control
    /// which samples appear at each curriculum stage.</para>
    ///
    /// <para>Difficulty can be based on:
    /// - Model uncertainty on validation set
    /// - Label complexity (e.g., number of classes)
    /// - Expert annotation
    /// - Automatic difficulty estimation</para>
    /// </remarks>
    void SetSampleDifficulty(int sampleIndex, double difficulty);

    /// <summary>
    /// Determines if a sample should be included in the current curriculum stage.
    /// </summary>
    /// <param name="sampleIndex">Index of the sample.</param>
    /// <returns>True if the sample should be included in current training.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to filter your training batch based on
    /// the current curriculum stage.</para>
    ///
    /// <para>Example:
    /// <code>
    /// foreach (var (sample, index) in trainingSamples.WithIndex())
    /// {
    ///     if (!curriculumStrategy.ShouldIncludeSample(index))
    ///         continue; // Skip this sample for now
    ///
    ///     // ... train on this sample
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    bool ShouldIncludeSample(int sampleIndex);

    /// <summary>
    /// Computes the curriculum-adjusted temperature based on current progress.
    /// </summary>
    /// <returns>Temperature value for current curriculum stage.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This method should return a temperature that
    /// progresses from MaxTemperature to MinTemperature (Easy-to-Hard) or vice versa
    /// (Hard-to-Easy) based on CurriculumProgress.</para>
    /// </remarks>
    double ComputeCurriculumTemperature();

    /// <summary>
    /// Gets the difficulty score for a specific sample, if set.
    /// </summary>
    /// <param name="sampleIndex">Index of the sample.</param>
    /// <returns>Difficulty score, or null if not set.</returns>
    double? GetSampleDifficulty(int sampleIndex);
}
