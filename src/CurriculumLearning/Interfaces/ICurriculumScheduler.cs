namespace AiDotNet.CurriculumLearning.Interfaces;

/// <summary>
/// Interface for curriculum schedulers that control training progression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A curriculum scheduler decides when and how to introduce
/// harder training samples. Think of it like a teacher who decides when students are
/// ready to move from addition to multiplication.</para>
///
/// <para><b>Common Scheduling Strategies:</b></para>
/// <list type="bullet">
/// <item><description><b>Linear:</b> Increase data fraction steadily over epochs</description></item>
/// <item><description><b>Exponential:</b> Start slow, then rapidly include more data</description></item>
/// <item><description><b>Step:</b> Discrete jumps at fixed intervals</description></item>
/// <item><description><b>Self-paced:</b> Adapt based on model's learning progress</description></item>
/// <item><description><b>Competence-based:</b> Advance when model masters current content</description></item>
/// </list>
///
/// <para><b>Key Concepts:</b></para>
/// <list type="bullet">
/// <item><description><b>Phase:</b> A value in [0, 1] indicating curriculum progress</description></item>
/// <item><description><b>Data Fraction:</b> Portion of training data available at current phase</description></item>
/// <item><description><b>Difficulty Threshold:</b> Maximum difficulty of samples in current phase</description></item>
/// </list>
/// </remarks>
public interface ICurriculumScheduler<T>
{
    /// <summary>
    /// Gets the name of the scheduler.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the current phase (0 to 1).
    /// </summary>
    /// <remarks>
    /// <para>Phase 0 means only easiest samples are available.
    /// Phase 1 means all samples are available.</para>
    /// </remarks>
    T CurrentPhase { get; }

    /// <summary>
    /// Gets the current epoch number.
    /// </summary>
    int CurrentEpoch { get; }

    /// <summary>
    /// Gets the total number of phases in the curriculum.
    /// </summary>
    int TotalPhases { get; }

    /// <summary>
    /// Gets the current phase number (0-indexed).
    /// </summary>
    int CurrentPhaseNumber { get; }

    /// <summary>
    /// Gets whether the curriculum is complete (all samples available).
    /// </summary>
    bool IsComplete { get; }

    /// <summary>
    /// Gets the data fraction available at the current phase.
    /// </summary>
    /// <returns>Fraction of data to use (0 to 1).</returns>
    T GetDataFraction();

    /// <summary>
    /// Gets the difficulty threshold for the current phase.
    /// </summary>
    /// <returns>Maximum difficulty score allowed in current phase.</returns>
    T GetDifficultyThreshold();

    /// <summary>
    /// Updates the scheduler after an epoch.
    /// </summary>
    /// <param name="epochMetrics">Metrics from the completed epoch (loss, accuracy, etc.).</param>
    /// <returns>True if the phase should advance, false otherwise.</returns>
    bool StepEpoch(CurriculumEpochMetrics<T> epochMetrics);

    /// <summary>
    /// Advances to the next phase.
    /// </summary>
    /// <returns>True if advanced, false if already at final phase.</returns>
    bool AdvancePhase();

    /// <summary>
    /// Resets the scheduler to the initial phase.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets the indices of samples available at the current phase.
    /// </summary>
    /// <param name="sortedIndices">Indices sorted by difficulty (easy to hard).</param>
    /// <param name="totalSamples">Total number of samples.</param>
    /// <returns>Indices of samples available for training.</returns>
    int[] GetCurrentIndices(int[] sortedIndices, int totalSamples);

    /// <summary>
    /// Gets the indices of samples available at a specific phase.
    /// </summary>
    /// <param name="sortedIndices">Indices sorted by difficulty (easy to hard).</param>
    /// <param name="totalSamples">Total number of samples.</param>
    /// <param name="phase">The phase to get indices for (0 to 1).</param>
    /// <returns>Indices of samples available at the specified phase.</returns>
    int[] GetIndicesAtPhase(int[] sortedIndices, int totalSamples, T phase);

    /// <summary>
    /// Gets statistics about the scheduler's current state.
    /// </summary>
    /// <returns>Dictionary of statistics.</returns>
    Dictionary<string, object> GetStatistics();
}

/// <summary>
/// Interface for self-paced curriculum schedulers that adapt based on model performance.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public interface ISelfPacedScheduler<T> : ICurriculumScheduler<T>
{
    /// <summary>
    /// Gets or sets the pace parameter (lambda in self-paced learning).
    /// </summary>
    /// <remarks>
    /// <para>Controls how quickly the curriculum advances. Higher values
    /// lead to faster inclusion of harder samples.</para>
    /// </remarks>
    T PaceParameter { get; set; }

    /// <summary>
    /// Gets or sets the growth rate for the pace parameter.
    /// </summary>
    T GrowthRate { get; set; }

    /// <summary>
    /// Computes sample weights for self-paced learning.
    /// </summary>
    /// <param name="losses">Per-sample losses.</param>
    /// <returns>Weights for each sample (0 = ignore, 1 = full weight).</returns>
    Vector<T> ComputeSampleWeights(Vector<T> losses);
}

/// <summary>
/// Interface for competence-based curriculum schedulers.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public interface ICompetenceBasedScheduler<T> : ICurriculumScheduler<T>
{
    /// <summary>
    /// Gets the current competence level of the model.
    /// </summary>
    T CurrentCompetence { get; }

    /// <summary>
    /// Gets or sets the competence threshold to advance to next phase.
    /// </summary>
    T CompetenceThreshold { get; set; }

    /// <summary>
    /// Updates the competence estimate based on model performance.
    /// </summary>
    /// <param name="metrics">Performance metrics from current epoch.</param>
    void UpdateCompetence(CurriculumEpochMetrics<T> metrics);

    /// <summary>
    /// Gets whether the model has mastered the current curriculum content.
    /// </summary>
    /// <returns>True if competence exceeds threshold.</returns>
    bool HasMasteredCurrentContent();
}

/// <summary>
/// Metrics from a curriculum learning epoch.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class CurriculumEpochMetrics<T>
{
    /// <summary>
    /// Gets or sets the average training loss.
    /// </summary>
    public T TrainingLoss { get; set; } = default!;

    /// <summary>
    /// Gets or sets the validation loss (if validation data provided).
    /// </summary>
    public T? ValidationLoss { get; set; }

    /// <summary>
    /// Gets or sets the training accuracy (for classification).
    /// </summary>
    public T? TrainingAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the validation accuracy (if applicable).
    /// </summary>
    public T? ValidationAccuracy { get; set; }

    /// <summary>
    /// Gets or sets per-sample losses.
    /// </summary>
    public Vector<T>? SampleLosses { get; set; }

    /// <summary>
    /// Gets or sets the number of samples used in this epoch.
    /// </summary>
    public int SamplesUsed { get; set; }

    /// <summary>
    /// Gets or sets the current epoch number.
    /// </summary>
    public int Epoch { get; set; }

    /// <summary>
    /// Gets or sets the current phase number.
    /// </summary>
    public int Phase { get; set; }

    /// <summary>
    /// Gets or sets the improvement in loss from previous epoch.
    /// </summary>
    public T? LossImprovement { get; set; }

    /// <summary>
    /// Gets or sets whether this epoch showed improvement.
    /// </summary>
    public bool Improved { get; set; }
}

/// <summary>
/// Types of curriculum scheduling strategies.
/// </summary>
public enum CurriculumScheduleType
{
    /// <summary>
    /// Linear increase in data fraction over epochs.
    /// </summary>
    Linear,

    /// <summary>
    /// Exponential increase in data fraction.
    /// </summary>
    Exponential,

    /// <summary>
    /// Fixed step increases at regular intervals.
    /// </summary>
    Step,

    /// <summary>
    /// Logarithmic growth (fast initial increase, then slow).
    /// </summary>
    Logarithmic,

    /// <summary>
    /// Self-paced learning (SPL) - adapts based on sample losses.
    /// </summary>
    SelfPaced,

    /// <summary>
    /// Competence-based - advances when model masters content.
    /// </summary>
    CompetenceBased,

    /// <summary>
    /// Baby steps - very gradual introduction of harder samples.
    /// </summary>
    BabySteps,

    /// <summary>
    /// One-pass - each sample seen exactly once in curriculum order.
    /// </summary>
    OnePass,

    /// <summary>
    /// Polynomial curve progression.
    /// </summary>
    Polynomial,

    /// <summary>
    /// Cosine annealing progression.
    /// </summary>
    Cosine
}
