namespace AiDotNet.CurriculumLearning.Interfaces;

/// <summary>
/// Configuration interface for curriculum learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This configuration controls how curriculum learning
/// behaves. You can set things like how many epochs to train, how to schedule
/// the introduction of harder samples, and when to stop training.</para>
///
/// <para><b>Key Configuration Areas:</b></para>
/// <list type="bullet">
/// <item><description><b>Training Duration:</b> Total epochs, epochs per phase</description></item>
/// <item><description><b>Curriculum Pacing:</b> Starting fraction, schedule type</description></item>
/// <item><description><b>Early Stopping:</b> Patience, improvement threshold</description></item>
/// <item><description><b>Difficulty Settings:</b> Recalculation frequency, normalization</description></item>
/// </list>
/// </remarks>
public interface ICurriculumLearnerConfig<T>
{
    /// <summary>
    /// Gets the total number of training epochs.
    /// </summary>
    int TotalEpochs { get; }

    /// <summary>
    /// Gets the number of curriculum phases.
    /// </summary>
    int NumPhases { get; }

    /// <summary>
    /// Gets the number of epochs per curriculum phase.
    /// </summary>
    int EpochsPerPhase { get; }

    /// <summary>
    /// Gets the initial data fraction (at phase 0).
    /// </summary>
    /// <remarks>
    /// <para>Typically starts between 0.1 and 0.4 to include only
    /// the easiest samples initially.</para>
    /// </remarks>
    T InitialDataFraction { get; }

    /// <summary>
    /// Gets the final data fraction (at final phase).
    /// </summary>
    /// <remarks>
    /// <para>Usually 1.0 to include all samples by the end of training.</para>
    /// </remarks>
    T FinalDataFraction { get; }

    /// <summary>
    /// Gets the type of curriculum schedule to use.
    /// </summary>
    CurriculumScheduleType ScheduleType { get; }

    /// <summary>
    /// Gets whether to recalculate difficulty scores during training.
    /// </summary>
    /// <remarks>
    /// <para>If true, difficulties are updated as the model learns.
    /// This can be more accurate but also more expensive.</para>
    /// </remarks>
    bool RecalculateDifficulties { get; }

    /// <summary>
    /// Gets how often to recalculate difficulties (in epochs).
    /// </summary>
    int DifficultyRecalculationFrequency { get; }

    /// <summary>
    /// Gets whether to normalize difficulty scores to [0, 1].
    /// </summary>
    bool NormalizeDifficulties { get; }

    /// <summary>
    /// Gets the patience for early stopping (epochs without improvement).
    /// </summary>
    int EarlyStoppingPatience { get; }

    /// <summary>
    /// Gets the minimum improvement required to reset patience.
    /// </summary>
    T EarlyStoppingMinDelta { get; }

    /// <summary>
    /// Gets whether to use early stopping.
    /// </summary>
    bool UseEarlyStopping { get; }

    /// <summary>
    /// Gets the batch size for training.
    /// </summary>
    int BatchSize { get; }

    /// <summary>
    /// Gets the learning rate.
    /// </summary>
    T LearningRate { get; }

    /// <summary>
    /// Gets whether to shuffle within curriculum phases.
    /// </summary>
    bool ShuffleWithinPhase { get; }

    /// <summary>
    /// Gets whether to apply sample weighting based on difficulty.
    /// </summary>
    bool UseDifficultyWeighting { get; }

    /// <summary>
    /// Gets the random seed for reproducibility.
    /// </summary>
    int? RandomSeed { get; }

    /// <summary>
    /// Gets the verbosity level for logging.
    /// </summary>
    CurriculumVerbosity Verbosity { get; }

    /// <summary>
    /// Gets the custom logging action.
    /// </summary>
    /// <remarks>
    /// <para>If null, logs to Console.WriteLine by default.
    /// Provide a custom action to integrate with your logging framework
    /// (e.g., Serilog, NLog, Microsoft.Extensions.Logging).</para>
    /// </remarks>
    Action<string>? LogAction { get; }
}

/// <summary>
/// Builder for curriculum learner configuration.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public interface ICurriculumLearnerConfigBuilder<T>
{
    /// <summary>
    /// Sets the total number of training epochs.
    /// </summary>
    ICurriculumLearnerConfigBuilder<T> WithTotalEpochs(int epochs);

    /// <summary>
    /// Sets the number of curriculum phases.
    /// </summary>
    ICurriculumLearnerConfigBuilder<T> WithNumPhases(int phases);

    /// <summary>
    /// Sets the initial data fraction.
    /// </summary>
    ICurriculumLearnerConfigBuilder<T> WithInitialDataFraction(T fraction);

    /// <summary>
    /// Sets the curriculum schedule type.
    /// </summary>
    ICurriculumLearnerConfigBuilder<T> WithScheduleType(CurriculumScheduleType scheduleType);

    /// <summary>
    /// Enables difficulty recalculation during training.
    /// </summary>
    ICurriculumLearnerConfigBuilder<T> WithDifficultyRecalculation(int frequency);

    /// <summary>
    /// Configures early stopping.
    /// </summary>
    ICurriculumLearnerConfigBuilder<T> WithEarlyStopping(int patience, T minDelta);

    /// <summary>
    /// Sets the batch size.
    /// </summary>
    ICurriculumLearnerConfigBuilder<T> WithBatchSize(int batchSize);

    /// <summary>
    /// Sets the learning rate.
    /// </summary>
    ICurriculumLearnerConfigBuilder<T> WithLearningRate(T learningRate);

    /// <summary>
    /// Sets the random seed.
    /// </summary>
    ICurriculumLearnerConfigBuilder<T> WithRandomSeed(int seed);

    /// <summary>
    /// Sets the verbosity level.
    /// </summary>
    ICurriculumLearnerConfigBuilder<T> WithVerbosity(CurriculumVerbosity verbosity);

    /// <summary>
    /// Sets a custom logging action.
    /// </summary>
    /// <param name="logAction">Action to invoke for logging messages.</param>
    ICurriculumLearnerConfigBuilder<T> WithLogAction(Action<string> logAction);

    /// <summary>
    /// Builds the configuration.
    /// </summary>
    ICurriculumLearnerConfig<T> Build();
}

/// <summary>
/// Verbosity levels for curriculum learning.
/// </summary>
public enum CurriculumVerbosity
{
    /// <summary>
    /// No logging.
    /// </summary>
    Silent,

    /// <summary>
    /// Log only phase transitions.
    /// </summary>
    Minimal,

    /// <summary>
    /// Log phase transitions and per-epoch metrics.
    /// </summary>
    Normal,

    /// <summary>
    /// Log detailed per-epoch information.
    /// </summary>
    Verbose,

    /// <summary>
    /// Log everything including per-sample information.
    /// </summary>
    Debug
}
