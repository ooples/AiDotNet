using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.Interfaces;

namespace AiDotNet.CurriculumLearning;

/// <summary>
/// Configuration for curriculum learning training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class holds all the settings that control how
/// curriculum learning works. You can configure things like how many training phases
/// to use, when to stop early if the model isn't improving, and how to schedule
/// the progression from easy to hard samples.</para>
///
/// <para><b>Key Configuration Areas:</b></para>
/// <list type="bullet">
/// <item><description><b>Training:</b> TotalEpochs, BatchSize, LearningRate</description></item>
/// <item><description><b>Curriculum:</b> NumPhases, InitialDataFraction, FinalDataFraction</description></item>
/// <item><description><b>Schedule:</b> ScheduleType (Linear, Exponential, Step, etc.)</description></item>
/// <item><description><b>Early Stopping:</b> Patience, MinDelta for stopping when plateaued</description></item>
/// </list>
/// </remarks>
public class CurriculumLearnerConfig<T> : ICurriculumLearnerConfig<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the total number of training epochs.
    /// </summary>
    public int TotalEpochs { get; init; } = 100;

    /// <summary>
    /// Gets the number of curriculum phases.
    /// </summary>
    public int NumPhases { get; init; } = 5;

    /// <summary>
    /// Gets the number of epochs per phase.
    /// </summary>
    public int EpochsPerPhase => TotalEpochs / Math.Max(1, NumPhases);

    /// <summary>
    /// Gets the initial data fraction (starting fraction of easiest samples).
    /// </summary>
    public T InitialDataFraction { get; init; }

    /// <summary>
    /// Gets the final data fraction (usually 1.0 to include all samples).
    /// </summary>
    public T FinalDataFraction { get; init; }

    /// <summary>
    /// Gets the curriculum schedule type.
    /// </summary>
    public CurriculumScheduleType ScheduleType { get; init; } = CurriculumScheduleType.Linear;

    /// <summary>
    /// Gets whether to recalculate sample difficulties during training.
    /// </summary>
    public bool RecalculateDifficulties { get; init; } = false;

    /// <summary>
    /// Gets how often to recalculate difficulties (every N epochs).
    /// </summary>
    public int DifficultyRecalculationFrequency { get; init; } = 10;

    /// <summary>
    /// Gets whether to normalize difficulty scores to [0, 1].
    /// </summary>
    public bool NormalizeDifficulties { get; init; } = true;

    /// <summary>
    /// Gets the number of epochs without improvement before early stopping.
    /// </summary>
    public int EarlyStoppingPatience { get; init; } = 10;

    /// <summary>
    /// Gets the minimum improvement required to reset early stopping counter.
    /// </summary>
    public T EarlyStoppingMinDelta { get; init; }

    /// <summary>
    /// Gets whether early stopping is enabled.
    /// </summary>
    public bool UseEarlyStopping { get; init; } = true;

    /// <summary>
    /// Gets the batch size for training.
    /// </summary>
    public int BatchSize { get; init; } = 32;

    /// <summary>
    /// Gets the learning rate.
    /// </summary>
    public T LearningRate { get; init; }

    /// <summary>
    /// Gets whether to shuffle samples within each phase.
    /// </summary>
    public bool ShuffleWithinPhase { get; init; } = true;

    /// <summary>
    /// Gets whether to weight sample contributions by difficulty.
    /// </summary>
    public bool UseDifficultyWeighting { get; init; } = false;

    /// <summary>
    /// Gets the random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get; init; }

    /// <summary>
    /// Gets the verbosity level for logging.
    /// </summary>
    public CurriculumVerbosity Verbosity { get; init; } = CurriculumVerbosity.Normal;

    /// <summary>
    /// Gets the custom logging action.
    /// </summary>
    /// <remarks>
    /// <para>If null, logs to Console.WriteLine by default.
    /// Provide a custom action to integrate with your logging framework
    /// (e.g., Serilog, NLog, Microsoft.Extensions.Logging).</para>
    /// </remarks>
    public Action<string>? LogAction { get; init; }

    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public CurriculumLearnerConfig()
    {
        InitialDataFraction = NumOps.FromDouble(0.2);
        FinalDataFraction = NumOps.One;
        EarlyStoppingMinDelta = NumOps.FromDouble(0.001);
        LearningRate = NumOps.FromDouble(0.001);
    }

    /// <summary>
    /// Creates a copy of this configuration.
    /// </summary>
    public CurriculumLearnerConfig<T> Clone()
    {
        return new CurriculumLearnerConfig<T>
        {
            TotalEpochs = TotalEpochs,
            NumPhases = NumPhases,
            InitialDataFraction = InitialDataFraction,
            FinalDataFraction = FinalDataFraction,
            ScheduleType = ScheduleType,
            RecalculateDifficulties = RecalculateDifficulties,
            DifficultyRecalculationFrequency = DifficultyRecalculationFrequency,
            NormalizeDifficulties = NormalizeDifficulties,
            EarlyStoppingPatience = EarlyStoppingPatience,
            EarlyStoppingMinDelta = EarlyStoppingMinDelta,
            UseEarlyStopping = UseEarlyStopping,
            BatchSize = BatchSize,
            LearningRate = LearningRate,
            ShuffleWithinPhase = ShuffleWithinPhase,
            UseDifficultyWeighting = UseDifficultyWeighting,
            RandomSeed = RandomSeed,
            Verbosity = Verbosity,
            LogAction = LogAction
        };
    }

    /// <summary>
    /// Creates a builder for fluent configuration.
    /// </summary>
    public static CurriculumLearnerConfigBuilder<T> CreateBuilder()
    {
        return new CurriculumLearnerConfigBuilder<T>();
    }
}
