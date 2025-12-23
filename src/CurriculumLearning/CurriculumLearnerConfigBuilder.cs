using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.Interfaces;

namespace AiDotNet.CurriculumLearning;

/// <summary>
/// Fluent builder for curriculum learning configuration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This builder lets you configure curriculum learning
/// using a fluent, readable style. Each method returns the builder so you can
/// chain multiple settings together.</para>
///
/// <para><b>Example Usage:</b></para>
/// <code>
/// var config = CurriculumLearnerConfig&lt;double&gt;.CreateBuilder()
///     .WithTotalEpochs(100)
///     .WithNumPhases(5)
///     .WithSchedule(CurriculumScheduleType.Linear)
///     .WithEarlyStopping(patience: 10)
///     .Build();
/// </code>
/// </remarks>
public class CurriculumLearnerConfigBuilder<T> : ICurriculumLearnerConfigBuilder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private int _totalEpochs = 100;
    private int _numPhases = 5;
    private T _initialDataFraction;
    private T _finalDataFraction;
    private CurriculumScheduleType _scheduleType = CurriculumScheduleType.Linear;
    private bool _recalculateDifficulties = false;
    private int _difficultyRecalculationFrequency = 10;
    private bool _normalizeDifficulties = true;
    private int _earlyStoppingPatience = 10;
    private T _earlyStoppingMinDelta;
    private bool _useEarlyStopping = true;
    private int _batchSize = 32;
    private T _learningRate;
    private bool _shuffleWithinPhase = true;
    private bool _useDifficultyWeighting = false;
    private int? _randomSeed;
    private CurriculumVerbosity _verbosity = CurriculumVerbosity.Normal;
    private Action<string>? _logAction;

    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public CurriculumLearnerConfigBuilder()
    {
        _initialDataFraction = NumOps.FromDouble(0.2);
        _finalDataFraction = NumOps.One;
        _earlyStoppingMinDelta = NumOps.FromDouble(0.001);
        _learningRate = NumOps.FromDouble(0.001);
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithTotalEpochs(int epochs)
    {
        if (epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(epochs), "Total epochs must be positive.");
        _totalEpochs = epochs;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithNumPhases(int phases)
    {
        if (phases <= 0)
            throw new ArgumentOutOfRangeException(nameof(phases), "Number of phases must be positive.");
        _numPhases = phases;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithInitialDataFraction(T fraction)
    {
        if (NumOps.Compare(fraction, NumOps.Zero) < 0 || NumOps.Compare(fraction, NumOps.One) > 0)
            throw new ArgumentOutOfRangeException(nameof(fraction), "Fraction must be between 0 and 1.");
        _initialDataFraction = fraction;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithFinalDataFraction(T fraction)
    {
        if (NumOps.Compare(fraction, NumOps.Zero) < 0 || NumOps.Compare(fraction, NumOps.One) > 0)
            throw new ArgumentOutOfRangeException(nameof(fraction), "Fraction must be between 0 and 1.");
        _finalDataFraction = fraction;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithScheduleType(CurriculumScheduleType scheduleType)
    {
        _scheduleType = scheduleType;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithDifficultyRecalculation(int frequency)
    {
        if (frequency <= 0)
            throw new ArgumentOutOfRangeException(nameof(frequency), "Frequency must be positive.");
        _recalculateDifficulties = true;
        _difficultyRecalculationFrequency = frequency;
        return this;
    }

    /// <summary>
    /// Enables or disables difficulty recalculation during training.
    /// </summary>
    /// <param name="enabled">Whether to enable difficulty recalculation.</param>
    /// <param name="frequency">How often to recalculate (in epochs).</param>
    public ICurriculumLearnerConfigBuilder<T> WithDifficultyRecalculation(bool enabled, int frequency = 10)
    {
        if (frequency <= 0)
            throw new ArgumentOutOfRangeException(nameof(frequency), "Frequency must be positive.");
        _recalculateDifficulties = enabled;
        _difficultyRecalculationFrequency = frequency;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithNormalizeDifficulties(bool normalize)
    {
        _normalizeDifficulties = normalize;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithEarlyStopping(int patience, T? minDelta = default)
    {
        if (patience <= 0)
            throw new ArgumentOutOfRangeException(nameof(patience), "Patience must be positive.");
        _useEarlyStopping = true;
        _earlyStoppingPatience = patience;
        if (minDelta != null)
        {
            _earlyStoppingMinDelta = minDelta;
        }
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithoutEarlyStopping()
    {
        _useEarlyStopping = false;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithBatchSize(int batchSize)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
        _batchSize = batchSize;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithLearningRate(T learningRate)
    {
        if (NumOps.Compare(learningRate, NumOps.Zero) <= 0)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        _learningRate = learningRate;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithShuffling(bool shuffle)
    {
        _shuffleWithinPhase = shuffle;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithDifficultyWeighting(bool useWeighting)
    {
        _useDifficultyWeighting = useWeighting;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithRandomSeed(int seed)
    {
        _randomSeed = seed;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithVerbosity(CurriculumVerbosity verbosity)
    {
        _verbosity = verbosity;
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfigBuilder<T> WithLogAction(Action<string> logAction)
    {
        _logAction = logAction ?? throw new ArgumentNullException(nameof(logAction));
        return this;
    }

    /// <inheritdoc/>
    public ICurriculumLearnerConfig<T> Build()
    {
        // Validate that initial <= final
        if (NumOps.Compare(_initialDataFraction, _finalDataFraction) > 0)
        {
            throw new InvalidOperationException(
                "Initial data fraction cannot be greater than final data fraction.");
        }

        return new CurriculumLearnerConfig<T>
        {
            TotalEpochs = _totalEpochs,
            NumPhases = _numPhases,
            InitialDataFraction = _initialDataFraction,
            FinalDataFraction = _finalDataFraction,
            ScheduleType = _scheduleType,
            RecalculateDifficulties = _recalculateDifficulties,
            DifficultyRecalculationFrequency = _difficultyRecalculationFrequency,
            NormalizeDifficulties = _normalizeDifficulties,
            EarlyStoppingPatience = _earlyStoppingPatience,
            EarlyStoppingMinDelta = _earlyStoppingMinDelta,
            UseEarlyStopping = _useEarlyStopping,
            BatchSize = _batchSize,
            LearningRate = _learningRate,
            ShuffleWithinPhase = _shuffleWithinPhase,
            UseDifficultyWeighting = _useDifficultyWeighting,
            RandomSeed = _randomSeed,
            Verbosity = _verbosity,
            LogAction = _logAction
        };
    }

    /// <summary>
    /// Resets the builder to default values.
    /// </summary>
    public ICurriculumLearnerConfigBuilder<T> Reset()
    {
        _totalEpochs = 100;
        _numPhases = 5;
        _initialDataFraction = NumOps.FromDouble(0.2);
        _finalDataFraction = NumOps.One;
        _scheduleType = CurriculumScheduleType.Linear;
        _recalculateDifficulties = false;
        _difficultyRecalculationFrequency = 10;
        _normalizeDifficulties = true;
        _earlyStoppingPatience = 10;
        _earlyStoppingMinDelta = NumOps.FromDouble(0.001);
        _useEarlyStopping = true;
        _batchSize = 32;
        _learningRate = NumOps.FromDouble(0.001);
        _shuffleWithinPhase = true;
        _useDifficultyWeighting = false;
        _randomSeed = null;
        _verbosity = CurriculumVerbosity.Normal;
        _logAction = null;
        return this;
    }
}
