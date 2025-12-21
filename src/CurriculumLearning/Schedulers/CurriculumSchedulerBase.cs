using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.Interfaces;

namespace AiDotNet.CurriculumLearning.Schedulers;

/// <summary>
/// Abstract base class for curriculum schedulers providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A curriculum scheduler determines how the training curriculum
/// progresses over time. It controls when and how many samples are included in training
/// as the model advances through the curriculum.</para>
///
/// <para><b>Core Responsibilities:</b></para>
/// <list type="bullet">
/// <item><description>Track current training progress (epoch, phase)</description></item>
/// <item><description>Calculate data fraction to use at each training stage</description></item>
/// <item><description>Select samples based on difficulty scores and current progress</description></item>
/// <item><description>Determine when curriculum is complete</description></item>
/// </list>
///
/// <para><b>Common Scheduler Types:</b></para>
/// <list type="bullet">
/// <item><description><b>Linear:</b> Constant rate of curriculum progression</description></item>
/// <item><description><b>Exponential:</b> Starts slow, accelerates progression</description></item>
/// <item><description><b>Step:</b> Discrete jumps in curriculum phases</description></item>
/// <item><description><b>Self-Paced:</b> Adapts based on model performance</description></item>
/// <item><description><b>Competence-Based:</b> Advances when model masters current content</description></item>
/// </list>
/// </remarks>
public abstract class CurriculumSchedulerBase<T> : ICurriculumScheduler<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Total number of epochs for training.
    /// </summary>
    protected int TotalEpochs { get; }

    /// <summary>
    /// Minimum data fraction to start with.
    /// </summary>
    protected T MinFraction { get; }

    /// <summary>
    /// Maximum data fraction to end with (usually 1.0).
    /// </summary>
    protected T MaxFraction { get; }

    /// <summary>
    /// Number of phases in the curriculum (for phase-based schedulers).
    /// </summary>
    private readonly int _totalPhases;

    /// <summary>
    /// Current phase number (0-indexed).
    /// </summary>
    private int _currentPhaseNumber;

    /// <summary>
    /// Gets the name of this scheduler.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the current phase as a value between 0 and 1.
    /// </summary>
    /// <remarks>
    /// <para>Phase 0 means only easiest samples are available.
    /// Phase 1 means all samples are available.</para>
    /// </remarks>
    public virtual T CurrentPhase
    {
        get
        {
            if (TotalEpochs <= 1) return NumOps.One;
            var progress = (double)CurrentEpoch / (TotalEpochs - 1);
            return NumOps.FromDouble(Math.Min(1.0, progress));
        }
    }

    /// <summary>
    /// Gets the current epoch number.
    /// </summary>
    public int CurrentEpoch { get; protected set; }

    /// <summary>
    /// Gets the total number of phases in the curriculum.
    /// </summary>
    public virtual int TotalPhases => _totalPhases;

    /// <summary>
    /// Gets the current phase number (0-indexed).
    /// </summary>
    public virtual int CurrentPhaseNumber => _currentPhaseNumber;

    /// <summary>
    /// Gets whether the curriculum has completed (all samples available).
    /// </summary>
    public virtual bool IsComplete => CurrentEpoch >= TotalEpochs;

    /// <summary>
    /// Initializes a new instance of the <see cref="CurriculumSchedulerBase{T}"/> class.
    /// </summary>
    /// <param name="totalEpochs">Total number of training epochs.</param>
    /// <param name="minFraction">Minimum data fraction to start with (default 0.1).</param>
    /// <param name="maxFraction">Maximum data fraction to end with (default 1.0).</param>
    /// <param name="totalPhases">Total number of curriculum phases (default matches epochs).</param>
    protected CurriculumSchedulerBase(
        int totalEpochs,
        T? minFraction = default,
        T? maxFraction = default,
        int? totalPhases = null)
    {
        if (totalEpochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalEpochs),
                "Total epochs must be positive.");
        }

        TotalEpochs = totalEpochs;
        MinFraction = minFraction ?? NumOps.FromDouble(0.1);
        MaxFraction = maxFraction ?? NumOps.One;
        _totalPhases = totalPhases ?? totalEpochs;

        // Validate fractions
        if (NumOps.Compare(MinFraction, NumOps.Zero) < 0 ||
            NumOps.Compare(MinFraction, NumOps.One) > 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minFraction),
                "Minimum fraction must be between 0 and 1.");
        }

        if (NumOps.Compare(MaxFraction, NumOps.Zero) < 0 ||
            NumOps.Compare(MaxFraction, NumOps.One) > 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxFraction),
                "Maximum fraction must be between 0 and 1.");
        }

        if (NumOps.Compare(MinFraction, MaxFraction) > 0)
        {
            throw new ArgumentException(
                "Minimum fraction cannot be greater than maximum fraction.");
        }

        CurrentEpoch = 0;
        _currentPhaseNumber = 0;
    }

    /// <summary>
    /// Gets the data fraction available at the current phase.
    /// </summary>
    /// <returns>Fraction of data to use (0 to 1).</returns>
    public abstract T GetDataFraction();

    /// <summary>
    /// Gets the difficulty threshold for the current phase.
    /// </summary>
    /// <returns>Maximum difficulty score allowed in current phase.</returns>
    /// <remarks>
    /// <para>Default implementation returns the data fraction as the threshold,
    /// meaning samples with normalized difficulty below this value are included.</para>
    /// </remarks>
    public virtual T GetDifficultyThreshold()
    {
        // Default: difficulty threshold equals data fraction
        return GetDataFraction();
    }

    /// <summary>
    /// Updates the scheduler after an epoch.
    /// </summary>
    /// <param name="epochMetrics">Metrics from the completed epoch.</param>
    /// <returns>True if the phase should advance, false otherwise.</returns>
    public virtual bool StepEpoch(CurriculumEpochMetrics<T> epochMetrics)
    {
        CurrentEpoch++;

        // Check if we should advance to next phase
        var epochsPerPhase = TotalEpochs / Math.Max(1, _totalPhases);
        var newPhaseNumber = CurrentEpoch / Math.Max(1, epochsPerPhase);
        newPhaseNumber = Math.Min(newPhaseNumber, _totalPhases - 1);

        if (newPhaseNumber > _currentPhaseNumber)
        {
            _currentPhaseNumber = newPhaseNumber;
            return true;
        }

        return false;
    }

    /// <summary>
    /// Advances to the next phase.
    /// </summary>
    /// <returns>True if advanced, false if already at final phase.</returns>
    public virtual bool AdvancePhase()
    {
        if (_currentPhaseNumber >= _totalPhases - 1)
        {
            return false;
        }

        _currentPhaseNumber++;
        return true;
    }

    /// <summary>
    /// Resets the scheduler to the initial phase.
    /// </summary>
    public virtual void Reset()
    {
        CurrentEpoch = 0;
        _currentPhaseNumber = 0;
    }

    /// <summary>
    /// Gets the indices of samples available at the current phase.
    /// </summary>
    /// <param name="sortedIndices">Indices sorted by difficulty (easy to hard).</param>
    /// <param name="totalSamples">Total number of samples.</param>
    /// <returns>Indices of samples available for training.</returns>
    public virtual int[] GetCurrentIndices(int[] sortedIndices, int totalSamples)
    {
        return GetIndicesAtPhase(sortedIndices, totalSamples, CurrentPhase);
    }

    /// <summary>
    /// Gets the indices of samples available at a specific phase.
    /// </summary>
    /// <param name="sortedIndices">Indices sorted by difficulty (easy to hard).</param>
    /// <param name="totalSamples">Total number of samples.</param>
    /// <param name="phase">The phase to get indices for (0 to 1).</param>
    /// <returns>Indices of samples available at the specified phase.</returns>
    public virtual int[] GetIndicesAtPhase(int[] sortedIndices, int totalSamples, T phase)
    {
        if (sortedIndices is null) throw new ArgumentNullException(nameof(sortedIndices));

        if (sortedIndices.Length != totalSamples)
        {
            throw new ArgumentException(
                $"Sorted indices count ({sortedIndices.Length}) must match total samples ({totalSamples}).");
        }

        // Calculate how many samples to include based on the phase
        var fraction = InterpolateFraction(phase);
        var numSamples = (int)Math.Ceiling(NumOps.ToDouble(fraction) * totalSamples);
        numSamples = Math.Max(1, Math.Min(numSamples, totalSamples));

        // Select easiest samples up to the calculated count
        var selected = new int[numSamples];
        Array.Copy(sortedIndices, selected, numSamples);

        return selected;
    }

    /// <summary>
    /// Interpolates between min and max fraction based on progress.
    /// </summary>
    /// <param name="t">Interpolation parameter [0, 1].</param>
    /// <returns>Interpolated fraction.</returns>
    protected T InterpolateFraction(T t)
    {
        // Clamp t to [0, 1]
        if (NumOps.Compare(t, NumOps.Zero) < 0)
            t = NumOps.Zero;
        if (NumOps.Compare(t, NumOps.One) > 0)
            t = NumOps.One;

        // Linear interpolation: min + t * (max - min)
        var range = NumOps.Subtract(MaxFraction, MinFraction);
        return NumOps.Add(MinFraction, NumOps.Multiply(t, range));
    }

    /// <summary>
    /// Gets scheduler-specific statistics.
    /// </summary>
    /// <returns>Dictionary of statistics about the scheduler state.</returns>
    public virtual Dictionary<string, object> GetStatistics()
    {
        return new Dictionary<string, object>
        {
            ["Name"] = Name,
            ["CurrentEpoch"] = CurrentEpoch,
            ["TotalEpochs"] = TotalEpochs,
            ["CurrentPhase"] = NumOps.ToDouble(CurrentPhase),
            ["CurrentPhaseNumber"] = CurrentPhaseNumber,
            ["TotalPhases"] = TotalPhases,
            ["DataFraction"] = NumOps.ToDouble(GetDataFraction()),
            ["IsComplete"] = IsComplete
        };
    }
}
