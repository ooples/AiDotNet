namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Chains multiple learning rate schedulers together in sequence.
/// </summary>
/// <remarks>
/// <para>
/// SequentialLR allows you to compose multiple schedulers, each running for a specified
/// number of steps. This is useful for complex training schedules that combine different
/// strategies at different phases of training.
/// </para>
/// <para><b>For Beginners:</b> Sometimes you want different learning rate strategies at
/// different points in training. For example, you might want linear warmup for the first
/// 1000 steps, then cosine annealing for the next 9000 steps. This scheduler lets you
/// chain multiple schedulers together, specifying when to switch from one to the next.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Warmup for 1000 steps, then cosine annealing for 9000 steps
/// var schedulers = new List&lt;ILearningRateScheduler&gt;
/// {
///     new LinearWarmupScheduler(0.001, 1000, 1000),
///     new CosineAnnealingLRScheduler(0.001, 9000)
/// };
/// var milestones = new[] { 1000 };  // Switch after step 1000
/// var scheduler = new SequentialLRScheduler(schedulers, milestones);
/// </code>
/// </example>
public class SequentialLRScheduler : LearningRateSchedulerBase
{
    private readonly List<ILearningRateScheduler> _schedulers;
    private readonly int[] _milestones;
    private int _currentSchedulerIndex;

    /// <summary>
    /// Initializes a new instance of the SequentialLRScheduler class.
    /// </summary>
    /// <param name="schedulers">List of schedulers to chain together.</param>
    /// <param name="milestones">Steps at which to switch to the next scheduler.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public SequentialLRScheduler(
        IList<ILearningRateScheduler> schedulers,
        int[] milestones)
        : base(schedulers.FirstOrDefault()?.BaseLearningRate ?? 0.001)
    {
        if (schedulers == null || schedulers.Count == 0)
            throw new ArgumentException("Schedulers list cannot be null or empty.", nameof(schedulers));
        if (milestones == null || milestones.Length != schedulers.Count - 1)
            throw new ArgumentException($"Milestones must have {schedulers.Count - 1} elements (one less than schedulers).", nameof(milestones));

        // Validate milestones are increasing
        for (int i = 1; i < milestones.Length; i++)
        {
            if (milestones[i] <= milestones[i - 1])
                throw new ArgumentException("Milestones must be in strictly increasing order.", nameof(milestones));
        }

        _schedulers = schedulers.ToList();
        _milestones = milestones.ToArray();
        _currentSchedulerIndex = 0;
        _currentLearningRate = _schedulers[0].CurrentLearningRate;
    }

    /// <summary>
    /// Gets the current active scheduler index.
    /// </summary>
    public int CurrentSchedulerIndex => _currentSchedulerIndex;

    /// <summary>
    /// Gets the current active scheduler.
    /// </summary>
    public ILearningRateScheduler CurrentScheduler => _schedulers[_currentSchedulerIndex];

    /// <inheritdoc/>
    public override double Step()
    {
        _currentStep++;

        // Check if we need to switch to next scheduler
        while (_currentSchedulerIndex < _milestones.Length &&
               _currentStep > _milestones[_currentSchedulerIndex])
        {
            _currentSchedulerIndex++;
        }

        _currentLearningRate = _schedulers[_currentSchedulerIndex].Step();
        return _currentLearningRate;
    }

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        // Find which scheduler handles this step
        int schedulerIndex = 0;
        int schedulerStartStep = 0;

        for (int i = 0; i < _milestones.Length; i++)
        {
            if (step > _milestones[i])
            {
                schedulerIndex = i + 1;
                schedulerStartStep = _milestones[i];
            }
            else
            {
                break;
            }
        }

        int localStep = step - schedulerStartStep;
        return _schedulers[schedulerIndex].GetLearningRateAtStep(localStep);
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _currentSchedulerIndex = 0;
        foreach (var scheduler in _schedulers)
        {
            scheduler.Reset();
        }
        _currentLearningRate = _schedulers[0].CurrentLearningRate;
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["current_scheduler_index"] = _currentSchedulerIndex;
        state["milestones"] = _milestones;
        state["scheduler_states"] = _schedulers.Select(s => s.GetState()).ToList();
        return state;
    }

    /// <inheritdoc/>
    public override void LoadState(Dictionary<string, object> state)
    {
        base.LoadState(state);
        if (state.TryGetValue("current_scheduler_index", out var idx))
            _currentSchedulerIndex = Convert.ToInt32(idx);
        if (state.TryGetValue("scheduler_states", out var states) &&
            states is List<Dictionary<string, object>> schedulerStates)
        {
            for (int i = 0; i < Math.Min(_schedulers.Count, schedulerStates.Count); i++)
            {
                _schedulers[i].LoadState(schedulerStates[i]);
            }
        }
    }
}
