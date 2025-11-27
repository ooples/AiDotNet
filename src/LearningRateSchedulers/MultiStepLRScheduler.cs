namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Decays the learning rate by gamma at each milestone step.
/// </summary>
/// <remarks>
/// <para>
/// MultiStepLR decays the learning rate by gamma once the number of steps reaches one of the milestones.
/// This allows for non-uniform decay schedules where you specify exactly when the learning rate should decrease.
/// </para>
/// <para><b>For Beginners:</b> Unlike StepLR which decays at regular intervals, MultiStepLR lets you
/// specify exactly which steps to decay the learning rate at. For example, you might want to decay
/// at epochs 30, 60, and 90, rather than every 30 epochs. This gives you more control over the training schedule.
/// </para>
/// <para>
/// This is useful when you know from experience or experimentation that certain epochs are good
/// points to reduce the learning rate.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Decay at epochs 30, 80, and 120
/// var scheduler = new MultiStepLRScheduler(
///     baseLearningRate: 0.1,
///     milestones: new[] { 30, 80, 120 },
///     gamma: 0.1
/// );
/// </code>
/// </example>
public class MultiStepLRScheduler : LearningRateSchedulerBase
{
    private readonly int[] _milestones;
    private readonly double _gamma;

    /// <summary>
    /// Initializes a new instance of the MultiStepLRScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">The initial learning rate.</param>
    /// <param name="milestones">List of step indices at which to decay the learning rate. Must be increasing.</param>
    /// <param name="gamma">Multiplicative factor of learning rate decay. Default: 0.1</param>
    /// <param name="minLearningRate">Minimum learning rate floor. Default: 0</param>
    /// <exception cref="ArgumentException">Thrown when milestones is empty or not in increasing order.</exception>
    public MultiStepLRScheduler(
        double baseLearningRate,
        int[] milestones,
        double gamma = 0.1,
        double minLearningRate = 0.0)
        : base(baseLearningRate, minLearningRate)
    {
        if (milestones == null || milestones.Length == 0)
            throw new ArgumentException("Milestones cannot be null or empty.", nameof(milestones));
        if (gamma <= 0 || gamma > 1)
            throw new ArgumentException("Gamma must be in (0, 1].", nameof(gamma));

        // Validate milestones are in increasing order
        for (int i = 1; i < milestones.Length; i++)
        {
            if (milestones[i] <= milestones[i - 1])
                throw new ArgumentException("Milestones must be in strictly increasing order.", nameof(milestones));
        }

        _milestones = milestones.ToArray();
        _gamma = gamma;
    }

    /// <summary>
    /// Gets the milestones.
    /// </summary>
    public IReadOnlyList<int> Milestones => _milestones;

    /// <summary>
    /// Gets the multiplicative factor of learning rate decay.
    /// </summary>
    public double Gamma => _gamma;

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        int decayCount = _milestones.Count(m => step >= m);
        return _baseLearningRate * Math.Pow(_gamma, decayCount);
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["milestones"] = _milestones;
        state["gamma"] = _gamma;
        return state;
    }
}
