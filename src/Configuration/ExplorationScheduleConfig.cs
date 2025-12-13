using AiDotNet.Helpers;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration for exploration schedule (epsilon decay for epsilon-greedy).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> The agent needs to explore early in training (try random actions)
/// but exploit more later (use learned policy). This schedule controls that transition.
/// </remarks>
public class ExplorationScheduleConfig<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new instance with default values.
    /// </summary>
    public ExplorationScheduleConfig()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        InitialEpsilon = _numOps.FromDouble(1.0);
        FinalEpsilon = _numOps.FromDouble(0.01);
    }

    /// <summary>
    /// Initial exploration rate (1.0 = fully random).
    /// </summary>
    public T InitialEpsilon { get; set; }

    /// <summary>
    /// Final exploration rate (0.01 = mostly learned policy).
    /// </summary>
    public T FinalEpsilon { get; set; }

    /// <summary>
    /// Number of steps over which to decay from initial to final epsilon.
    /// </summary>
    public int DecaySteps { get; set; } = 100000;

    /// <summary>
    /// Type of decay schedule.
    /// </summary>
    public ExplorationDecayType DecayType { get; set; } = ExplorationDecayType.Linear;
}
