using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.StoppingCriteria;

/// <summary>
/// Stopping criterion based on time budget exhaustion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This criterion stops learning when the total time
/// spent on active learning exceeds a specified budget. Useful when you have
/// time constraints on annotation or training.</para>
///
/// <para><b>When to Use:</b></para>
/// <list type="bullet">
/// <item><description>Running within a time-limited job or service</description></item>
/// <item><description>Need predictable completion times</description></item>
/// <item><description>Annotation involves human time that must be scheduled</description></item>
/// </list>
///
/// <para><b>Considerations:</b></para>
/// <list type="bullet">
/// <item><description>Combine with budget criterion for more control</description></item>
/// <item><description>Consider annotation time vs. training time separately</description></item>
/// </list>
/// </remarks>
public class TimeBudgetCriterion<T> : IStoppingCriterion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly TimeSpan _maxTime;

    /// <inheritdoc/>
    public string Name => "Time Budget";

    /// <inheritdoc/>
    public string Description => $"Stops when elapsed time exceeds {_maxTime.TotalMinutes:F1} minutes";

    /// <summary>
    /// Gets the maximum allowed time.
    /// </summary>
    public TimeSpan MaxTime => _maxTime;

    /// <summary>
    /// Initializes a new TimeBudget criterion with a 1-hour default.
    /// </summary>
    public TimeBudgetCriterion()
        : this(TimeSpan.FromHours(1))
    {
    }

    /// <summary>
    /// Initializes a new TimeBudget criterion with specified time limit.
    /// </summary>
    /// <param name="maxTime">Maximum time allowed for active learning.</param>
    public TimeBudgetCriterion(TimeSpan maxTime)
    {
        _maxTime = maxTime.TotalSeconds > 0 ? maxTime : TimeSpan.FromHours(1);
    }

    /// <inheritdoc/>
    public bool ShouldStop(ActiveLearningContext<T> context)
    {
        // Check context's max time if set, otherwise use our limit
        var effectiveMax = context.MaxTime ?? _maxTime;
        return context.ElapsedTime >= effectiveMax;
    }

    /// <inheritdoc/>
    public T GetProgress(ActiveLearningContext<T> context)
    {
        var effectiveMax = context.MaxTime ?? _maxTime;
        if (effectiveMax.TotalSeconds <= 0)
        {
            return NumOps.One;
        }

        var progress = context.ElapsedTime.TotalSeconds / effectiveMax.TotalSeconds;
        return NumOps.FromDouble(Math.Min(1.0, progress));
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // Stateless criterion - no reset needed
        // Time is tracked in context, not in the criterion
    }
}
