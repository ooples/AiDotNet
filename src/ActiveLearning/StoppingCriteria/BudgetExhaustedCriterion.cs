using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning.StoppingCriteria;

/// <summary>
/// Stopping criterion based on labeling budget exhaustion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This is the most basic stopping criterion - it simply
/// checks if you've labeled as many samples as your budget allows.</para>
///
/// <para><b>When to Use:</b></para>
/// <list type="bullet">
/// <item><description>You have a fixed annotation budget</description></item>
/// <item><description>Want to compare different strategies with same budget</description></item>
/// <item><description>As a fallback criterion to ensure learning terminates</description></item>
/// </list>
///
/// <para><b>Implementation:</b> Compares TotalLabeled against MaxBudget in the context.</para>
/// </remarks>
public class BudgetExhaustedCriterion<T> : IStoppingCriterion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int? _overrideBudget;

    /// <inheritdoc/>
    public string Name => "Budget Exhausted";

    /// <inheritdoc/>
    public string Description =>
        _overrideBudget.HasValue
            ? $"Stops when {_overrideBudget.Value} samples have been labeled"
            : "Stops when the maximum labeling budget is exhausted";

    /// <summary>
    /// Initializes a new BudgetExhausted criterion using context budget.
    /// </summary>
    public BudgetExhaustedCriterion()
        : this(null)
    {
    }

    /// <summary>
    /// Initializes a new BudgetExhausted criterion with a specific budget.
    /// </summary>
    /// <param name="budget">The labeling budget (null to use context budget).</param>
    public BudgetExhaustedCriterion(int? budget)
    {
        _overrideBudget = budget;
    }

    /// <inheritdoc/>
    public bool ShouldStop(ActiveLearningContext<T> context)
    {
        var budget = _overrideBudget ?? context.MaxBudget;
        return context.TotalLabeled >= budget;
    }

    /// <inheritdoc/>
    public T GetProgress(ActiveLearningContext<T> context)
    {
        var budget = _overrideBudget ?? context.MaxBudget;
        if (budget <= 0)
        {
            return NumOps.One;
        }

        var progress = (double)context.TotalLabeled / budget;
        return NumOps.FromDouble(Math.Min(1.0, progress));
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // Stateless criterion - no reset needed
    }
}
