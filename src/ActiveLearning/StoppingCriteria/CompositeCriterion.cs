using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.StoppingCriteria;

/// <summary>
/// Composite stopping criterion that combines multiple criteria.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Sometimes you want to stop based on multiple conditions.
/// For example, stop if you run out of budget OR if performance plateaus. This composite
/// criterion lets you combine multiple criteria with AND/OR logic.</para>
///
/// <para><b>Combination Modes:</b></para>
/// <list type="bullet">
/// <item><description><b>Any (OR):</b> Stop if ANY criterion says to stop</description></item>
/// <item><description><b>All (AND):</b> Stop only if ALL criteria say to stop</description></item>
/// </list>
///
/// <para><b>Common Combinations:</b></para>
/// <list type="bullet">
/// <item><description>Budget OR Plateau: Stop when budget exhausted or performance plateaus</description></item>
/// <item><description>Budget AND Convergence: Stop only after both conditions met</description></item>
/// <item><description>Time OR (Budget AND Plateau): Complex nested logic</description></item>
/// </list>
///
/// <para><b>Example Usage:</b></para>
/// <code>
/// var composite = new CompositeCriterion&lt;double&gt;(CombinationMode.Any);
/// composite.AddCriterion(new BudgetExhaustedCriterion&lt;double&gt;(100));
/// composite.AddCriterion(new PerformancePlateauCriterion&lt;double&gt;());
/// composite.AddCriterion(new TimeBudgetCriterion&lt;double&gt;(TimeSpan.FromHours(2)));
/// </code>
/// </remarks>
public class CompositeCriterion<T> : ICompositeCriterion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly List<IStoppingCriterion<T>> _criteria;
    private readonly CombinationMode _mode;

    /// <inheritdoc/>
    public string Name => _mode == CombinationMode.Any
        ? $"Any of {_criteria.Count} criteria"
        : $"All of {_criteria.Count} criteria";

    /// <inheritdoc/>
    public string Description
    {
        get
        {
            if (_criteria.Count == 0)
            {
                return "No criteria configured";
            }

            var criteriaNames = string.Join(_mode == CombinationMode.Any ? " OR " : " AND ",
                _criteria.Select(c => c.Name));
            return $"Stops when: {criteriaNames}";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<IStoppingCriterion<T>> Criteria => _criteria.AsReadOnly();

    /// <summary>
    /// Gets the combination mode for this composite.
    /// </summary>
    public CombinationMode Mode => _mode;

    /// <summary>
    /// Initializes a new empty CompositeCriterion with Any mode.
    /// </summary>
    public CompositeCriterion()
        : this(CombinationMode.Any)
    {
    }

    /// <summary>
    /// Initializes a new empty CompositeCriterion with specified mode.
    /// </summary>
    /// <param name="mode">How to combine criteria (Any = OR, All = AND).</param>
    public CompositeCriterion(CombinationMode mode)
    {
        _mode = mode;
        _criteria = new List<IStoppingCriterion<T>>();
    }

    /// <summary>
    /// Initializes a CompositeCriterion with specified criteria and mode.
    /// </summary>
    /// <param name="criteria">Initial criteria to include.</param>
    /// <param name="mode">How to combine criteria.</param>
    public CompositeCriterion(IEnumerable<IStoppingCriterion<T>> criteria, CombinationMode mode = CombinationMode.Any)
    {
        _mode = mode;
        _criteria = criteria.ToList();
    }

    /// <inheritdoc/>
    public bool ShouldStop(ActiveLearningContext<T> context)
    {
        if (_criteria.Count == 0)
        {
            return false; // No criteria = never stop
        }

        return _mode == CombinationMode.Any
            ? _criteria.Any(c => c.ShouldStop(context))
            : _criteria.All(c => c.ShouldStop(context));
    }

    /// <inheritdoc/>
    public T GetProgress(ActiveLearningContext<T> context)
    {
        if (_criteria.Count == 0)
        {
            return NumOps.Zero;
        }

        var progressValues = _criteria.Select(c => NumOps.ToDouble(c.GetProgress(context))).ToList();

        return _mode == CombinationMode.Any
            ? NumOps.FromDouble(progressValues.Max()) // Max for OR (any one being 1 means done)
            : NumOps.FromDouble(progressValues.Min()); // Min for AND (all must be 1 to be done)
    }

    /// <inheritdoc/>
    public void Reset()
    {
        foreach (var criterion in _criteria)
        {
            criterion.Reset();
        }
    }

    /// <inheritdoc/>
    public void AddCriterion(IStoppingCriterion<T> criterion)
    {
        if (criterion != null && !_criteria.Contains(criterion))
        {
            _criteria.Add(criterion);
        }
    }

    /// <inheritdoc/>
    public bool RemoveCriterion(IStoppingCriterion<T> criterion)
    {
        return _criteria.Remove(criterion);
    }

    /// <summary>
    /// Creates a composite criterion that stops when ANY of the given criteria is met.
    /// </summary>
    /// <param name="criteria">Criteria to combine with OR logic.</param>
    /// <returns>A new composite criterion.</returns>
    public static CompositeCriterion<T> Any(params IStoppingCriterion<T>[] criteria)
    {
        return new CompositeCriterion<T>(criteria, CombinationMode.Any);
    }

    /// <summary>
    /// Creates a composite criterion that stops when ALL of the given criteria are met.
    /// </summary>
    /// <param name="criteria">Criteria to combine with AND logic.</param>
    /// <returns>A new composite criterion.</returns>
    public static CompositeCriterion<T> All(params IStoppingCriterion<T>[] criteria)
    {
        return new CompositeCriterion<T>(criteria, CombinationMode.All);
    }
}

/// <summary>
/// Mode for combining multiple stopping criteria.
/// </summary>
public enum CombinationMode
{
    /// <summary>
    /// Stop if ANY criterion is met (OR logic).
    /// </summary>
    Any,

    /// <summary>
    /// Stop only if ALL criteria are met (AND logic).
    /// </summary>
    All
}
