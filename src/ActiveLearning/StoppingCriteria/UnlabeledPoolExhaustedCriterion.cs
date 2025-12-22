using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.StoppingCriteria;

/// <summary>
/// Stopping criterion that triggers when the unlabeled pool is exhausted.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This is a simple but important criterion - it stops
/// active learning when there are no more unlabeled samples to query. This is the
/// natural endpoint when all data has been labeled.</para>
///
/// <para><b>When to Use:</b></para>
/// <list type="bullet">
/// <item><description>Always include as a fallback criterion</description></item>
/// <item><description>Small datasets where you might label everything</description></item>
/// <item><description>Stream settings where pool might be exhausted</description></item>
/// </list>
///
/// <para><b>Optional Threshold:</b> Can also stop when pool falls below a minimum
/// size, useful when very small pools aren't worth querying.</para>
/// </remarks>
public class UnlabeledPoolExhaustedCriterion<T> : IStoppingCriterion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _minimumRemaining;
    private int _initialPoolSize;
    private bool _initialized;

    /// <inheritdoc/>
    public string Name => "Unlabeled Pool Exhausted";

    /// <inheritdoc/>
    public string Description =>
        _minimumRemaining > 0
            ? $"Stops when fewer than {_minimumRemaining} unlabeled samples remain"
            : "Stops when all samples have been labeled";

    /// <summary>
    /// Gets the minimum pool size before stopping.
    /// </summary>
    public int MinimumRemaining => _minimumRemaining;

    /// <summary>
    /// Initializes a new UnlabeledPoolExhausted criterion that stops at zero remaining.
    /// </summary>
    public UnlabeledPoolExhaustedCriterion()
        : this(minimumRemaining: 0)
    {
    }

    /// <summary>
    /// Initializes a new UnlabeledPoolExhausted criterion with minimum threshold.
    /// </summary>
    /// <param name="minimumRemaining">Stop when pool size falls below this.</param>
    public UnlabeledPoolExhaustedCriterion(int minimumRemaining)
    {
        _minimumRemaining = minimumRemaining >= 0 ? minimumRemaining : 0;
        _initialPoolSize = 0;
        _initialized = false;
    }

    /// <inheritdoc/>
    public bool ShouldStop(ActiveLearningContext<T> context)
    {
        // Initialize with first seen pool size
        if (!_initialized && context.UnlabeledRemaining > 0)
        {
            _initialPoolSize = context.UnlabeledRemaining + context.TotalLabeled;
            _initialized = true;
        }

        return context.UnlabeledRemaining <= _minimumRemaining;
    }

    /// <inheritdoc/>
    public T GetProgress(ActiveLearningContext<T> context)
    {
        if (!_initialized || _initialPoolSize <= _minimumRemaining)
        {
            if (context.UnlabeledRemaining <= _minimumRemaining)
            {
                return NumOps.One;
            }
            return NumOps.Zero;
        }

        // Progress is fraction of pool that has been labeled
        int effectivePool = _initialPoolSize - _minimumRemaining;
        int labeled = _initialPoolSize - context.UnlabeledRemaining;

        var progress = (double)labeled / effectivePool;
        return NumOps.FromDouble(MathHelper.Clamp(progress, 0.0, 1.0));
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _initialPoolSize = 0;
        _initialized = false;
    }
}
