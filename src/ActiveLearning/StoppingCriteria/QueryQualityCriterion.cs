using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning.StoppingCriteria;

/// <summary>
/// Stopping criterion based on quality of remaining query candidates.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This criterion stops learning when the informativeness
/// scores of the best remaining samples fall below a threshold. If no informative
/// samples remain, there's little benefit to continuing.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Monitors the query scores (informativeness) over iterations</description></item>
/// <item><description>Tracks the maximum score of selected samples</description></item>
/// <item><description>Stops when best remaining score is below threshold</description></item>
/// </list>
///
/// <para><b>Intuition:</b> Early in active learning, the query strategy finds highly
/// informative samples. As learning progresses, the most informative samples get labeled,
/// and remaining samples become less valuable.</para>
///
/// <para><b>Key Parameters:</b></para>
/// <list type="bullet">
/// <item><description><b>Threshold:</b> Minimum query score to consider continuing</description></item>
/// <item><description><b>Relative Mode:</b> Compare to initial scores rather than absolute</description></item>
/// </list>
/// </remarks>
public class QueryQualityCriterion<T> : IStoppingCriterion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _threshold;
    private readonly bool _useRelativeThreshold;

    private T _initialMaxScore;
    private T _currentMaxScore;
    private bool _initialized;

    /// <inheritdoc/>
    public string Name => "Query Quality";

    /// <inheritdoc/>
    public string Description =>
        _useRelativeThreshold
            ? $"Stops when query quality drops below {NumOps.ToDouble(_threshold):P0} of initial quality"
            : $"Stops when query quality drops below {NumOps.ToDouble(_threshold):F3}";

    /// <summary>
    /// Gets the current maximum query score.
    /// </summary>
    public T CurrentMaxScore => _currentMaxScore;

    /// <summary>
    /// Gets the initial maximum query score (for relative mode).
    /// </summary>
    public T InitialMaxScore => _initialMaxScore;

    /// <summary>
    /// Initializes a new QueryQuality criterion with default parameters.
    /// </summary>
    public QueryQualityCriterion()
        : this(threshold: 0.1, useRelativeThreshold: true)
    {
    }

    /// <summary>
    /// Initializes a new QueryQuality criterion with specified parameters.
    /// </summary>
    /// <param name="threshold">Minimum query score (absolute or relative).</param>
    /// <param name="useRelativeThreshold">Whether to use relative (vs. absolute) threshold.</param>
    public QueryQualityCriterion(double threshold, bool useRelativeThreshold = true)
    {
        _threshold = NumOps.FromDouble(threshold > 0 ? threshold : 0.1);
        _useRelativeThreshold = useRelativeThreshold;

        _initialMaxScore = NumOps.Zero;
        _currentMaxScore = NumOps.One;
        _initialized = false;
    }

    /// <inheritdoc/>
    public bool ShouldStop(ActiveLearningContext<T> context)
    {
        // Need query score history
        if (context.QueryScoreHistory == null || context.QueryScoreHistory.Count == 0)
        {
            return false;
        }

        // Get current max score
        _currentMaxScore = context.QueryScoreHistory[^1];

        // Initialize with first score if needed
        if (!_initialized && context.QueryScoreHistory.Count > 0)
        {
            _initialMaxScore = context.QueryScoreHistory[0];
            _initialized = true;
        }

        // Check against threshold
        if (_useRelativeThreshold)
        {
            // Relative: current / initial < threshold
            if (NumOps.Compare(_initialMaxScore, NumOps.FromDouble(1e-10)) <= 0)
            {
                return false; // Can't compute relative if initial is zero
            }

            T ratio = NumOps.Divide(_currentMaxScore, _initialMaxScore);
            return NumOps.Compare(ratio, _threshold) < 0;
        }
        else
        {
            // Absolute: current < threshold
            return NumOps.Compare(_currentMaxScore, _threshold) < 0;
        }
    }

    /// <inheritdoc/>
    public T GetProgress(ActiveLearningContext<T> context)
    {
        if (!_initialized || NumOps.Compare(_initialMaxScore, NumOps.FromDouble(1e-10)) <= 0)
        {
            return NumOps.Zero;
        }

        if (_useRelativeThreshold)
        {
            // Progress is how much quality has declined
            T ratio = NumOps.Divide(_currentMaxScore, _initialMaxScore);
            // Map: ratio=1.0 -> progress=0, ratio=threshold -> progress=1
            T denominator = NumOps.Subtract(NumOps.One, _threshold);
            if (NumOps.Compare(denominator, NumOps.FromDouble(1e-10)) <= 0)
            {
                return NumOps.One;
            }

            T numerator = NumOps.Subtract(NumOps.One, ratio);
            T progress = NumOps.Divide(numerator, denominator);

            return NumOps.FromDouble(MathHelper.Clamp(NumOps.ToDouble(progress), 0.0, 1.0));
        }
        else
        {
            // Progress based on how close to absolute threshold
            T ratio = NumOps.Divide(_threshold, _currentMaxScore);
            return NumOps.FromDouble(Math.Min(1.0, NumOps.ToDouble(ratio)));
        }
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _initialMaxScore = NumOps.Zero;
        _currentMaxScore = NumOps.One;
        _initialized = false;
    }
}
