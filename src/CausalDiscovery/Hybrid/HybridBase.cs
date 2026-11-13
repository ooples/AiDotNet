using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.CausalDiscovery.Hybrid;

/// <summary>
/// Base class for hybrid causal discovery algorithms that combine constraint-based and score-based methods.
/// </summary>
/// <remarks>
/// <para>
/// Hybrid methods first use constraint-based tests to restrict the search space (e.g., finding
/// candidate parents via conditional independence tests), then use score-based search to find the
/// optimal DAG within the restricted space.
/// </para>
/// <para>
/// <b>For Beginners:</b> Hybrid algorithms get the best of both worlds. Constraint-based methods
/// are good at ruling out edges quickly, while score-based methods are good at finding the best
/// structure among remaining options. Combining them gives faster AND more accurate results.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class HybridBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.Hybrid;

    /// <summary>
    /// Significance level for constraint-based phase.
    /// </summary>
    protected double Alpha { get; set; } = 0.05;

    /// <summary>
    /// Maximum number of parents per variable (restricts search space).
    /// </summary>
    protected int MaxParents { get; set; } = 5;

    /// <summary>
    /// Applies options from CausalDiscoveryOptions.
    /// </summary>
    protected void ApplyHybridOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        if (options.SignificanceLevel.HasValue) Alpha = options.SignificanceLevel.Value;
        if (options.MaxParents.HasValue) MaxParents = options.MaxParents.Value;
    }

    /// <summary>
    /// Computes Pearson correlation between two columns of data.
    /// </summary>
    protected double ComputeCorrelation(Matrix<T> data, int col1, int col2)
    {
        return NumOps.ToDouble(CausalTensorHelpers.PearsonCorrelation(data, col1, col2));
    }

    /// <summary>
    /// Computes BIC score for a variable given its parents.
    /// </summary>
    protected double ComputeBIC(Matrix<T> data, int variable, List<int> parents)
    {
        int n = data.Rows;
        int p = parents.Count;

        // No parents: the RSS about the mean, one parameter; otherwise k slopes + intercept.
        T rss = p == 0
            ? CausalTensorHelpers.CenteredSumOfSquares(data, variable)
            : CausalTensorHelpers.InterceptOlsRss(data, variable, parents, 1e-6);
        T fit = NumOps.Multiply(NumOps.FromDouble(n),
            NumOps.Log(NumOps.Add(NumOps.Divide(rss, NumOps.FromDouble(n)), NumOps.FromDouble(1e-15))));
        return NumOps.ToDouble(NumOps.Add(fit, NumOps.FromDouble((p + 1) * Math.Log(n))));
    }
}
