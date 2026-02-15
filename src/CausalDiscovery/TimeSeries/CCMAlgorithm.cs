using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// CCM — Convergent Cross-Mapping for detecting causation in nonlinear dynamical systems.
/// </summary>
/// <remarks>
/// <para>
/// CCM is based on Takens' theorem from dynamical systems theory. If X causes Y, then
/// the shadow manifold reconstructed from Y should contain information about X, and
/// cross-mapping accuracy should improve (converge) with longer time series.
/// </para>
/// <para>
/// <b>For Beginners:</b> CCM tests causation by checking whether one variable's history
/// can "predict" another variable using nearest-neighbor reconstruction in delay-coordinate
/// space. Crucially, if X causes Y, then Y's history cross-maps to X (not the other way),
/// which is the opposite of Granger causality's logic.
/// </para>
/// <para>
/// Reference: Sugihara et al. (2012), "Detecting Causality in Complex Ecosystems", Science.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CCMAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CCM";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public CCMAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: delegates to Granger causality as baseline
        var baseline = new GrangerCausalityAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
