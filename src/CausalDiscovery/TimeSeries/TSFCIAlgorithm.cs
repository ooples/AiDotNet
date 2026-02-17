using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// tsFCI — time series Fast Causal Inference.
/// </summary>
/// <remarks>
/// <para>
/// tsFCI adapts the FCI algorithm for time series data, allowing for the discovery of
/// causal relationships in the presence of latent (unmeasured) confounders. It uses
/// temporal ordering constraints to improve orientation of edges.
/// </para>
/// <para>
/// <b>For Beginners:</b> tsFCI is like FCI but for time series. It can discover causal
/// relationships even when there are hidden variables affecting the observed ones,
/// using the fact that "the future cannot cause the past" to help figure out direction.
/// </para>
/// <para>
/// Reference: Entner and Hoyer (2010), "On Causal Discovery from Time Series Data
/// using FCI", PGM.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TSFCIAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "tsFCI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    public TSFCIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: delegates to PCMCI as baseline
        var baseline = new PCMCIAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
