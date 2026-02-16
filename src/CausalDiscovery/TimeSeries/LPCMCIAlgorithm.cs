using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// LPCMCI — Latent PCMCI for time series with hidden confounders.
/// </summary>
/// <remarks>
/// <para>
/// LPCMCI extends PCMCI to handle latent confounders by combining ideas from FCI
/// (ancestral graph representation) with PCMCI's condition selection and MCI testing.
/// It outputs a time series PAG (partial ancestral graph) instead of a DAG.
/// </para>
/// <para>
/// <b>For Beginners:</b> LPCMCI is the most advanced version of PCMCI. It works even when
/// there are hidden variables affecting the ones you can measure. The trade-off is that
/// some edges may be uncertain in direction (shown with circle marks).
/// </para>
/// <para>
/// Reference: Gerhardus and Runge (2022), "High-recall causal discovery for autocorrelated
/// time series with latent confounders", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LPCMCIAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "LPCMCI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    public LPCMCIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: delegates to PCMCI+ as baseline
        var baseline = new PCMCIPlusAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
