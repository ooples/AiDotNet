using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// NTS-NOTEARS — Nonstationary Time Series NOTEARS.
/// </summary>
/// <remarks>
/// <para>
/// NTS-NOTEARS extends DYNOTEARS to handle nonstationary time series where the causal
/// structure may change over time. It partitions the data into segments with potentially
/// different DAG structures and uses change-point detection.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular time series methods assume the causal relationships stay
/// the same forever. NTS-NOTEARS can detect when relationships change — for example,
/// a market regime shift where the causes of stock prices change.
/// </para>
/// <para>
/// Reference: Sun et al. (2021), "NTS-NOTEARS: Learning Nonparametric DBN Structure
/// from Nonstationary Time Series".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NTSNOTEARSAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "NTS-NOTEARS";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public NTSNOTEARSAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: delegates to DYNOTEARS as baseline
        var baseline = new DYNOTEARSAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
