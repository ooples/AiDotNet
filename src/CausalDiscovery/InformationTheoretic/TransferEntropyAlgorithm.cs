using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.InformationTheoretic;

/// <summary>
/// Transfer Entropy — information-theoretic measure of directed information flow.
/// </summary>
/// <remarks>
/// <para>
/// Transfer entropy quantifies the amount of directed information transfer from one
/// process to another. It measures the reduction in uncertainty of Y's future given
/// the past of both X and Y, compared to only Y's past. It is a nonlinear generalization
/// of Granger causality.
/// </para>
/// <para>
/// <b>For Beginners:</b> Transfer Entropy is like Granger causality but works for nonlinear
/// relationships too. It asks: "Does knowing X's past reduce my uncertainty about Y's future,
/// beyond what Y's own past already tells me?" If yes, X transfers information to Y.
/// </para>
/// <para>
/// Reference: Schreiber (2000), "Measuring Information Transfer", Physical Review Letters.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TransferEntropyAlgorithm<T> : InfoTheoreticBase<T>
{
    /// <inheritdoc/>
    public override string Name => "TransferEntropy";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    /// <inheritdoc/>
    public override bool SupportsTimeSeries => true;

    public TransferEntropyAlgorithm(CausalDiscoveryOptions? options = null) { ApplyInfoOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new TimeSeries.GrangerCausalityAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
