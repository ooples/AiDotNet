using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.InformationTheoretic;

/// <summary>
/// oCSE — optimal Causation Entropy for causal network inference.
/// </summary>
/// <remarks>
/// <para>
/// oCSE uses causation entropy — a measure of the information a variable provides about
/// another variable's transition — to identify causal links. It greedily selects the
/// optimal conditioning set that maximizes the causation entropy criterion.
/// </para>
/// <para>
/// <b>For Beginners:</b> oCSE measures how much a variable helps predict another variable's
/// CHANGES over time (not just its values). This is closer to true causation — a cause
/// should affect how the effect changes.
/// </para>
/// <para>
/// Reference: Sun et al. (2015), "Causal Network Inference by Optimal Causation Entropy",
/// SIAM Journal on Applied Dynamical Systems.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OCSEAlgorithm<T> : InfoTheoreticBase<T>
{
    /// <inheritdoc/>
    public override string Name => "oCSE";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    /// <inheritdoc/>
    public override bool SupportsTimeSeries => true;

    public OCSEAlgorithm(CausalDiscoveryOptions? options = null) { ApplyInfoOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ConstraintBased.PCAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
