using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// IAMB (Incremental Association Markov Blanket) — efficient Markov blanket discovery.
/// </summary>
/// <remarks>
/// <para>IAMB finds the Markov blanket of each variable using forward selection (add variables
/// that increase association) and backward pruning (remove false positives).</para>
/// <para>Reference: Tsamardinos et al. (2003), "Algorithms for Large Scale Markov Blanket Discovery".</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class IAMBAlgorithm<T> : ConstraintBasedBase<T>
{
    public override string Name => "IAMB";
    public override bool SupportsLatentConfounders => false;
    public override bool SupportsNonlinear => false;

    public IAMBAlgorithm(CausalDiscoveryOptions? options = null) { ApplyConstraintOptions(options); }

    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new MarkovBlanketAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
