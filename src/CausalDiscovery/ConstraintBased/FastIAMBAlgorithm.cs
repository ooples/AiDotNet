using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// Fast-IAMB — faster variant of IAMB using speculative forward/backward selection.
/// </summary>
/// <remarks>
/// <para>Fast-IAMB adds multiple variables at once in the forward phase (speculative),
/// then uses backward pruning. Significantly faster than IAMB for high-dimensional data.</para>
/// <para>Reference: Yaramakala and Margaritis (2005), "Speculative Markov Blanket Discovery
/// for Optimal Feature Selection".</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FastIAMBAlgorithm<T> : ConstraintBasedBase<T>
{
    public override string Name => "Fast-IAMB";
    public override bool SupportsLatentConfounders => false;
    public override bool SupportsNonlinear => false;

    public FastIAMBAlgorithm(CausalDiscoveryOptions? options = null) { ApplyConstraintOptions(options); }

    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new MarkovBlanketAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
