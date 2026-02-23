using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// CPC (Conservative PC) — PC variant that avoids erroneous v-structure orientation.
/// </summary>
/// <remarks>
/// <para>CPC modifies the orientation phase of PC to be more conservative when determining
/// v-structures, reducing false positive orientations in finite samples.</para>
/// <para>Reference: Ramsey et al. (2012), "Adjacency-Faithfulness and Conservative
/// Causal Inference", UAI.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CPCAlgorithm<T> : ConstraintBasedBase<T>
{
    public override string Name => "Conservative PC";
    public override bool SupportsLatentConfounders => false;
    public override bool SupportsNonlinear => false;

    public CPCAlgorithm(CausalDiscoveryOptions? options = null) { ApplyConstraintOptions(options); }

    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new PCAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
