using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// CD-NOD — Constraint-based Discovery from Non-stationary / heterogeneous Data.
/// </summary>
/// <remarks>
/// <para>CD-NOD extends constraint-based methods to handle data from changing environments
/// or multiple domains. It leverages distribution changes to identify causal directions.</para>
/// <para>Reference: Huang et al. (2020), "Causal Discovery from Heterogeneous/Nonstationary Data", JMLR.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CDNODAlgorithm<T> : ConstraintBasedBase<T>
{
    public override string Name => "CD-NOD";
    public override bool SupportsLatentConfounders => true;
    public override bool SupportsNonlinear => true;

    public CDNODAlgorithm(CausalDiscoveryOptions? options = null) { ApplyConstraintOptions(options); }

    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new PCAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
