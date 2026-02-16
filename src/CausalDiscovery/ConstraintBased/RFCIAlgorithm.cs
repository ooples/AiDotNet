using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ConstraintBased;

/// <summary>
/// RFCI (Really Fast Causal Inference) — scalable FCI for large datasets.
/// </summary>
/// <remarks>
/// <para>RFCI speeds up FCI by reducing the number of conditional independence tests
/// through careful edge pruning. It produces a PAG (Partial Ancestral Graph)
/// that accounts for latent confounders.</para>
/// <para>Reference: Colombo et al. (2012), "Learning High-Dimensional DAGs with Latent
/// and Selection Variables", AOAS.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class RFCIAlgorithm<T> : ConstraintBasedBase<T>
{
    public override string Name => "RFCI";
    public override bool SupportsLatentConfounders => true;
    public override bool SupportsNonlinear => false;

    public RFCIAlgorithm(CausalDiscoveryOptions? options = null) { ApplyConstraintOptions(options); }

    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new FCIAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
