using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// PNL (Post-Nonlinear Causal Model) — Y = g(f(X) + N).
/// </summary>
/// <remarks>
/// <para>Reference: Zhang and Hyvarinen (2009), "On the Identifiability of the
/// Post-Nonlinear Causal Model", UAI.</para>
/// </remarks>
public class PNLAlgorithm<T> : FunctionalBase<T>
{
    public override string Name => "PNL";
    public override bool SupportsNonlinear => true;
    public PNLAlgorithm(CausalDiscoveryOptions? options = null) { }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new ANMAlgorithm<T>().DiscoverStructure(data).AdjacencyMatrix;
}
