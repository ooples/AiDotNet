using AiDotNet.Models.Options;
namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// PNL (Post-Nonlinear Causal Model) — Y = g(f(X) + N).
/// </summary>
/// <remarks>
/// <para>
/// This is a baseline implementation that delegates to ANM.
/// A full PNL implementation with post-nonlinear noise testing is planned.
/// </para>
/// <para>Reference: Zhang and Hyvarinen (2009), "On the Identifiability of the
/// Post-Nonlinear Causal Model", UAI.</para>
/// </remarks>
internal class PNLAlgorithm<T> : FunctionalBase<T>
{
    private readonly CausalDiscoveryOptions? _options;
    public override string Name => "PNL";
    public override bool SupportsNonlinear => false;
    public PNLAlgorithm(CausalDiscoveryOptions? options = null)
    {
        _options = options;

    }
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data) =>
        new ANMAlgorithm<T>(_options).DiscoverStructure(data).AdjacencyMatrix;
}
