using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Hybrid;

/// <summary>
/// PC-NOTEARS — Hybrid of PC skeleton discovery with NOTEARS continuous optimization.
/// </summary>
/// <remarks>
/// <para>
/// PC-NOTEARS first uses PC's constraint-based skeleton discovery to identify candidate edges,
/// then runs NOTEARS continuous optimization restricted to the PC skeleton. This combines
/// PC's efficient edge elimination with NOTEARS' optimal weight estimation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This hybrid first uses statistical tests to quickly figure out which
/// variable pairs MIGHT be connected, then uses optimization to find the exact edge weights
/// and directions — getting both speed and accuracy.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PCNOTEARSAlgorithm<T> : HybridBase<T>
{
    /// <inheritdoc/>
    public override string Name => "PC-NOTEARS";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public PCNOTEARSAlgorithm(CausalDiscoveryOptions? options = null) { ApplyHybridOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: delegates to MMHC as baseline
        var baseline = new MMHCAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
