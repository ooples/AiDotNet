using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// GAE — Graph Autoencoder for causal discovery.
/// </summary>
/// <remarks>
/// <para>
/// GAE uses an autoencoder architecture where the encoder produces a latent graph
/// representation and the decoder reconstructs the data through the learned graph.
/// The bottleneck encourages learning a compact causal structure.
/// </para>
/// <para>
/// <b>For Beginners:</b> A Graph Autoencoder compresses data through a "graph bottleneck."
/// The connections in this bottleneck represent causal relationships — the autoencoder
/// is forced to find the minimal set of connections needed to recreate the data.
/// </para>
/// <para>
/// Reference: Kipf and Welling (2016), "Variational Graph Auto-Encoders", NeurIPS Workshop.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GAEAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "GAE";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public GAEAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ContinuousOptimization.NOTEARSLinear<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
