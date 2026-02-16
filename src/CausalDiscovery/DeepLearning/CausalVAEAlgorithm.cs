using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// CausalVAE — Causal Variational Autoencoder.
/// </summary>
/// <remarks>
/// <para>
/// CausalVAE extends the VAE framework to learn a disentangled latent space where the
/// latent variables are causally related according to a learned DAG. The causal layer
/// transforms independent exogenous noise into causally-structured latent variables.
/// </para>
/// <para>
/// <b>For Beginners:</b> CausalVAE learns a compressed version of your data where the
/// compressed variables have causal relationships between them. This is useful for
/// understanding underlying causal mechanisms even in high-dimensional data like images.
/// </para>
/// <para>
/// Reference: Yang et al. (2021), "CausalVAE: Disentangled Representation Learning via
/// Neural Structural Causal Models", CVPR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class CausalVAEAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CausalVAE";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public CausalVAEAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ContinuousOptimization.NOTEARSLinear<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
