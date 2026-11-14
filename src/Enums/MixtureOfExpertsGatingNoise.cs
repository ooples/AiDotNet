namespace AiDotNet.Enums;

/// <summary>
/// How a <see cref="AiDotNet.NeuralNetworks.Layers.MixtureOfExpertsLayer{T}"/> scales the Gaussian noise
/// it adds to the gating logits during training.
/// </summary>
/// <remarks>
/// <para>
/// Noisy top-k gating (Shazeer et al. 2017, eq. 4) is
/// <c>H(x) = x W_g + StandardNormal() * Softplus(x W_noise)</c>. The noise scale has its own learned
/// weights, so the layer can learn how much to explore per input and per expert.
/// </para>
/// <para><b>For Beginners:</b> keep the default to get the gating described in the paper. Choose
/// <see cref="SoftplusOfRouterLogits"/> only to reproduce models trained with the earlier behaviour.
/// </para>
/// </remarks>
public enum MixtureOfExpertsGatingNoise
{
    /// <summary>Noise scaled by <c>Softplus(x W_noise)</c> with learned, zero-initialized <c>W_noise</c> (paper).</summary>
    LearnedNoiseWeights = 0,

    /// <summary>Noise scaled by <c>Softplus</c> of the router's own logits, the layer's earlier behaviour.</summary>
    SoftplusOfRouterLogits = 1,
}
