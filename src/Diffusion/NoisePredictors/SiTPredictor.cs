using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Scalable Interpolant Transformer (SiT) noise predictor for flow-based diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SiT (Ma et al., ECCV 2024) deliberately reuses the <b>DiT backbone unchanged</b>
/// — the paper's §3 states it adopts the DiT architecture and isolates its
/// contribution to the <i>training/sampling</i> side (a learnable interpolant
/// between data and noise that unifies score-based diffusion and flow matching).
/// The network itself is the identical patchify → adaptive-LayerNorm
/// self-attention transformer stack → final AdaLN layer. We therefore realize SiT
/// faithfully as a configured <see cref="DiTNoisePredictor{T}"/>: the interpolant
/// difference lives in the diffusion model's scheduler (flow-matching /
/// stochastic-interpolant), not in this noise-prediction network.
/// </para>
/// <para>
/// <b>For Beginners:</b> SiT is "DiT with a more flexible noise process." The neural
/// network that predicts the noise is exactly the DiT transformer; what changes is
/// the math used to add/remove noise during training and sampling, which the
/// scheduler handles. Reusing the DiT backbone here means SiT gets real
/// self-attention, real timestep conditioning (AdaLN), and the MLP expansion of a
/// proper transformer — not a placeholder.
/// </para>
/// <para>
/// Reference: Ma et al., "SiT: Exploring Flow and Diffusion-based Generative Models
/// with Scalable Interpolant Transformers", ECCV 2024 (arXiv:2401.08740).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var predictor = new SiTPredictor&lt;float&gt;(inputChannels: 4, hiddenSize: 1152, numLayers: 28, numHeads: 16);
/// var noisyLatent = Tensor&lt;float&gt;.Random(new[] { 1, 4, 32, 32 });
/// var predicted = predictor.PredictNoise(noisyLatent, timestep: 500);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Denoising)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers", "https://arxiv.org/abs/2401.08740")]
public class SiTPredictor<T> : DiTNoisePredictor<T>
{
    // Construction config retained so Clone() can rebuild a SiT (not a bare DiT).
    private readonly int _sitInputChannels;
    private readonly int _sitHiddenSize;
    private readonly int _sitNumLayers;
    private readonly int _sitNumHeads;
    private readonly int? _sitSeed;

    /// <summary>
    /// Initializes a new SiT predictor on the faithful DiT backbone. Layers are
    /// lazily allocated on first use (inherited from <see cref="DiTNoisePredictor{T}"/>)
    /// so foundation-scale configs construct in O(1) without allocating weights.
    /// </summary>
    /// <param name="inputChannels">Latent channels (default: 4).</param>
    /// <param name="hiddenSize">Transformer hidden dimension (default: 1152, DiT-XL).</param>
    /// <param name="numLayers">Number of transformer blocks (default: 28, DiT-XL).</param>
    /// <param name="numHeads">Number of attention heads (default: 16).</param>
    /// <param name="seed">Optional random seed for deterministic initialization.</param>
    public SiTPredictor(
        int inputChannels = 4,
        int hiddenSize = 1152,
        int numLayers = 28,
        int numHeads = 16,
        int? seed = null)
        : base(
            inputChannels: inputChannels,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            numHeads: numHeads,
            patchSize: 2,
            contextDim: hiddenSize,
            seed: seed)
    {
        _sitInputChannels = inputChannels;
        _sitHiddenSize = hiddenSize;
        _sitNumLayers = numLayers;
        _sitNumHeads = numHeads;
        _sitSeed = seed;
    }

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new SiTPredictor<T>(_sitInputChannels, _sitHiddenSize, _sitNumLayers, _sitNumHeads, _sitSeed);
        // #1711: DiT LazyDense weights resolve via the FORWARD path, so a naive
        // SetParameters(GetParameters()) / COW clone re-RNG-initializes on its first forward and
        // diverges from the source. Use the base DiT clone semantics (probe-forward + copy), which
        // also no-ops cleanly when the source has no materialized weights.
        ProbeMaterializeAndCopyInto(clone);
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
}
