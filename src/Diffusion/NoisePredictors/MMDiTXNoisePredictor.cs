using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Extended Multimodal Diffusion Transformer (MMDiT-X) noise predictor for the
/// Stable Diffusion 3.5 architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MMDiT-X is the SD3.5 evolution of MMDiT (Esser et al. 2024): dual image/text
/// streams with joint (concatenated) self-attention, separate per-stream MLPs,
/// adaptive-LayerNorm timestep modulation, and QK-normalization. It is realized on
/// the faithful <see cref="MMDiTNoisePredictor{T}"/> dual-stream backbone at the
/// SD3.5 scale selected by <see cref="MMDiTXVariant"/> (Medium: hidden 2048 / 24
/// blocks / 16 heads; Large &amp; LargeTurbo: hidden 2560 / 38 blocks / 20 heads).
/// This replaces the previous Dense-only placeholder that had no attention, no
/// timestep conditioning, and ignored the text context.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the transformer behind Stable Diffusion 3.5.
/// Image patches and text tokens look at each other in one shared attention step,
/// and the diffusion timestep steers every layer. Pick a size with the variant.
/// </para>
/// <para>
/// Reference: Esser et al., "Scaling Rectified Flow Transformers for
/// High-Resolution Image Synthesis", ICML 2024 (arXiv:2403.03206).
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Denoising)]
[ModelTask(ModelTask.TextToImage)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", "https://arxiv.org/abs/2403.03206")]
public class MMDiTXNoisePredictor<T> : MMDiTNoisePredictor<T>
{
    // Retained for a type-correct Clone().
    private readonly MMDiTXVariant _variant;
    private readonly int _mmxInputChannels;
    private readonly int _mmxPatchSize;
    private readonly int _mmxContextDim;
    private readonly int? _mmxSeed;
    private readonly int _mmxHiddenOverride;
    private readonly int _mmxLayersOverride;
    private readonly int _mmxHeadsOverride;

    /// <summary>
    /// Initializes a new MMDiT-X (SD3.5) predictor on the faithful MMDiT
    /// dual-stream backbone at the scale selected by <paramref name="variant"/>.
    /// </summary>
    /// <param name="variant">SD3.5 size variant (default: Medium).</param>
    /// <param name="inputChannels">Latent channels (default: 16).</param>
    /// <param name="patchSize">Patch size (default: 2).</param>
    /// <param name="contextDim">Text-conditioning dimension (default: 4096).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <param name="hiddenSizeOverride">Override the variant hidden size (0 = use variant).</param>
    /// <param name="numLayersOverride">Override the variant joint-block count (0 = use variant).</param>
    /// <param name="numHeadsOverride">Override the variant head count (0 = use variant).</param>
    public MMDiTXNoisePredictor(
        MMDiTXVariant variant = MMDiTXVariant.Medium,
        int inputChannels = 16,
        int patchSize = 2,
        int contextDim = 4096,
        int? seed = null,
        int hiddenSizeOverride = 0,
        int numLayersOverride = 0,
        int numHeadsOverride = 0)
        : base(
            inputChannels: inputChannels,
            hiddenSize: hiddenSizeOverride > 0 ? hiddenSizeOverride : GetHiddenSize(variant),
            numJointLayers: numLayersOverride > 0 ? numLayersOverride : GetNumLayers(variant),
            numSingleLayers: 0,
            numHeads: numHeadsOverride > 0 ? numHeadsOverride : GetNumHeads(variant),
            patchSize: patchSize,
            contextDim: contextDim,
            seed: seed)
    {
        _variant = variant;
        _mmxInputChannels = inputChannels;
        _mmxPatchSize = patchSize;
        _mmxContextDim = contextDim;
        _mmxSeed = seed;
        _mmxHiddenOverride = hiddenSizeOverride;
        _mmxLayersOverride = numLayersOverride;
        _mmxHeadsOverride = numHeadsOverride;
    }

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new MMDiTXNoisePredictor<T>(
            _variant, _mmxInputChannels, _mmxPatchSize, _mmxContextDim, _mmxSeed,
            _mmxHiddenOverride, _mmxLayersOverride, _mmxHeadsOverride);
        // #1706: probe-materialize the clone through the forward path, THEN copy weights — the same
        // pattern the base MMDiTNoisePredictor.Clone uses. The previous TryShareParametersFrom /
        // SetParameters path copied onto unmaterialized lazy layers, which then re-RNG-initialized
        // on the clone's first real forward and diverged from the source (HiDream
        // Clone_ShouldProduceIdenticalOutput, maxDiff ~7e2).
        ProbeMaterializeAndCopyInto(clone);
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    // SD3.5 variant dimensions. hidden % heads == 0 for each (2048%16=128, 2560%20=128).
    private static int GetHiddenSize(MMDiTXVariant variant) => variant switch
    {
        MMDiTXVariant.Medium => 2048,
        MMDiTXVariant.LargeTurbo => 2560,
        _ => 2560
    };

    private static int GetNumLayers(MMDiTXVariant variant) => variant switch
    {
        MMDiTXVariant.Medium => 24,
        _ => 38
    };

    private static int GetNumHeads(MMDiTXVariant variant) => variant switch
    {
        MMDiTXVariant.Medium => 16,
        _ => 20
    };
}
