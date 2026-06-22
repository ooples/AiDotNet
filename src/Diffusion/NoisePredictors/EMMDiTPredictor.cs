using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// E-MMDiT (Efficient Multimodal Diffusion Transformer) noise predictor — a
/// compact configuration of the MMDiT architecture (Stable Diffusion 3,
/// Esser et al. 2024) for parameter-efficient inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// E-MMDiT keeps the full MMDiT block — dual image/text streams with joint
/// (concatenated) self-attention, separate per-stream MLPs, adaptive-LayerNorm
/// modulation from the timestep + pooled-text conditioning, and QK-normalization
/// — but at a smaller width/depth (hidden 1024, 12 joint blocks, 16 heads) than
/// the full SD3 MMDiT. It is therefore realized as a configured
/// <see cref="MMDiTNoisePredictor{T}"/>: the architecture is identical to MMDiT,
/// only the scale differs. (The earlier Dense-only placeholder had no attention
/// or timestep conditioning and was not the MMDiT architecture it claimed.)
/// </para>
/// <para>
/// <b>For Beginners:</b> This is a "small but real" version of the Stable
/// Diffusion 3 transformer. It does exactly what the big one does — image and
/// text tokens attend to each other jointly, and the timestep controls every
/// layer — just with fewer/narrower layers so it runs lighter.
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
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", "https://arxiv.org/abs/2403.03206")]
public class EMMDiTPredictor<T> : MMDiTNoisePredictor<T>
{
    // Compact-MMDiT scale. hidden % heads == 0 (1024 % 16 = 64).
    private const int EMMDIT_HIDDEN_SIZE = 1024;
    private const int EMMDIT_NUM_LAYERS = 12;
    private const int EMMDIT_NUM_HEADS = 16;

    // Retained for a type-correct Clone().
    private readonly int _emmInputChannels;
    private readonly int _emmContextDim;
    private readonly int? _emmSeed;

    /// <summary>
    /// Initializes a new E-MMDiT predictor on the faithful MMDiT backbone at the
    /// compact (hidden 1024, 12 joint blocks, 16 heads) configuration.
    /// </summary>
    /// <param name="inputChannels">Latent channels (default: 4).</param>
    /// <param name="contextDim">Text-conditioning dimension (default: 768, CLIP).</param>
    /// <param name="seed">Optional random seed.</param>
    public EMMDiTPredictor(
        int inputChannels = 4,
        int contextDim = 768,
        int? seed = null)
        : base(
            inputChannels: inputChannels,
            hiddenSize: EMMDIT_HIDDEN_SIZE,
            numJointLayers: EMMDIT_NUM_LAYERS,
            numSingleLayers: 0,
            numHeads: EMMDIT_NUM_HEADS,
            contextDim: contextDim,
            seed: seed)
    {
        _emmInputChannels = inputChannels;
        _emmContextDim = contextDim;
        _emmSeed = seed;
    }

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new EMMDiTPredictor<T>(_emmInputChannels, _emmContextDim, _emmSeed);
        if (!clone.TryShareParametersFrom(this)) clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
}
