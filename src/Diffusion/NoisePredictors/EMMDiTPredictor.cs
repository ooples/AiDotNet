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
    private readonly int _emmHiddenSizeOverride;
    private readonly int _emmNumLayersOverride;
    private readonly int _emmNumHeadsOverride;

    /// <summary>
    /// Initializes a new E-MMDiT predictor on the faithful MMDiT backbone at the
    /// compact (hidden 1024, 12 joint blocks, 16 heads) configuration.
    /// </summary>
    /// <param name="inputChannels">Latent channels (default: 4).</param>
    /// <param name="contextDim">Text-conditioning dimension (default: 768, CLIP).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <param name="hiddenSizeOverride">Hidden size; 0 keeps the compact configuration's 1024.</param>
    /// <param name="numLayersOverride">Joint blocks; 0 keeps the compact configuration's 12.</param>
    /// <param name="numHeadsOverride">Attention heads; 0 keeps the compact configuration's 16.</param>
    /// <remarks>
    /// The overrides keep the compact configuration the default while letting a caller (or a test) build the same
    /// backbone at another size. Without them the model was fixed at about 460M parameters.
    /// </remarks>
    public EMMDiTPredictor(
        int inputChannels = 4,
        int contextDim = 768,
        int? seed = null,
        int hiddenSizeOverride = 0,
        int numLayersOverride = 0,
        int numHeadsOverride = 0)
        : base(
            inputChannels: inputChannels,
            hiddenSize: ResolveOverride(hiddenSizeOverride, EMMDIT_HIDDEN_SIZE, nameof(hiddenSizeOverride)),
            numJointLayers: ResolveOverride(numLayersOverride, EMMDIT_NUM_LAYERS, nameof(numLayersOverride)),
            numSingleLayers: 0,
            numHeads: ResolveOverride(numHeadsOverride, EMMDIT_NUM_HEADS, nameof(numHeadsOverride)),
            contextDim: contextDim,
            seed: seed)
    {
        _emmInputChannels = inputChannels;
        _emmContextDim = contextDim;
        _emmSeed = seed;
        _emmHiddenSizeOverride = hiddenSizeOverride;
        _emmNumLayersOverride = numLayersOverride;
        _emmNumHeadsOverride = numHeadsOverride;
    }

    /// <summary>
    /// Initializes a new E-MMDiT predictor at the compact configuration: the constructor earlier releases compiled
    /// against, kept so they still bind after an upgrade.
    /// </summary>
    /// <param name="inputChannels">Latent channels.</param>
    /// <param name="contextDim">Text-conditioning dimension.</param>
    /// <param name="seed">Optional random seed.</param>
    public EMMDiTPredictor(int inputChannels, int contextDim, int? seed)
        : this(inputChannels, contextDim, seed, 0, 0, 0)
    {
    }

    // 0 keeps the compact configuration's value; a negative size is a caller error, not "use the default".
    private static int ResolveOverride(int value, int compactDefault, string parameterName) =>
        value < 0
            ? throw new ArgumentOutOfRangeException(parameterName, value, "A size override must be positive, or 0 for the default.")
            : value > 0 ? value : compactDefault;

}
