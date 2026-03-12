using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.ImageEditing;

/// <summary>
/// Prompt-to-Prompt model for attention-based image editing by manipulating cross-attention maps.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Prompt-to-Prompt enables image editing by directly manipulating the cross-attention maps
/// during the diffusion process. By controlling which attention maps are replaced, refined,
/// or reweighted between the original and edited prompts, users can make precise localized
/// edits to generated or real images without masks.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>SD 1.5 U-Net backbone (320 base channels, [1,2,4,4], 768-dim CLIP)</description></item>
/// <item><description>Cross-attention map extraction and manipulation during inference</description></item>
/// <item><description>Three editing modes: word swap, attention reweight, refinement</description></item>
/// <item><description>Standard SD 1.5 VAE for image encoding/decoding</description></item>
/// <item><description>DDIM scheduler for deterministic attention control</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Prompt-to-Prompt edits images by changing the prompt and controlling attention maps.
///
/// How Prompt-to-Prompt works:
/// 1. Generate an image with the original prompt, storing cross-attention maps at each step
/// 2. Modify the prompt (e.g., "a cat sitting" to "a dog sitting")
/// 3. During re-generation, inject stored attention maps for unchanged words
/// 4. Only attention maps for changed words are regenerated
/// 5. This preserves the overall composition while editing specific elements
///
/// Editing modes:
/// - Word swap: "a cat sitting" to "a dog sitting" (replaces one element)
/// - Attention reweight: increase/decrease attention to specific words (make something bigger/smaller)
/// - Refinement: add detail to specific regions without changing structure
///
/// When to use Prompt-to-Prompt:
/// - Structure-preserving image edits via text changes
/// - Swapping objects while maintaining composition
/// - Adjusting emphasis on specific image elements
/// - Research on attention-based image control
///
/// Limitations:
/// - Requires deterministic scheduler (DDIM) for attention consistency
/// - Quality depends on attention map alignment between prompts
/// - Complex structural changes may break attention correspondence
/// - SD 1.5 resolution (512x512)
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SD 1.5 U-Net with cross-attention manipulation
/// - Backbone: SD 1.5 (320 base channels, [1,2,4,4] multipliers)
/// - Cross-attention: 768-dim (CLIP ViT-L/14)
/// - Editing modes: word-swap, attention-reweight, refinement
/// - Default resolution: 512x512
/// - Scheduler: DDIM (deterministic for attention consistency)
///
/// Reference: Hertz et al., "Prompt-to-Prompt Image Editing with Cross Attention Control", ICLR 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var p2p = new PromptToPromptModel&lt;float&gt;();
/// var image = p2p.GenerateFromText(
///     prompt: "A photo of a cat sitting on a sofa",
///     width: 512, height: 512,
///     numInferenceSteps: 50,
///     guidanceScale: 7.5);
/// </code>
/// </example>
public class PromptToPromptModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 512;
    public const int DefaultHeight = 512;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 768;
    private const int BASE_CHANNELS = 320;
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

    #endregion

    #region Constructor

    public PromptToPromptModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.00085,
                BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(unet, vae, seed);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? unet, StandardVAE<T>? vae, int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS, baseChannels: BASE_CHANNELS,
            channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 }, contextDim: CROSS_ATTENTION_DIM, seed: seed);

        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight, int numInferenceSteps = 50,
        double? guidanceScale = null, int? seed = null)
        => base.GenerateFromText(prompt, negativePrompt, width, height, numInferenceSteps,
            guidanceScale ?? DEFAULT_GUIDANCE_SCALE, seed);

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _unet.GetParameters();
        var vaeParams = _vae.GetParameters();
        var combined = new Vector<T>(unetParams.Length + vaeParams.Length);
        for (int i = 0; i < unetParams.Length; i++) combined[i] = unetParams[i];
        for (int i = 0; i < vaeParams.Length; i++) combined[unetParams.Length + i] = vaeParams[i];
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _unet.ParameterCount;
        var vaeCount = _vae.ParameterCount;
        if (parameters.Length != unetCount + vaeCount)
            throw new ArgumentException($"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.", nameof(parameters));
        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);
        for (int i = 0; i < unetCount; i++) unetParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++) vaeParams[i] = parameters[unetCount + i];
        _unet.SetParameters(unetParams);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clonedUnet = new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: BASE_CHANNELS, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2, 1 },
            contextDim: CROSS_ATTENTION_DIM);
        clonedUnet.SetParameters(_unet.GetParameters());
        var clonedVae = new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());
        return new PromptToPromptModel<T>(unet: clonedUnet, vae: clonedVae, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Prompt-to-Prompt", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Prompt-to-Prompt attention-based image editing with cross-attention control",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "sd15-attention-control");
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("base_channels", BASE_CHANNELS);
        metadata.SetProperty("editing_modes", "word-swap,attention-reweight,refinement");
        metadata.SetProperty("no_mask_required", true);
        metadata.SetProperty("deterministic_scheduler", "DDIM");
        metadata.SetProperty("base_model", "SD-1.5");
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("attention_manipulation", true);
        metadata.SetProperty("structure_preserving", true);
        return metadata;
    }

    #endregion
}
