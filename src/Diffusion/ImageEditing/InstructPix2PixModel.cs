using System.Diagnostics.CodeAnalysis;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion;

/// <summary>
/// InstructPix2Pix model for instruction-based image editing via natural language text prompts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// InstructPix2Pix enables editing images by following natural language instructions.
/// It takes an input image and a text instruction (e.g., "make it winter") and produces
/// the edited result without requiring masks or per-example fine-tuning. The model uses
/// dual classifier-free guidance with separate image and text guidance scales.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>SD 1.5 U-Net with 8 input channels (4 latent noise + 4 image conditioning)</description></item>
/// <item><description>CLIP ViT-L/14 text encoder for 768-dim instruction embedding</description></item>
/// <item><description>Dual classifier-free guidance (image guidance + text guidance)</description></item>
/// <item><description>Standard SD 1.5 VAE for image encoding/decoding</description></item>
/// <item><description>Euler discrete scheduler for efficient inference</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> InstructPix2Pix edits images using text instructions.
///
/// How InstructPix2Pix works:
/// 1. Input image is encoded to latent space via VAE (4 channels)
/// 2. Text instruction is encoded by CLIP into 768-dim features
/// 3. Image latent is concatenated with noise latent (4+4 = 8 input channels)
/// 4. U-Net denoises with dual guidance: image fidelity + instruction following
/// 5. Image guidance scale controls how much to preserve the original image
/// 6. Text guidance scale controls how strongly to follow the instruction
/// 7. Decoded result is the edited image
///
/// Key characteristics:
/// - Natural language editing: "make the sky sunset colors", "add snow"
/// - No mask needed: the model figures out what to change
/// - Dual guidance: image guidance (1.0-2.0) + text guidance (7.0-12.0)
/// - Based on SD 1.5 with additional input channels
/// - Trained on GPT-4 generated instruction-image pairs
///
/// When to use InstructPix2Pix:
/// - Text-based image editing without masks
/// - Batch editing with consistent instructions
/// - Style transfer via natural language
/// - Quick creative edits from text descriptions
///
/// Limitations:
/// - Edit quality depends on instruction clarity
/// - May over-edit or under-edit without careful guidance tuning
/// - 512x512 base resolution (SD 1.5)
/// - Cannot handle fine-grained local edits as well as mask-based methods
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SD 1.5 U-Net with 8 input channels
/// - Input: 8 channels (4 latent noise + 4 image conditioning)
/// - Text encoder: CLIP ViT-L/14 (768-dim)
/// - Image guidance scale: 1.0-2.0 recommended
/// - Text guidance scale: 7.0-12.0 recommended
/// - Default resolution: 512x512
/// - Scheduler: Euler discrete
///
/// Reference: Brooks et al., "InstructPix2Pix: Learning to Follow Image Editing Instructions", CVPR 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var instructPix2Pix = new InstructPix2PixModel&lt;float&gt;();
/// var editedImage = instructPix2Pix.GenerateFromText(
///     prompt: "Make it a winter scene with snow",
///     width: 512, height: 512,
///     numInferenceSteps: 20,
///     guidanceScale: 7.5);
/// </code>
/// </example>
public class InstructPix2PixModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 512;
    public const int DefaultHeight = 512;
    private const int LATENT_CHANNELS = 4;
    private const int INPUT_CHANNELS = 8;
    private const int CROSS_ATTENTION_DIM = 768;
    private const int BASE_CHANNELS = 320;
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;
    private const double DEFAULT_IMAGE_GUIDANCE = 1.5;

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

    /// <summary>Gets the default image guidance scale for balancing image fidelity.</summary>
    public double DefaultImageGuidance => DEFAULT_IMAGE_GUIDANCE;

    #endregion

    #region Constructor

    public InstructPix2PixModel(
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
            scheduler ?? new EulerDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
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
            architecture: Architecture, inputChannels: INPUT_CHANNELS,
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
        int width = DefaultWidth, int height = DefaultHeight, int numInferenceSteps = 20,
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
            inputChannels: INPUT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: BASE_CHANNELS, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2, 1 },
            contextDim: CROSS_ATTENTION_DIM);
        clonedUnet.SetParameters(_unet.GetParameters());
        var clonedVae = new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());
        return new InstructPix2PixModel<T>(unet: clonedUnet, vae: clonedVae, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "InstructPix2Pix", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "InstructPix2Pix instruction-based image editing with dual guidance",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "sd15-8ch-input-dual-guidance");
        metadata.SetProperty("input_channels", INPUT_CHANNELS);
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("base_channels", BASE_CHANNELS);
        metadata.SetProperty("image_guidance_scale", DEFAULT_IMAGE_GUIDANCE);
        metadata.SetProperty("text_guidance_scale", DEFAULT_GUIDANCE_SCALE);
        metadata.SetProperty("no_mask_required", true);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("scheduler", "Euler-discrete");
        metadata.SetProperty("training_data", "GPT-4-generated-pairs");
        return metadata;
    }

    #endregion
}
