using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// InstructPix2Pix model — instruction-based image editing via text prompts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// InstructPix2Pix enables editing images by following natural language instructions.
/// It takes an input image and a text instruction (e.g., "make it winter") and produces
/// the edited result without requiring masks or per-example fine-tuning.
/// </para>
/// <para>
/// <b>For Beginners:</b> InstructPix2Pix edits images using text instructions:
///
/// Key characteristics:
/// - Natural language editing: "make the sky sunset colors"
/// - No mask needed: the model figures out what to change
/// - Dual guidance: image guidance + text guidance scales
/// - Based on SD 1.5 architecture with additional input channels
///
/// Use InstructPix2Pix when you need:
/// - Text-based image editing
/// - No-mask editing workflows
/// - Batch editing with consistent instructions
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SD 1.5 U-Net with 8 input channels (4 latent + 4 conditioning)
/// - Text encoder: CLIP ViT-L/14
/// - Image guidance scale: 1.0-2.0 recommended
/// - Text guidance scale: 7.0-12.0 recommended
///
/// Reference: Brooks et al., "InstructPix2Pix: Learning to Follow Image Editing Instructions", CVPR 2023
/// </para>
/// </remarks>
public class InstructPix2PixModel<T> : LatentDiffusionModelBase<T>
{
    public const int DefaultWidth = 512;
    public const int DefaultHeight = 512;
    private const int LATENT_CHANNELS = 4;
    private const int INPUT_CHANNELS = 8;
    private const int CROSS_ATTENTION_DIM = 768;
    private const double DEFAULT_GUIDANCE_SCALE = 7.5;
    private const double DEFAULT_IMAGE_GUIDANCE = 1.5;

    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

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
    /// <summary>Gets the default image guidance scale.</summary>
    public double DefaultImageGuidance => DEFAULT_IMAGE_GUIDANCE;

    public InstructPix2PixModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null, StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
               scheduler ?? new EulerDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()))
    {
        _conditioner = conditioner;
        InitializeLayers(unet, vae, seed);
    }

    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? unet, StandardVAE<T>? vae, int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: INPUT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2, attentionResolutions: [4, 2, 1],
            contextDim: CROSS_ATTENTION_DIM, seed: seed);
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
    }

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight, int numInferenceSteps = 20,
        double? guidanceScale = null, int? seed = null)
        => base.GenerateFromText(prompt, negativePrompt, width, height, numInferenceSteps, guidanceScale ?? DEFAULT_GUIDANCE_SCALE, seed);

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var up = _unet.GetParameters(); var vp = _vae.GetParameters();
        var c = new Vector<T>(up.Length + vp.Length);
        for (int i = 0; i < up.Length; i++) c[i] = up[i];
        for (int i = 0; i < vp.Length; i++) c[up.Length + i] = vp[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int uc = _unet.ParameterCount, vc = _vae.ParameterCount;
        if (parameters.Length != uc + vc) throw new ArgumentException($"Expected {uc + vc}, got {parameters.Length}.", nameof(parameters));
        var up = new Vector<T>(uc); var vp = new Vector<T>(vc);
        for (int i = 0; i < uc; i++) up[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[uc + i];
        _unet.SetParameters(up); _vae.SetParameters(vp);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var cu = new UNetNoisePredictor<T>(
            inputChannels: INPUT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2, attentionResolutions: [4, 2, 1],
            contextDim: CROSS_ATTENTION_DIM);
        cu.SetParameters(_unet.GetParameters());
        var cv = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
        cv.SetParameters(_vae.GetParameters());
        return new InstructPix2PixModel<T>(unet: cu, vae: cv, conditioner: _conditioner);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "InstructPix2Pix", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "InstructPix2Pix instruction-based image editing", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "sd15-8ch-input");
        m.SetProperty("input_channels", INPUT_CHANNELS);
        m.SetProperty("image_guidance_scale", DEFAULT_IMAGE_GUIDANCE);
        m.SetProperty("default_resolution", DefaultWidth);
        return m;
    }
}
