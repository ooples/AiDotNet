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

namespace AiDotNet.Diffusion.Control;

/// <summary>
/// InstantID model — zero-shot identity-preserving image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// InstantID enables identity-preserving image generation from a single reference face image
/// without any fine-tuning. It combines a face encoder, IP-Adapter, and IdentityNet
/// to preserve facial identity while following text prompts.
/// </para>
/// <para>
/// <b>For Beginners:</b> InstantID generates images of a specific person from one photo:
///
/// Key characteristics:
/// - Single reference image: no training or fine-tuning needed
/// - Face identity preservation: maintains facial features
/// - Text prompt control: change style, pose, expression
/// - Works with SDXL for high-quality output
///
/// How InstantID works:
/// 1. Face encoder extracts identity embedding from reference image
/// 2. IP-Adapter injects identity into cross-attention layers
/// 3. IdentityNet provides spatial face guidance (landmarks)
/// 4. Text prompt controls style, background, and expression
///
/// Use InstantID when you need:
/// - Personalized image generation from one photo
/// - Face-consistent character generation
/// - Style transfer while preserving identity
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: SDXL + IP-Adapter + IdentityNet
/// - Face encoder: InsightFace (512-dim embedding)
/// - Base model: SDXL U-Net
/// - Control: face landmarks via IdentityNet
/// - Resolution: 1024x1024
///
/// Reference: Wang et al., "InstantID: Zero-shot Identity-Preserving Generation in Seconds", 2024
/// </para>
/// </remarks>
public class InstantIDModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 1024;
    public const int DefaultHeight = 1024;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 2048;
    private const double DEFAULT_GUIDANCE_SCALE = 5.0;
    private const int FACE_EMBEDDING_DIM = 512;

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
    /// <summary>Gets the face embedding dimension.</summary>
    public int FaceEmbeddingDim => FACE_EMBEDDING_DIM;

    #endregion

    #region Constructor

    public InstantIDModel(
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
            architecture: Architecture, inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4],
            numResBlocks: 2, attentionResolutions: [4, 2],
            contextDim: CROSS_ATTENTION_DIM, seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.13025, seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(
        string prompt, string? negativePrompt = null,
        int width = DefaultWidth, int height = DefaultHeight,
        int numInferenceSteps = 30, double? guidanceScale = null, int? seed = null)
    {
        return base.GenerateFromText(prompt, negativePrompt, width, height,
            numInferenceSteps, guidanceScale ?? DEFAULT_GUIDANCE_SCALE, seed);
    }

    #endregion

    #region IParameterizable

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
        if (parameters.Length != uc + vc)
            throw new ArgumentException($"Expected {uc + vc}, got {parameters.Length}.", nameof(parameters));
        var up = new Vector<T>(uc); var vp = new Vector<T>(vc);
        for (int i = 0; i < uc; i++) up[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[uc + i];
        _unet.SetParameters(up); _vae.SetParameters(vp);
    }

    #endregion

    #region ICloneable

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var cu = new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4],
            numResBlocks: 2, attentionResolutions: [4, 2],
            contextDim: CROSS_ATTENTION_DIM);
        cu.SetParameters(_unet.GetParameters());
        var cv = new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.13025);
        cv.SetParameters(_vae.GetParameters());
        return new InstantIDModel<T>(unet: cu, vae: cv, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "InstantID", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "InstantID zero-shot identity-preserving image generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "sdxl-ipadapter-identitynet");
        m.SetProperty("base_model", "Stable Diffusion XL");
        m.SetProperty("text_encoder", "CLIP ViT-L/14 + OpenCLIP ViT-bigG/14");
        m.SetProperty("context_dim", CROSS_ATTENTION_DIM);
        m.SetProperty("face_encoder", "InsightFace (Antelopev2)");
        m.SetProperty("face_embedding_dim", FACE_EMBEDDING_DIM);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE_SCALE);
        m.SetProperty("default_resolution", DefaultWidth);
        return m;
    }

    #endregion
}
