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
/// IP-Adapter FaceID model for face-specific identity preservation using facial recognition embeddings.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// IP-Adapter FaceID extends the IP-Adapter framework with facial recognition embeddings
/// (ArcFace/InsightFace) instead of CLIP image embeddings, providing significantly more
/// accurate face identity preservation during image generation.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>SD 1.5 U-Net backbone (320 base channels, [1,2,4,4], 768-dim CLIP)</description></item>
/// <item><description>ArcFace/InsightFace face encoder producing 512-dim face embeddings</description></item>
/// <item><description>Face embedding projection layer to cross-attention token space</description></item>
/// <item><description>Optional LoRA layers for improved generation quality</description></item>
/// <item><description>Standard SD 1.5 VAE for image encoding/decoding</description></item>
/// <item><description>Euler discrete scheduler for efficient inference</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> IP-Adapter FaceID preserves face identity in generated images.
///
/// How IP-Adapter FaceID works:
/// 1. A reference face image is processed by ArcFace to extract a 512-dim face embedding
/// 2. The face embedding is projected to cross-attention token space via a learned projection
/// 3. Face tokens are injected into the U-Net's cross-attention alongside text tokens
/// 4. The U-Net generates an image that matches both the text prompt and reference face
/// 5. Optional LoRA weights further improve generation quality and identity fidelity
///
/// Key differences from standard IP-Adapter:
/// - Uses face recognition (ArcFace) embeddings instead of CLIP image embeddings
/// - Much better face identity preservation (ID similarity)
/// - Specialized for face-centric generation tasks
/// - Compatible with SD 1.5 and SDXL backbones
///
/// When to use IP-Adapter FaceID:
/// - High-fidelity face identity transfer to new scenes
/// - Face-consistent character generation across images
/// - Personalized avatar and portrait creation
/// - Marketing and content with specific faces
///
/// Limitations:
/// - Requires clear frontal face in reference image
/// - May struggle with extreme poses or occlusions
/// - Face identity can drift with complex text prompts
/// - Single-face: multi-face requires separate handling
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: IP-Adapter with ArcFace face encoder
/// - Face encoder: ArcFace/InsightFace (512-dim embedding)
/// - Backbone: SD 1.5 (320 base, [1,2,4,4], 768-dim CLIP)
/// - Projection: face embedding to cross-attention tokens
/// - Default resolution: 512x512
/// - Scheduler: Euler discrete
/// - Optional: LoRA for quality improvement
/// - Compatible: SD 1.5, SDXL
///
/// Reference: Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var faceID = new IPAdapterFaceIDModel&lt;float&gt;();
/// var image = faceID.GenerateFromText(
///     prompt: "A portrait photo in a garden setting",
///     width: 512, height: 512,
///     numInferenceSteps: 30,
///     guidanceScale: 7.5);
/// </code>
/// </example>
public class IPAdapterFaceIDModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    public const int DefaultWidth = 512;
    public const int DefaultHeight = 512;
    private const int LATENT_CHANNELS = 4;
    private const int CROSS_ATTENTION_DIM = 768;
    private const int BASE_CHANNELS = 320;
    private const int FACE_EMBEDDING_DIM = 512;
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

    public IPAdapterFaceIDModel(
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
        int width = DefaultWidth, int height = DefaultHeight, int numInferenceSteps = 30,
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
        return new IPAdapterFaceIDModel<T>(unet: clonedUnet, vae: clonedVae, conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "IP-Adapter FaceID", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "IP-Adapter FaceID face-specific identity preservation with ArcFace embeddings",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "ip-adapter-arcface");
        metadata.SetProperty("face_encoder", "ArcFace/InsightFace");
        metadata.SetProperty("face_embedding_dim", FACE_EMBEDDING_DIM);
        metadata.SetProperty("cross_attention_dim", CROSS_ATTENTION_DIM);
        metadata.SetProperty("base_channels", BASE_CHANNELS);
        metadata.SetProperty("base_model", "SD-1.5");
        metadata.SetProperty("optional_lora", true);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("scheduler", "Euler-discrete");
        metadata.SetProperty("compatible_models", "SD-1.5,SDXL");
        return metadata;
    }

    #endregion
}
