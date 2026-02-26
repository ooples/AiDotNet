using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Control;

/// <summary>
/// IP-Adapter Plus model for image prompt conditioning in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// IP-Adapter Plus enables image-based conditioning for diffusion models by extracting
/// image features through a vision encoder and injecting them via cross-attention.
/// The "Plus" variant uses fine-grained image features with decoupled cross-attention
/// for higher fidelity image prompting.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of describing what you want with text, you can show
/// the AI a reference image. IP-Adapter Plus extracts the style and content from your
/// reference and applies them to the generation, like saying "make something like this."
/// </para>
/// <para>
/// Reference: Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models", 2023
/// </para>
/// </remarks>
public class IPAdapterPlusModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const int IMAGE_EMBED_DIM = 1024;
    private const double DEFAULT_GUIDANCE = 7.5;

    private UNetNoisePredictor<T> _baseUNet;
    private StandardVAE<T> _vae;
    private DenseLayer<T> _imageProjection;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly double _ipAdapterScale;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _baseUNet;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _baseUNet.ParameterCount + _imageProjection.ParameterCount;

    /// <summary>
    /// Initializes a new IP-Adapter Plus model.
    /// </summary>
    public IPAdapterPlusModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? baseUNet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        double ipAdapterScale = 0.6,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        _ipAdapterScale = ipAdapterScale;
        InitializeLayers(baseUNet, vae, seed);
    }

    [MemberNotNull(nameof(_baseUNet), nameof(_vae), nameof(_imageProjection))]
    private void InitializeLayers(UNetNoisePredictor<T>? baseUNet, StandardVAE<T>? vae, int? seed)
    {
        _baseUNet = baseUNet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: 768,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);

        // Image projection: maps CLIP image embeddings to cross-attention space
        _imageProjection = new DenseLayer<T>(IMAGE_EMBED_DIM, 768);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        var baseParams = _baseUNet.GetParameters();
        for (int i = 0; i < baseParams.Length; i++) allParams.Add(baseParams[i]);
        var projParams = _imageProjection.GetParameters();
        for (int i = 0; i < projParams.Length; i++) allParams.Add(projParams[i]);
        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        var baseCount = _baseUNet.ParameterCount;
        var baseParams = new T[baseCount];
        for (int i = 0; i < baseCount; i++) baseParams[i] = parameters[offset + i];
        _baseUNet.SetParameters(new Vector<T>(baseParams));
        offset += baseCount;

        var projCount = _imageProjection.ParameterCount;
        var projParams = new T[projCount];
        for (int i = 0; i < projCount; i++) projParams[i] = parameters[offset + i];
        _imageProjection.SetParameters(new Vector<T>(projParams));
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new IPAdapterPlusModel<T>(
            conditioner: _conditioner,
            ipAdapterScale: _ipAdapterScale,
            seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "IP-Adapter-Plus",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Image prompt adapter with fine-grained image feature injection",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "unet-decoupled-cross-attention");
        metadata.SetProperty("image_embed_dim", IMAGE_EMBED_DIM);
        metadata.SetProperty("ip_adapter_scale", _ipAdapterScale);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_guidance_scale", DEFAULT_GUIDANCE);

        return metadata;
    }
}
