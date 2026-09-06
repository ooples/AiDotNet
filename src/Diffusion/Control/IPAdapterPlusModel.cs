using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
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
/// <example>
/// <code>
/// // Create IP-Adapter Plus for image-prompt-guided generation
/// var options = new DiffusionModelOptions&lt;float&gt;
/// {
///     LatentChannels = 4,
///     DefaultInferenceSteps = 30
/// };
///
/// // Use a reference image to guide generation style and content
/// var imageFeatures = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 1024 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new IPAdapterPlusModel&lt;float&gt;(options: options))
///     .Build(trainX, trainY);
/// var prediction = result.Predict(imageFeatures);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models", "https://arxiv.org/abs/2308.06721", Year = 2023, Authors = "Ye et al.")]
public partial class IPAdapterPlusModel<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_baseUNet);
        RegisterParameterComponent(_imageProjection);
    }

    private const int LATENT_CHANNELS = 4;
    private const int IMAGE_EMBED_DIM = 1024;
    private const int CROSS_ATTENTION_DIM = 768;
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
            contextDim: CROSS_ATTENTION_DIM,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);

        // Image projection: maps CLIP image embeddings (IMAGE_EMBED_DIM = 1024) to the
        // UNet's cross-attention space (CROSS_ATTENTION_DIM = 768). Pre-resolved so
        // ParameterCount, GetParameters, SetParameters, and Clone work before any forward.
        _imageProjection = new DenseLayer<T>(CROSS_ATTENTION_DIM);
        _imageProjection.ResolveFromShape(new[] { 1, IMAGE_EMBED_DIM });
    }



    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "IP-Adapter-Plus",
            Version = "1.0",
            Description = "Image prompt adapter with fine-grained image feature injection",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "unet-decoupled-cross-attention");
        metadata.SetProperty("base_model", "Stable Diffusion 1.5");
        metadata.SetProperty("text_encoder", "CLIP ViT-L/14");
        metadata.SetProperty("image_encoder", "CLIP ViT-H/14");
        metadata.SetProperty("context_dim", 768);
        metadata.SetProperty("image_embed_dim", IMAGE_EMBED_DIM);
        metadata.SetProperty("ip_adapter_scale", _ipAdapterScale);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("guidance_scale", DEFAULT_GUIDANCE);

        return metadata;
    }
}
