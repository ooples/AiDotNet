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

namespace AiDotNet.Diffusion.Control;

/// <summary>
/// ControlNet Union Pro model that supports multiple control types in a single model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ControlNet Union Pro consolidates multiple control types into a single unified model,
/// eliminating the need to load separate ControlNet checkpoints for each control type.
/// Supports switching between and combining control modes at inference time.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of needing a different model file for edges, depth,
/// poses, etc., this single model handles all control types. You just tell it which
/// type of control image you're providing, and it adapts automatically.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a unified ControlNet that handles multiple control types
/// var options = new DiffusionModelOptions&lt;float&gt;
/// {
///     LatentChannels = 4,
///     DefaultInferenceSteps = 25
/// };
///
/// // Generate with any control type using the same model
/// var controlInput = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 1, 512, 512 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new ControlNetUnionProModel&lt;float&gt;(options: options))
///     .Build(trainX, trainY);
/// var prediction = result.Predict(controlInput);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback", "https://arxiv.org/abs/2404.07987")]
public partial class ControlNetUnionProModel<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_baseUNet);
    }

    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 7.5;

    private UNetNoisePredictor<T> _baseUNet;
    private StandardVAE<T> _vae;
    private Dictionary<ControlType, ControlNetEncoder<T>> _encoderCache;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly ControlType[] _supportedTypes;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _baseUNet;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;



    /// <summary>
    /// Initializes a new ControlNet Union Pro model.
    /// </summary>
    public ControlNetUnionProModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? baseUNet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        ControlType[]? supportedTypes = null,
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
        _supportedTypes = supportedTypes ?? new[]
        {
            ControlType.Canny, ControlType.Depth, ControlType.Pose,
            ControlType.Normal, ControlType.Segmentation, ControlType.LineArt,
            ControlType.SoftEdge, ControlType.Scribble, ControlType.Tile
        };
        InitializeLayers(baseUNet, vae, seed);
    }

    [MemberNotNull(nameof(_baseUNet), nameof(_vae), nameof(_encoderCache))]
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

        _encoderCache = new Dictionary<ControlType, ControlNetEncoder<T>>();
        foreach (var ct in _supportedTypes)
        {
            _encoderCache[ct] = new ControlNetEncoder<T>(
                inputChannels: 3,
                baseChannels: 320,
                channelMultipliers: new[] { 1, 2, 4, 4 },
                seed: seed);
        }
    }



    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ControlNet-Union-Pro",
            Version = "1.0",
            Description = "Unified ControlNet supporting multiple control types in a single model",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "unet-multi-type-controlnet");
        metadata.SetProperty("base_model", "Stable Diffusion 1.5");
        metadata.SetProperty("text_encoder", "CLIP ViT-L/14");
        metadata.SetProperty("context_dim", 768);
        metadata.SetProperty("supported_types", string.Join(", ", _supportedTypes.Select(t => t.ToString())));
        metadata.SetProperty("num_types", _supportedTypes.Length);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("guidance_scale", DEFAULT_GUIDANCE);

        return metadata;
    }
}
