using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
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
public class ControlNetUnionProModel<T> : LatentDiffusionModelBase<T>
{
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

    /// <inheritdoc />
    public override int ParameterCount
    {
        get
        {
            int count = _baseUNet.ParameterCount;
            foreach (var enc in _encoderCache.Values)
                count += enc.ParameterCount;
            return count;
        }
    }

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
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        var baseParams = _baseUNet.GetParameters();
        for (int i = 0; i < baseParams.Length; i++) allParams.Add(baseParams[i]);
        foreach (var kvp in _encoderCache.OrderBy(kv => kv.Key))
        {
            var encParams = kvp.Value.GetParameters();
            for (int i = 0; i < encParams.Length; i++) allParams.Add(encParams[i]);
        }
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

        foreach (var kvp in _encoderCache.OrderBy(kv => kv.Key))
        {
            var encCount = kvp.Value.ParameterCount;
            var encParams = new T[encCount];
            for (int i = 0; i < encCount; i++) encParams[i] = parameters[offset + i];
            kvp.Value.SetParameters(new Vector<T>(encParams));
            offset += encCount;
        }
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new ControlNetUnionProModel<T>(
            conditioner: _conditioner,
            supportedTypes: _supportedTypes,
            seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ControlNet-Union-Pro",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Unified ControlNet supporting multiple control types in a single model",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "unet-multi-type-controlnet");
        metadata.SetProperty("supported_types", string.Join(", ", _supportedTypes.Select(t => t.ToString())));
        metadata.SetProperty("num_types", _supportedTypes.Length);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("default_guidance_scale", DEFAULT_GUIDANCE);

        return metadata;
    }
}
