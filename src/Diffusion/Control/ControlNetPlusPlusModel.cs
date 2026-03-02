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
/// ControlNet++ model with improved conditioning via reward-guided training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ControlNet++ improves upon ControlNet by using reward-guided training that
/// produces more consistent and higher-quality control signal adherence. It supports
/// multiple control types simultaneously with better composability.
/// </para>
/// <para>
/// <b>For Beginners:</b> ControlNet++ is a better version of ControlNet that follows
/// your control images (edges, depth, poses) more accurately. It was trained with
/// a smarter method that teaches it to match control signals more faithfully.
/// </para>
/// <para>
/// Reference: Li et al., "ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback", ECCV 2024
/// </para>
/// </remarks>
public class ControlNetPlusPlusModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 7.5;
    private const double DEFAULT_REWARD_WEIGHT = 0.5;

    private UNetNoisePredictor<T> _baseUNet;
    private ControlNetEncoder<T> _controlEncoder;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly ControlType _controlType;
    private readonly double _rewardWeight;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _baseUNet;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _baseUNet.ParameterCount + _controlEncoder.ParameterCount;

    /// <summary>
    /// Initializes a new ControlNet++ model.
    /// </summary>
    public ControlNetPlusPlusModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? baseUNet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        ControlType controlType = ControlType.Canny,
        double rewardWeight = DEFAULT_REWARD_WEIGHT,
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
        _controlType = controlType;
        _conditioner = conditioner;
        _rewardWeight = rewardWeight;
        InitializeLayers(baseUNet, vae, seed);
    }

    [MemberNotNull(nameof(_baseUNet), nameof(_vae), nameof(_controlEncoder))]
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

        _controlEncoder = new ControlNetEncoder<T>(
            inputChannels: 3,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            seed: seed);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        var baseParams = _baseUNet.GetParameters();
        for (int i = 0; i < baseParams.Length; i++) allParams.Add(baseParams[i]);
        var ctrlParams = _controlEncoder.GetParameters();
        for (int i = 0; i < ctrlParams.Length; i++) allParams.Add(ctrlParams[i]);
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

        var ctrlCount = _controlEncoder.ParameterCount;
        var ctrlParams = new T[ctrlCount];
        for (int i = 0; i < ctrlCount; i++) ctrlParams[i] = parameters[offset + i];
        _controlEncoder.SetParameters(new Vector<T>(ctrlParams));
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new ControlNetPlusPlusModel<T>(
            controlType: _controlType,
            conditioner: _conditioner,
            rewardWeight: _rewardWeight,
            seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ControlNet++",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Improved ControlNet with reward-guided training for better control adherence",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "unet-reward-guided-controlnet");
        metadata.SetProperty("base_model", "Stable Diffusion 1.5");
        metadata.SetProperty("text_encoder", "CLIP ViT-L/14");
        metadata.SetProperty("context_dim", 768);
        metadata.SetProperty("control_type", _controlType.ToString());
        metadata.SetProperty("reward_weight", _rewardWeight);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("guidance_scale", DEFAULT_GUIDANCE);

        return metadata;
    }
}
