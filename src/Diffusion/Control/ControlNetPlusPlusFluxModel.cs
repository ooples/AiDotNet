using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.Control;

/// <summary>
/// ControlNet++ adapted for FLUX architecture with reward-guided training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Combines ControlNet++ reward-guided training with FLUX's flow-matching architecture.
/// Uses 16-channel latent space with double-stream transformer blocks for improved
/// control signal adherence on FLUX-based models.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the most advanced version of ControlNet that works
/// with FLUX models. It combines the improved training (ControlNet++) with FLUX's
/// powerful architecture for the best possible control over generated images.
/// </para>
/// </remarks>
public class ControlNetPlusPlusFluxModel<T> : LatentDiffusionModelBase<T>
{
    private const int FLUX_LATENT_CHANNELS = 16;
    private const int FLUX_CONTEXT_DIM = 4096;
    private const double DEFAULT_GUIDANCE = 3.5;

    private FluxDoubleStreamPredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private ControlNetEncoder<T> _controlEncoder;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly ControlType _controlType;
    private readonly double _rewardWeight;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => FLUX_LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _controlEncoder.ParameterCount;

    public ControlNetPlusPlusFluxModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        FluxDoubleStreamPredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        ControlType controlType = ControlType.Canny,
        double rewardWeight = 0.5,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _controlType = controlType;
        _conditioner = conditioner;
        _rewardWeight = rewardWeight;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae), nameof(_controlEncoder))]
    private void InitializeLayers(FluxDoubleStreamPredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new FluxDoubleStreamPredictor<T>(seed: seed);
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: FLUX_LATENT_CHANNELS, baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocksPerLevel: 2, seed: seed);
        _controlEncoder = new ControlNetEncoder<T>(
            inputChannels: 3, baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 }, seed: seed);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var all = new List<T>();
        var p1 = _predictor.GetParameters(); for (int i = 0; i < p1.Length; i++) all.Add(p1[i]);
        var p2 = _controlEncoder.GetParameters(); for (int i = 0; i < p2.Length; i++) all.Add(p2[i]);
        return new Vector<T>(all.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int o = 0;
        var c1 = _predictor.ParameterCount; var a1 = new T[c1]; for (int i = 0; i < c1; i++) a1[i] = parameters[o + i]; _predictor.SetParameters(new Vector<T>(a1)); o += c1;
        var c2 = _controlEncoder.ParameterCount; var a2 = new T[c2]; for (int i = 0; i < c2; i++) a2[i] = parameters[o + i]; _controlEncoder.SetParameters(new Vector<T>(a2));
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new ControlNetPlusPlusFluxModel<T>(
            controlType: _controlType, conditioner: _conditioner, rewardWeight: _rewardWeight, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ControlNet++-FLUX", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "ControlNet++ with reward-guided training for FLUX flow-matching architecture",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "flux-controlnet-plus-plus");
        metadata.SetProperty("base_model", "FLUX.1");
        metadata.SetProperty("text_encoder", "CLIP-L + T5-XXL");
        metadata.SetProperty("context_dim", FLUX_CONTEXT_DIM);
        metadata.SetProperty("control_type", _controlType.ToString());
        metadata.SetProperty("reward_weight", _rewardWeight);
        metadata.SetProperty("latent_channels", FLUX_LATENT_CHANNELS);
        metadata.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return metadata;
    }
}
