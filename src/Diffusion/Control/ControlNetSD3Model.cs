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
/// ControlNet adapted for Stable Diffusion 3's MMDiT architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Adapts ControlNet conditioning for SD3's Multi-Modal Diffusion Transformer (MMDiT)
/// architecture. Uses 16-channel latent space and supports dual text encoders
/// (CLIP + T5) for enhanced prompt understanding with control signal injection.
/// </para>
/// <para>
/// <b>For Beginners:</b> This brings ControlNet control to Stable Diffusion 3 models.
/// SD3 uses a completely different architecture from SD1.5/SDXL, so this version
/// is specially designed to inject control signals into the transformer blocks.
/// </para>
/// </remarks>
public class ControlNetSD3Model<T> : LatentDiffusionModelBase<T>
{
    private const int SD3_LATENT_CHANNELS = 16;
    private const int SD3_CONTEXT_DIM = 4096;
    private const double DEFAULT_GUIDANCE = 7.0;

    private MMDiTXNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private ControlNetEncoder<T> _controlEncoder;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly ControlType _controlType;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => SD3_LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _controlEncoder.ParameterCount;

    /// <summary>
    /// Initializes a new ControlNet-SD3 model.
    /// </summary>
    public ControlNetSD3Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        MMDiTXNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        ControlType controlType = ControlType.Canny,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 1.0,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _controlType = controlType;
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae), nameof(_controlEncoder))]
    private void InitializeLayers(MMDiTXNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new MMDiTXNoisePredictor<T>(seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SD3_LATENT_CHANNELS,
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
        var predParams = _predictor.GetParameters();
        for (int i = 0; i < predParams.Length; i++) allParams.Add(predParams[i]);
        var ctrlParams = _controlEncoder.GetParameters();
        for (int i = 0; i < ctrlParams.Length; i++) allParams.Add(ctrlParams[i]);
        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        var predCount = _predictor.ParameterCount;
        var predParams = new T[predCount];
        for (int i = 0; i < predCount; i++) predParams[i] = parameters[offset + i];
        _predictor.SetParameters(new Vector<T>(predParams));
        offset += predCount;

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
        var clone = new ControlNetSD3Model<T>(
            controlType: _controlType,
            conditioner: _conditioner,
            seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ControlNet-SD3",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "ControlNet adapted for Stable Diffusion 3 MMDiT architecture",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "mmdit-x-controlnet");
        metadata.SetProperty("base_model", "Stable Diffusion 3");
        metadata.SetProperty("text_encoder", "CLIP-L + CLIP-G + T5-XXL");
        metadata.SetProperty("context_dim", SD3_CONTEXT_DIM);
        metadata.SetProperty("control_type", _controlType.ToString());
        metadata.SetProperty("latent_channels", SD3_LATENT_CHANNELS);
        metadata.SetProperty("guidance_scale", DEFAULT_GUIDANCE);

        return metadata;
    }
}
