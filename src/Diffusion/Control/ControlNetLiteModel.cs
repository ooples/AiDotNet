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
/// Lightweight ControlNet model with reduced parameter count for faster inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ControlNet Lite reduces the encoder to approximately 25% of the full ControlNet's
/// parameters while maintaining acceptable control quality. Suitable for real-time
/// or resource-constrained applications.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is a smaller, faster version of ControlNet. It uses
/// fewer parameters so it runs quicker and uses less memory, at the cost of slightly
/// less precise control signal adherence.
/// </para>
/// </remarks>
public class ControlNetLiteModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;

    private UNetNoisePredictor<T> _baseUNet;
    private StandardVAE<T> _vae;
    private ControlNetEncoder<T> _controlEncoder;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly ControlType _controlType;

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
    /// Initializes a new ControlNet Lite model.
    /// </summary>
    public ControlNetLiteModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? baseUNet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        ControlType controlType = ControlType.Canny,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _controlType = controlType;
        _conditioner = conditioner;
        InitializeLayers(baseUNet, vae, seed);
    }

    [MemberNotNull(nameof(_baseUNet), nameof(_vae), nameof(_controlEncoder))]
    private void InitializeLayers(UNetNoisePredictor<T>? baseUNet, StandardVAE<T>? vae, int? seed)
    {
        _baseUNet = baseUNet ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 }, contextDim: 768, seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS, baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocksPerLevel: 2, seed: seed);

        // Lite: smaller base channels (160 instead of 320) and fewer multipliers
        _controlEncoder = new ControlNetEncoder<T>(
            inputChannels: 3, baseChannels: 160, channelMultipliers: new[] { 1, 2, 4 }, seed: seed);
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
        var clone = new ControlNetLiteModel<T>(controlType: _controlType, conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ControlNet-Lite", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Lightweight ControlNet with ~25% parameters for faster inference",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "unet-controlnet-lite");
        metadata.SetProperty("control_type", _controlType.ToString());
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        return metadata;
    }
}
