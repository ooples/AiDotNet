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
/// ControlNeXt model with improved efficiency and generalization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ControlNeXt improves upon ControlNet by using a more parameter-efficient architecture
/// with cross-normalization layers instead of a full encoder copy. This reduces memory
/// usage while improving generalization across different control types.
/// </para>
/// <para>
/// <b>For Beginners:</b> ControlNeXt is a newer, more efficient version of ControlNet.
/// It uses a smarter design that requires fewer parameters and less memory while
/// working just as well (or better) at following control signals.
/// </para>
/// <para>
/// Reference: Peng et al., "ControlNeXt: Powerful and Efficient Control for Image and Video Generation", 2024
/// </para>
/// </remarks>
public class ControlNeXtModel<T> : LatentDiffusionModelBase<T>
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

    public ControlNeXtModel(
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
        // ControlNeXt uses cross-normalization, simulated here with a smaller encoder
        _controlEncoder = new ControlNetEncoder<T>(
            inputChannels: 3, baseChannels: 256, channelMultipliers: new[] { 1, 2, 4 }, seed: seed);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var all = new List<T>();
        var p1 = _baseUNet.GetParameters(); for (int i = 0; i < p1.Length; i++) all.Add(p1[i]);
        var p2 = _controlEncoder.GetParameters(); for (int i = 0; i < p2.Length; i++) all.Add(p2[i]);
        return new Vector<T>(all.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int o = 0;
        var c1 = _baseUNet.ParameterCount; var a1 = new T[c1]; for (int i = 0; i < c1; i++) a1[i] = parameters[o + i]; _baseUNet.SetParameters(new Vector<T>(a1)); o += c1;
        var c2 = _controlEncoder.ParameterCount; var a2 = new T[c2]; for (int i = 0; i < c2; i++) a2[i] = parameters[o + i]; _controlEncoder.SetParameters(new Vector<T>(a2));
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new ControlNeXtModel<T>(controlType: _controlType, conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ControlNeXt", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Parameter-efficient control with cross-normalization for improved generalization",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "unet-controlnext");
        metadata.SetProperty("base_model", "Stable Diffusion 1.5");
        metadata.SetProperty("text_encoder", "CLIP ViT-L/14");
        metadata.SetProperty("context_dim", 768);
        metadata.SetProperty("control_type", _controlType.ToString());
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("guidance_scale", 7.5);
        return metadata;
    }
}
