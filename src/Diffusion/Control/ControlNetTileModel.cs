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
/// ControlNet Tile model for upscaling and detail enhancement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ControlNet Tile preserves overall color and composition from a blurred/downsampled
/// input while regenerating fine details. Commonly used for image upscaling,
/// detail enhancement, and texture regeneration.
/// </para>
/// <para>
/// <b>For Beginners:</b> Give this model a blurry or low-resolution image, and it
/// will generate a sharp, detailed version while keeping the same colors and layout.
/// It's great for making images look clearer and more detailed.
/// </para>
/// </remarks>
public class ControlNetTileModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;

    private UNetNoisePredictor<T> _baseUNet;
    private StandardVAE<T> _vae;
    private ControlNetEncoder<T> _controlEncoder;
    private readonly IConditioningModule<T>? _conditioner;

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

    public ControlNetTileModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? baseUNet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
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
        _controlEncoder = new ControlNetEncoder<T>(
            inputChannels: 3, baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 }, seed: seed);
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
        var clone = new ControlNetTileModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ControlNet-Tile", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "ControlNet Tile for upscaling and detail enhancement",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "unet-controlnet-tile");
        metadata.SetProperty("control_type", ControlType.Tile.ToString());
        return metadata;
    }
}
