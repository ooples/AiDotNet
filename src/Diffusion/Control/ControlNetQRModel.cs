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
/// ControlNet QR model specialized for embedding QR codes in generated artwork.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Specialized ControlNet fine-tuned for QR code pattern control. Trained to
/// embed scannable QR codes into aesthetically pleasing generated images while
/// maintaining QR readability.
/// </para>
/// <para>
/// <b>For Beginners:</b> This model generates beautiful artwork that secretly
/// contains a working QR code. When you scan the generated image with a QR reader,
/// it works as a real QR code, but the image looks like art rather than a plain barcode.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a ControlNet QR model for artistic QR code generation
/// var options = new LatentDiffusionOptions&lt;float&gt;
/// {
///     LatentChannels = 4,
///     Height = 768,
///     Width = 768,
///     NumInferenceSteps = 30
/// };
/// var model = new ControlNetQRModel&lt;float&gt;(options);
///
/// // Generate artwork with embedded QR code
/// var qrPattern = Tensor&lt;float&gt;.Random(new[] { 1, 1, 768, 768 });
/// var artisticQR = model.Predict(qrPattern);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Adding Conditional Control to Text-to-Image Diffusion Models", "https://arxiv.org/abs/2302.05543")]
public partial class ControlNetQRModel<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_baseUNet);
        RegisterParameterComponent(_controlEncoder);
        RegisterParameterComponent(_vae);
    }

    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 10.0;

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

    /// <summary>
    /// Initializes a new ControlNet QR model.
    /// </summary>
    public ControlNetQRModel(
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

        // Per the ControlNet paper the control branch is a trainable copy of the base UNet's encoder,
        // so it must mirror the base UNet's channel configuration (else the injected control residuals
        // would not align with the UNet feature maps). Derive its width/depth from _baseUNet rather than
        // hardcoding SD1.5's 320/[1,2,4,4], so an injected tiny base UNet yields a matching tiny branch.
        _controlEncoder = new ControlNetEncoder<T>(
            inputChannels: 1, baseChannels: _baseUNet.BaseChannels,
            channelMultipliers: _baseUNet.ChannelMultipliers, seed: seed);
    }



    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        // Lazy-preserving Clone (recipe from #1596): delegate to the base UNet's and VAE's own Clone()
        // (preserves materialized weights, reconstructs from actual config) instead of rebuilding a
        // default-scale model and SetParameters(GetParameters()), which mismatches an injected non-default
        // variant and re-randomizes the clone's unmaterialized lazy weights.
        var clone = new ControlNetQRModel<T>(
            baseUNet: (UNetNoisePredictor<T>)_baseUNet.Clone(),
            vae: (StandardVAE<T>)_vae.Clone(),
            conditioner: _conditioner);

        // The control-branch encoder is a separate trainable component (counted
        // in ParameterCount/GetParameters); the constructor builds a fresh one,
        // so transfer this model's trained weights into the clone explicitly —
        // otherwise the clone silently loses the control-branch state.
        if (_controlEncoder.ParameterCount > 0)
        {
            clone._controlEncoder.SetParameters(_controlEncoder.GetParameters());
        }

        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ControlNet-QR", Version = "1.0",
            Description = "ControlNet specialized for embedding QR codes in generated artwork",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "unet-controlnet-qr");
        metadata.SetProperty("base_model", "Stable Diffusion 1.5");
        metadata.SetProperty("text_encoder", "CLIP ViT-L/14");
        metadata.SetProperty("context_dim", 768);
        metadata.SetProperty("control_type", ControlType.QR.ToString());
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return metadata;
    }
}
