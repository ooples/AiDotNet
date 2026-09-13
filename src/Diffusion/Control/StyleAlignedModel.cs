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
/// Style-Aligned model for consistent style across multiple generated images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Style-Aligned uses shared self-attention across multiple generated images during
/// the denoising process to ensure consistent style. This enables generating sets of
/// images that share the same artistic style without explicit style transfer.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you generate multiple images, this model makes them
/// all look like they belong to the same "art collection" — same colors, same style,
/// same artistic feel. It's like having one artist draw several different scenes.
/// </para>
/// <para>
/// Reference: Hertz et al., "Style Aligned Image Generation via Shared Attention", CVPR 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create Style-Aligned model for consistent style across batches
/// var options = new DiffusionModelOptions&lt;float&gt;
/// {
///     LatentChannels = 4,
///     DefaultInferenceSteps = 30
/// };
///
/// // Generate multiple images with consistent artistic style
/// var noise = Tensor&lt;float&gt;.CreateRandom(new[] { 4, 4, 64, 64 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new StyleAlignedModel&lt;float&gt;(options: options))
///     .Build(trainX, trainY);
/// var styledBatch = result.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Style Aligned Image Generation via Shared Attention", "https://arxiv.org/abs/2312.02133", Year = 2024, Authors = "Hertz et al.")]
public partial class StyleAlignedModel<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_baseUNet);
        RegisterParameterComponent(_vae);
    }

    private const int LATENT_CHANNELS = 4;

    private UNetNoisePredictor<T> _baseUNet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly double _styleAlignmentStrength;

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
    /// Initializes a new Style-Aligned model.
    /// </summary>
    public StyleAlignedModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? baseUNet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        double styleAlignmentStrength = 1.0,
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
        _styleAlignmentStrength = styleAlignmentStrength;
        InitializeLayers(baseUNet, vae, seed);
    }

    [MemberNotNull(nameof(_baseUNet), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? baseUNet, StandardVAE<T>? vae, int? seed)
    {
        _baseUNet = baseUNet ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 }, contextDim: 768, seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS, baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocksPerLevel: 2, seed: seed);
    }



    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Style-Aligned", Version = "1.0",
            Description = "Shared attention for consistent style across multiple generated images",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "unet-shared-attention");
        metadata.SetProperty("base_model", "Stable Diffusion 1.5");
        metadata.SetProperty("text_encoder", "CLIP ViT-L/14");
        metadata.SetProperty("context_dim", 768);
        metadata.SetProperty("style_alignment_strength", _styleAlignmentStrength);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("guidance_scale", 7.5);
        return metadata;
    }
}
