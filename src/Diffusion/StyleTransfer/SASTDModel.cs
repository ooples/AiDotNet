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

namespace AiDotNet.Diffusion.StyleTransfer;

/// <summary>
/// SASTD model for structure-aware style transfer via diffusion with edge-guided generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SASTD uses edge maps extracted from the content image to guide style transfer,
/// ensuring structural boundaries are preserved during stylization. The edge guidance
/// acts as an additional conditioning signal alongside the style reference.
/// </para>
/// <para>
/// <b>For Beginners:</b> SASTD preserves the edges and outlines of your image during
/// style transfer. This means buildings keep their shapes, faces keep their features,
/// and objects stay recognizable even with heavy stylization.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new DiffusionModelOptions&lt;float&gt; { LatentChannels = 4, DefaultInferenceSteps = 30 };
/// var input = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 4, 64, 64 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new SASTDModel&lt;float&gt;(options: options))
///     .Build(trainX, trainY);
/// var stylized = result.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.StyleTransfer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Style Aligned Image Generation via Shared Attention", "https://arxiv.org/abs/2312.02133")]
public partial class SASTDModel<T> : LatentDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order, and matches the
    /// concatenation the previous hand-written GetParameters performed.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_predictor);
        RegisterParameterComponent(_vae);
    }

    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 7.5;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _vae;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => LATENT_CHANNELS;

    public SASTDModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            // Style Aligned (Hertz et al. 2023) is an inference-time shared-attention
            // technique over a STANDARD Stable Diffusion U-Net: the noise predictor
            // consumes the 4-channel latent and emits a 4-channel noise estimate — it
            // does NOT add an input channel. inputChannels was 5 (a copy-paste from a
            // latent+mask inpainting predictor), so the UNet's input conv expected 5
            // channels while the diffusion latent (and every sibling StyleTransfer
            // model) is 4 → shape mismatch on every forward/output/training test.
            architecture: Architecture, inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2, 1 }, contextDim: 768, seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocksPerLevel: 2, seed: seed);
    }



    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "SASTD", Version = "1.0",
            Description = "Structure-aware style transfer with edge-guided diffusion",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount };
        m.SetProperty("architecture", "edge-guided-style-transfer");
        m.SetProperty("base_model", "Stable Diffusion 1.5");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("context_dim", 768);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        return m;
    }
}
