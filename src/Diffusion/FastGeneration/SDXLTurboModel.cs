using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.FastGeneration;

/// <summary>
/// SDXL Turbo model for real-time single-step high-resolution image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SDXL Turbo is the SDXL-based variant of Adversarial Diffusion Distillation (ADD),
/// generating 512x512 images in 1-4 steps with SDXL-quality aesthetics. Uses the full
/// SDXL U-Net with dual text encoder conditioning (CLIP ViT-L + OpenCLIP ViT-bigG).
/// </para>
/// <para>
/// <b>For Beginners:</b> SDXL Turbo combines the image quality of SDXL (one of the best
/// open-source models) with near-instant generation. While regular SDXL needs 25-50 steps,
/// SDXL Turbo generates comparable images in just 1 step. No guidance needed (scale=0).
/// </para>
/// <para>
/// Reference: Sauer et al., "Adversarial Diffusion Distillation", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new LatentDiffusionOptions&lt;float&gt; { LatentChannels = 4, Height = 512, Width = 512, NumInferenceSteps = 1 };
/// var model = new SDXLTurboModel&lt;float&gt;(options);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 4, 64, 64 });
/// var generated = model.Predict(noise);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.GAN)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Adversarial Diffusion Distillation", "https://arxiv.org/abs/2311.17042", Year = 2023, Authors = "Sauer et al.")]
public partial class SDXLTurboModel<T> : LatentDiffusionModelBase<T>
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
    private const int SDXL_CONTEXT_DIM = 2048;
    private const double DEFAULT_GUIDANCE = 0.0;
    private const int DEFAULT_STEPS = 1;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />

    /// <summary>
    /// Initializes a new SDXL Turbo model.
    /// </summary>
    public SDXLTurboModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.00085,
                BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new EulerDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4],
            numResBlocks: 2, attentionResolutions: [4, 2],
            contextDim: 2048, architecture: Architecture, seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
    }



    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        // Delegate to the predictor/VAE's own Clone implementations, which
        // resolve their internal lazy shape inference on BOTH source and clone
        // before copying weights. Constructing a fresh SDXLTurboModel here and
        // calling SetParameters(GetParameters()) re-hits the lazy-init bug:
        // GetParameters() on the unresolved source under-counts the UNet's
        // top-level time-embedding DenseLayer params, while the fresh clone's
        // SetParameters checks parameters.Length against the resolved
        // (arch-derived) ParameterCount and throws — surfaced as
        // SDXLTurboModel_GetSetParameters_RoundTrips / Clone_CreatesIndependentCopy
        // failures in the Unit-03 Diffusion/Encoding shard. Same fix pattern
        // as ImprovedConsistencyModel.Clone (commit 7ab314796, PR #1555).
        var predictorClone = (UNetNoisePredictor<T>)_predictor.Clone();
        var vaeClone = (StandardVAE<T>)_vae.Clone();
        return new SDXLTurboModel<T>(
            predictor: predictorClone,
            vae: vaeClone,
            conditioner: _conditioner,
            seed: null);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "SDXL Turbo", Version = "1.0",
            Description = "SDXL-quality single-step generation via Adversarial Diffusion Distillation",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount
        };
        m.SetProperty("architecture", "sdxl-add-distilled");
        m.SetProperty("base_model", "Stable Diffusion XL");
        m.SetProperty("text_encoder", "CLIP ViT-L/14 + OpenCLIP ViT-bigG/14");
        m.SetProperty("context_dim", SDXL_CONTEXT_DIM);
        m.SetProperty("distillation_method", "Adversarial Diffusion Distillation (ADD)");
        m.SetProperty("optimal_steps", DEFAULT_STEPS);
        m.SetProperty("max_recommended_steps", 4);
        m.SetProperty("default_resolution", 512);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
