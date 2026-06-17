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
/// SD Turbo / SDXL Turbo model for real-time single-step image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SD Turbo and SDXL Turbo are distilled versions of Stable Diffusion and SDXL
/// that can generate images in 1-4 steps using Adversarial Diffusion Distillation (ADD).
/// </para>
/// <para>
/// <b>For Beginners:</b> SD Turbo generates images almost instantly:
///
/// How SD/SDXL Turbo works:
/// 1. Uses the same architecture as SD 1.5 / SDXL
/// 2. Trained with Adversarial Diffusion Distillation (ADD)
/// 3. A discriminator network enforces realism at each step
/// 4. Can generate high-quality images in just 1-4 denoising steps
///
/// Key characteristics:
/// - SD Turbo: 512x512, based on SD 2.1 architecture
/// - SDXL Turbo: 512x512, based on SDXL architecture
/// - 1-4 steps instead of 20-50 (10-50x faster)
/// - No classifier-free guidance needed (guidance scale = 0)
/// - Uses ADD (Adversarial Diffusion Distillation) training
///
/// Advantages:
/// - Near real-time generation (~0.1 seconds)
/// - Single-step generation possible
/// - Same quality as multi-step at much lower latency
///
/// Limitations:
/// - Lower diversity than full multi-step models
/// - Less control via guidance scale (guidance=0 recommended)
/// - Smaller effective resolution than full SDXL
/// - Non-commercial license for Turbo models
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Distilled SD 2.1 / SDXL via Adversarial Diffusion Distillation
/// - SD Turbo: SD 2.1 U-Net (865M params), 1024-dim cross-attention
/// - SDXL Turbo: SDXL U-Net (2.6B params), dual text encoders
/// - Steps: 1-4 (optimal: 1 for speed, 4 for quality)
/// - Guidance scale: 0.0 (no CFG needed)
/// - Resolution: 512x512 (both variants)
/// - Distillation: ADD with adversarial loss + diffusion loss
///
/// Reference: Sauer et al., "Adversarial Diffusion Distillation", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create SD Turbo for ultra-fast generation
/// var turbo = new SDTurboModel&lt;float&gt;();
///
/// // Generate in just 1 step
/// var image = turbo.GenerateFromText(
///     prompt: "A photo of a cat wearing sunglasses",
///     width: 512,
///     height: 512,
///     numInferenceSteps: 1,
///     guidanceScale: 0.0,
///     seed: 42);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.GAN)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Adversarial Diffusion Distillation", "https://arxiv.org/abs/2311.17042", Year = 2023, Authors = "Sauer et al.")]
public class SDTurboModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default image width for SD Turbo generation.
    /// </summary>
    public const int DefaultWidth = 512;

    /// <summary>
    /// Default image height for SD Turbo generation.
    /// </summary>
    public const int DefaultHeight = 512;

    private const int TURBO_LATENT_CHANNELS = 4;

    /// <summary>
    /// Cross-attention dimension (1024 for SD Turbo based on SD 2.1).
    /// </summary>
    private const int TURBO_CROSS_ATTENTION_DIM = 1024;

    /// <summary>
    /// Default guidance scale for Turbo models (0.0 = no CFG, as recommended).
    /// </summary>
    private const double TURBO_DEFAULT_GUIDANCE_SCALE = 0.0;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _unet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly bool _isXLVariant;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => TURBO_LATENT_CHANNELS;

    /// <inheritdoc />
    public override long ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets whether this is the SDXL Turbo variant (true) or SD Turbo variant (false).
    /// </summary>
    public bool IsXLVariant => _isXLVariant;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of SDTurboModel with full customization support.
    /// </summary>
    /// <param name="options">
    /// Configuration options. If null, uses SD Turbo defaults.
    /// </param>
    /// <param name="scheduler">
    /// Custom noise scheduler. If null, creates a DDIM scheduler optimized for few steps.
    /// </param>
    /// <param name="unet">
    /// Custom U-Net. If null, creates the standard distilled SD 2.1 or SDXL U-Net.
    /// </param>
    /// <param name="vae">
    /// Custom VAE. If null, creates the standard SD VAE.
    /// </param>
    /// <param name="conditioner">
    /// Text encoder conditioning module.
    /// </param>
    /// <param name="isXLVariant">
    /// Whether to use SDXL Turbo architecture (default: false = SD Turbo).
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SDTurboModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        bool isXLVariant = false,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear,
                // SDTurbo paper (Sauer et al. 2023, "Adversarial Diffusion Distillation"):
                // single-step generation by design.
                DefaultInferenceSteps = 1,
                // Propagate `seed` so the base RandomGenerator (training-timestep
                // sampling) is deterministic instead of a non-deterministic secure RNG
                // (see SpotDiffusionModel for the detailed rationale).
                Seed = seed
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        _isXLVariant = isXLVariant;

        InitializeLayers(unet, vae, seed);

        SetGuidanceScale(TURBO_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the distilled U-Net and VAE layers,
    /// using custom layers from the user if provided or creating industry-standard
    /// layers from the ADD paper.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        if (_isXLVariant)
        {
            // SDXL Turbo: Distilled SDXL U-Net (~2.6B parameters)
            _unet = unet ?? new UNetNoisePredictor<T>(
                inputChannels: TURBO_LATENT_CHANNELS,
                outputChannels: TURBO_LATENT_CHANNELS,
                baseChannels: 320,
                channelMultipliers: [1, 2, 4],
                numResBlocks: 2,
                attentionResolutions: [4, 2],
                contextDim: 2048,
                architecture: Architecture,
                seed: seed);
        }
        else
        {
            // SD Turbo: Distilled SD 2.1 U-Net (865M parameters)
            _unet = unet ?? new UNetNoisePredictor<T>(
                inputChannels: TURBO_LATENT_CHANNELS,
                outputChannels: TURBO_LATENT_CHANNELS,
                baseChannels: 320,
                channelMultipliers: [1, 2, 4, 4],
                numResBlocks: 2,
                attentionResolutions: [4, 2, 1],
                contextDim: TURBO_CROSS_ATTENTION_DIM,
                architecture: Architecture,
                seed: seed);
        }

        // Standard SD VAE (same for both variants)
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: TURBO_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215,
            seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <inheritdoc />
    public override Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = DefaultWidth,
        int height = DefaultHeight,
        int numInferenceSteps = 1,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? TURBO_DEFAULT_GUIDANCE_SCALE;

        return base.GenerateFromText(
            prompt,
            negativePrompt,
            width,
            height,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    /// <inheritdoc />
    public override Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.5,
        int numInferenceSteps = 2,
        double? guidanceScale = null,
        int? seed = null)
    {
        var effectiveGuidanceScale = guidanceScale ?? TURBO_DEFAULT_GUIDANCE_SCALE;

        return base.ImageToImage(
            inputImage,
            prompt,
            negativePrompt,
            strength,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _unet.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        for (int i = 0; i < unetParams.Length; i++)
        {
            combined[i] = unetParams[i];
        }

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[unetParams.Length + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = checked((int)_unet.ParameterCount);
        var vaeCount = checked((int)_vae.ParameterCount);

        if (parameters.Length != unetCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < unetCount; i++)
        {
            unetParams[i] = parameters[i];
        }

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[unetCount + i];
        }

        _unet.SetParameters(unetParams);
        _vae.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        // Clone the U-Net through its OWN Clone(), which reconstructs from the
        // source predictor's actual architecture fields (input/output channels,
        // base channels, channel multipliers, res-block count, attention
        // resolutions, context dim, heads, input height) and copies weights via a
        // paired per-layer walk. The previous code rebuilt clonedUnet from
        // HARDCODED SD-/SDXL-Turbo defaults (baseChannels 320, [1,2,4,4],
        // numResBlocks 2, contextDim 1024) and then SetParameters'd the source's
        // weights into it — correct only when the model used the production default
        // U-Net. With a CUSTOM U-Net (e.g. a smaller test configuration) the
        // architectures differed, so SetParameters mis-distributed the source's
        // shorter parameter vector and the clone's later layers kept their random
        // init — diverging from the original despite "identical" parameters.
        var clonedUnet = (UNetNoisePredictor<T>)_unet.Clone();

        return new SDTurboModel<T>(
            unet: clonedUnet,
            vae: (StandardVAE<T>)_vae.Clone(),
            conditioner: _conditioner,
            isXLVariant: _isXLVariant);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var variant = _isXLVariant ? "SDXL Turbo" : "SD Turbo";
        var metadata = new ModelMetadata<T>
        {
            Name = variant,
            Version = "1.0",
            Description = $"{variant} distilled single/few-step image generation via Adversarial Diffusion Distillation",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "distilled-latent-diffusion");
        metadata.SetProperty("distillation_method", "ADD");
        metadata.SetProperty("optimal_steps", 1);
        metadata.SetProperty("max_recommended_steps", 4);
        metadata.SetProperty("guidance_scale", 0.0);
        metadata.SetProperty("is_xl_variant", _isXLVariant);
        metadata.SetProperty("latent_channels", TURBO_LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);

        return metadata;
    }

    #endregion
}
