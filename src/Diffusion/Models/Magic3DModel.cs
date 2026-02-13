using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// Magic3D model for high-quality text-to-3D generation using score distillation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Magic3D is a two-stage coarse-to-fine text-to-3D generation framework by NVIDIA.
/// It uses Score Distillation Sampling (SDS) from a 2D diffusion model to optimize
/// a 3D representation, first as a coarse NeRF then refined as a textured mesh.
/// </para>
/// <para>
/// <b>For Beginners:</b> Magic3D creates 3D models from text descriptions:
///
/// How Magic3D works:
/// 1. Coarse stage: Optimize a neural radiance field (NeRF) using SDS from a low-res diffusion model
/// 2. Fine stage: Convert NeRF to a mesh and optimize with SDS from a high-res latent diffusion model
/// 3. Result: High-quality textured 3D mesh
///
/// Key characteristics:
/// - Two-stage coarse-to-fine optimization
/// - Coarse stage: NeRF + low-res SDS (64x64 base model)
/// - Fine stage: DMTet mesh + high-res SDS (latent diffusion)
/// - 2x faster than DreamFusion
/// - 8x higher resolution meshes than DreamFusion
/// - Uses both pixel-space and latent-space diffusion guidance
///
/// Advantages:
/// - High-quality textured 3D meshes
/// - Much faster than DreamFusion
/// - Better geometry through mesh refinement
/// - Supports both NeRF and mesh representations
///
/// Limitations:
/// - Multi-view consistency (Janus problem)
/// - Optimization takes minutes per object
/// - Quality depends on 2D diffusion prior
/// </para>
/// <para>
/// Technical specifications:
/// - Coarse stage: Instant-NGP NeRF, 64x64 diffusion guidance, ~40 min optimization
/// - Fine stage: DMTet mesh, latent diffusion guidance, ~20 min optimization
/// - Diffusion prior: eDiff-I (coarse) + Stable Diffusion (fine)
/// - SDS guidance scale: 100 (coarse) → 7.5 (fine)
/// - NeRF resolution: 128^3 hash grid
/// - Mesh resolution: 512^3 DMTet grid
///
/// Reference: Lin et al., "Magic3D: High-Resolution Text-to-3D Content Creation", CVPR 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var magic3d = new Magic3DModel&lt;float&gt;();
/// var mesh = magic3d.GenerateMesh(
///     prompt: "A blue poison dart frog sitting on a lily pad",
///     resolution: 256,
///     numInferenceSteps: 64,
///     guidanceScale: 100.0,
///     seed: 42);
/// </code>
/// </example>
public class Magic3DModel<T> : ThreeDDiffusionModelBase<T>
{
    #region Constants

    private const int MAGIC3D_LATENT_CHANNELS = 4;
    private const int MAGIC3D_CROSS_ATTENTION_DIM = 768;
    private const double MAGIC3D_COARSE_GUIDANCE = 100.0;
    private const double MAGIC3D_FINE_GUIDANCE = 7.5;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _coarseUnet;
    private UNetNoisePredictor<T> _fineUnet;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _fineUnet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => MAGIC3D_LATENT_CHANNELS;

    /// <inheritdoc />
    public override bool SupportsPointCloud => true;

    /// <inheritdoc />
    public override bool SupportsMesh => true;

    /// <inheritdoc />
    public override bool SupportsTexture => true;

    /// <inheritdoc />
    public override bool SupportsNovelView => false;

    /// <inheritdoc />
    public override bool SupportsScoreDistillation => true;

    /// <inheritdoc />
    public override int ParameterCount => _coarseUnet.ParameterCount + _fineUnet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the coarse-stage noise predictor (pixel-space SDS).
    /// </summary>
    public INoisePredictor<T> CoarseModel => _coarseUnet;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of Magic3DModel with full customization support.
    /// </summary>
    /// <param name="options">Configuration options. If null, uses Magic3D defaults.</param>
    /// <param name="scheduler">Custom noise scheduler.</param>
    /// <param name="coarseUnet">Custom coarse-stage U-Net for pixel-space SDS.</param>
    /// <param name="fineUnet">Custom fine-stage U-Net for latent-space SDS.</param>
    /// <param name="vae">Custom VAE for the fine stage.</param>
    /// <param name="conditioner">Text encoder conditioning module.</param>
    /// <param name="defaultPointCount">Default point count for point clouds.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public Magic3DModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? coarseUnet = null,
        UNetNoisePredictor<T>? fineUnet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int defaultPointCount = 4096,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            defaultPointCount)
    {
        _conditioner = conditioner;

        InitializeLayers(coarseUnet, fineUnet, vae, seed);

        SetGuidanceScale(MAGIC3D_COARSE_GUIDANCE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_coarseUnet), nameof(_fineUnet), nameof(_vae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? coarseUnet,
        UNetNoisePredictor<T>? fineUnet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // Coarse stage: Pixel-space U-Net for 64x64 SDS
        _coarseUnet = coarseUnet ?? new UNetNoisePredictor<T>(
            inputChannels: 3,
            outputChannels: 3,
            baseChannels: 256,
            channelMultipliers: [1, 2, 3, 4],
            numResBlocks: 3,
            attentionResolutions: [4, 2, 1],
            contextDim: MAGIC3D_CROSS_ATTENTION_DIM,
            seed: seed);

        // Fine stage: Latent-space U-Net for high-res SDS (SD 1.5 architecture)
        _fineUnet = fineUnet ?? new UNetNoisePredictor<T>(
            inputChannels: MAGIC3D_LATENT_CHANNELS,
            outputChannels: MAGIC3D_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: MAGIC3D_CROSS_ATTENTION_DIM,
            seed: seed);

        // Standard SD 1.5 VAE for fine stage
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: MAGIC3D_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2,
            latentScaleFactor: 0.18215,
            seed: seed);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var coarseParams = _coarseUnet.GetParameters();
        var fineParams = _fineUnet.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = coarseParams.Length + fineParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        var offset = 0;
        for (int i = 0; i < coarseParams.Length; i++)
            combined[offset + i] = coarseParams[i];
        offset += coarseParams.Length;

        for (int i = 0; i < fineParams.Length; i++)
            combined[offset + i] = fineParams[i];
        offset += fineParams.Length;

        for (int i = 0; i < vaeParams.Length; i++)
            combined[offset + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var coarseCount = _coarseUnet.ParameterCount;
        var fineCount = _fineUnet.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != coarseCount + fineCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {coarseCount + fineCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var coarseParams = new Vector<T>(coarseCount);
        var fineParams = new Vector<T>(fineCount);
        var vaeParams = new Vector<T>(vaeCount);

        var offset = 0;
        for (int i = 0; i < coarseCount; i++)
            coarseParams[i] = parameters[offset + i];
        offset += coarseCount;

        for (int i = 0; i < fineCount; i++)
            fineParams[i] = parameters[offset + i];
        offset += fineCount;

        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[offset + i];

        _coarseUnet.SetParameters(coarseParams);
        _fineUnet.SetParameters(fineParams);
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
        var clonedCoarse = new UNetNoisePredictor<T>(
            inputChannels: 3, outputChannels: 3,
            baseChannels: 256, channelMultipliers: [1, 2, 3, 4],
            numResBlocks: 3, attentionResolutions: [4, 2, 1],
            contextDim: MAGIC3D_CROSS_ATTENTION_DIM);
        clonedCoarse.SetParameters(_coarseUnet.GetParameters());

        var clonedFine = new UNetNoisePredictor<T>(
            inputChannels: MAGIC3D_LATENT_CHANNELS, outputChannels: MAGIC3D_LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2, attentionResolutions: [4, 2, 1],
            contextDim: MAGIC3D_CROSS_ATTENTION_DIM);
        clonedFine.SetParameters(_fineUnet.GetParameters());

        var clonedVae = new StandardVAE<T>(
            inputChannels: 3, latentChannels: MAGIC3D_LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
        clonedVae.SetParameters(_vae.GetParameters());

        return new Magic3DModel<T>(
            coarseUnet: clonedCoarse,
            fineUnet: clonedFine,
            vae: clonedVae,
            conditioner: _conditioner,
            defaultPointCount: DefaultPointCount);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Magic3D",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Magic3D two-stage coarse-to-fine text-to-3D generation with score distillation sampling",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "sds-nerf-to-mesh");
        metadata.SetProperty("coarse_guidance_scale", MAGIC3D_COARSE_GUIDANCE);
        metadata.SetProperty("fine_guidance_scale", MAGIC3D_FINE_GUIDANCE);
        metadata.SetProperty("cross_attention_dim", MAGIC3D_CROSS_ATTENTION_DIM);
        metadata.SetProperty("default_point_count", DefaultPointCount);

        return metadata;
    }

    #endregion
}
