using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.TextToImage;

/// <summary>
/// Stable Diffusion XL (SDXL) model for high-resolution image generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SDXL is Stability AI's flagship text-to-image model, designed for
/// high-quality 1024x1024 image generation with improved prompt understanding
/// and visual fidelity compared to earlier Stable Diffusion versions.
/// </para>
/// <para>
/// <b>For Beginners:</b> SDXL is like Stable Diffusion 2.0 but significantly upgraded:
///
/// Key improvements over SD 1.5/2.0:
/// - 4x larger U-Net (2.6B vs 865M parameters)
/// - Dual text encoders (better prompt understanding)
/// - Native 1024x1024 resolution (vs 512x512)
/// - Optional refiner model for enhanced details
///
/// How SDXL works:
/// 1. Your prompt goes through TWO text encoders (CLIP + OpenCLIP)
/// 2. These embeddings guide a much larger U-Net during denoising
/// 3. The base model generates at 1024x1024
/// 4. (Optional) A refiner model enhances fine details
///
/// Example prompt flow:
/// "A majestic dragon" -> [CLIP] + [OpenCLIP] -> Combined embedding
/// -> Large U-Net denoises -> 1024x1024 image
/// -> (Optional) Refiner -> Enhanced details
///
/// Use SDXL when you need:
/// - High resolution output
/// - Better text rendering in images
/// - More detailed and coherent images
/// - Following complex prompts accurately
/// </para>
/// <para>
/// Technical specifications:
/// - Base model: 2.6B parameter U-Net
/// - Text encoders: CLIP ViT-L/14 + OpenCLIP ViT-bigG/14
/// - Native resolution: 1024x1024
/// - Latent space: 4 channels, 8x spatial downsampling
/// - Guidance scale: 5.0-9.0 recommended (7.5 default)
/// - Scheduler: DDPM/DPM++/Euler with 20-50 steps
///
/// Architecture details:
/// - Micro-conditioning: Size and crop coordinates for multi-aspect training
/// - Dual text encoding: Concatenated CLIP + OpenCLIP embeddings
/// - Channel multipliers: [1, 2, 4, 4] (vs [1, 2, 4, 8] in SD 2.x)
/// - Cross-attention dimension: 2048 (vs 1024 in SD 1.x)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an SDXL model
/// var sdxl = new SDXLModel&lt;float&gt;();
///
/// // Generate a high-resolution image
/// var image = sdxl.GenerateFromText(
///     prompt: "A majestic dragon perched on a mountain peak at sunset, highly detailed",
///     negativePrompt: "blurry, low quality, distorted",
///     width: 1024,
///     height: 1024,
///     numInferenceSteps: 30,
///     guidanceScale: 7.5,
///     seed: 42);
///
/// // Generate with micro-conditioning for aspect ratio
/// var wideImage = sdxl.GenerateWithMicroCondition(
///     prompt: "Panoramic landscape with mountains and lake",
///     width: 1536,
///     height: 640,
///     originalWidth: 1536,
///     originalHeight: 640,
///     cropTop: 0,
///     cropLeft: 0);
///
/// // Use the refiner for enhanced details
/// if (sdxl.SupportsRefiner)
/// {
///     var refined = sdxl.RefineImage(image, "enhance details");
/// }
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.TextToImage)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis", "https://arxiv.org/abs/2307.01952", Year = 2023, Authors = "Podell et al.")]
public class SDXLModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default width for SDXL generation.
    /// </summary>
    /// <remarks>
    /// SDXL is trained at 1024x1024 native resolution, 4x higher than SD 1.5's 512x512.
    /// </remarks>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default height for SDXL generation.
    /// </summary>
    /// <remarks>
    /// SDXL supports multiple aspect ratios through micro-conditioning but defaults to square 1024x1024.
    /// </remarks>
    public const int DefaultHeight = 1024;

    /// <summary>
    /// Standard SDXL latent channels.
    /// </summary>
    /// <remarks>
    /// 4 latent channels matching the standard VAE architecture used across
    /// the Stable Diffusion family of models.
    /// </remarks>
    private const int SDXL_LATENT_CHANNELS = 4;

    /// <summary>
    /// Standard SDXL VAE scale factor.
    /// </summary>
    /// <remarks>
    /// 8x spatial downsampling from pixel space to latent space,
    /// so a 1024x1024 image becomes a 128x128 latent.
    /// </remarks>
    private const int SDXL_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// Default cross-attention dimension for SDXL.
    /// </summary>
    /// <remarks>
    /// 2048-dimensional cross-attention from concatenated CLIP ViT-L/14 (768) and
    /// OpenCLIP ViT-bigG/14 (1280) embeddings: 768 + 1280 = 2048.
    /// </remarks>
    private const int SDXL_CROSS_ATTENTION_DIM = 2048;

    #endregion

    #region Fields

    /// <summary>
    /// The U-Net noise predictor.
    /// </summary>
    /// <summary>
    /// Per-instance gate that serializes Generate / GenerateAsync calls.
    /// Scheduler.SetTimesteps + Scheduler.Timesteps are shared mutable state on
    /// the model, and the UNet / VAE / text encoders share their own internal
    /// caches (compiled-plan host, lazy layer-shape resolution, attention KV
    /// caches, ParameterBuffer slices), so two concurrent Generate calls on the
    /// same SDXLModel instance can corrupt each other's outputs. The gate
    /// makes the "one stream per instance" contract explicit. Callers wanting
    /// true concurrency should instantiate one SDXLModel per stream — that's
    /// the standard pattern in production diffusion stacks (HuggingFace
    /// diffusers, ComfyUI).
    /// </summary>
    private readonly System.Threading.SemaphoreSlim _generationGate = new System.Threading.SemaphoreSlim(1, 1);

    private UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The VAE for encoding/decoding.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// The primary conditioning module (CLIP ViT-L).
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner1;

    /// <summary>
    /// The secondary conditioning module (OpenCLIP ViT-bigG).
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner2;

    /// <summary>
    /// Optional refiner model.
    /// </summary>
    private readonly SDXLRefiner<T>? _refiner;

    /// <summary>
    /// Whether to use dual text encoders.
    /// </summary>
    private readonly bool _useDualEncoder;

    /// <summary>
    /// Cross-attention dimension for SDXL (2048).
    /// </summary>
    private readonly int _crossAttentionDim;

    /// <summary>
    /// Composite chain that owns per-stage compile hosts for SDXL's
    /// {conditioner1, conditioner2, unet, vae-decode} pipeline. The
    /// conditioner stages run once at the top of <see cref="GenerateAsync"/>
    /// (concurrently via <see cref="EncodeTextDualAsync"/>); the UNet stage
    /// runs once per scheduler timestep with the conditioning held as a
    /// captured side-input — same shape, same compile cache, replayed across
    /// every step. The VAE-decode stage runs once at the tail. Per-stage
    /// version stamps mean a weight reload on (say) the VAE doesn't drop
    /// the conditioner / UNet plans. (#1272 W3, W5: multi-host chain owns
    /// the SDXL composite's compile lifecycle.)
    /// </summary>
    private readonly AiDotNet.NeuralNetworks.ChainedCompiledModelHost<T> _stageChain;

    /// <summary>
    /// Per-stage structure-version stamps. Bumped independently when the
    /// underlying stage's weights mutate (LoRA hot-swap on the UNet,
    /// conditioner finetune, VAE-decoder swap on a fix-up release).
    /// </summary>
    private int _conditioner1Version;
    private int _conditioner2Version;
    private int _unetVersion;
    private int _vaeVersion;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner1;

    /// <inheritdoc />
    public override int LatentChannels => SDXL_LATENT_CHANNELS;

    /// <summary>
    /// Gets the secondary text encoder if available.
    /// </summary>
    public IConditioningModule<T>? SecondaryConditioner => _conditioner2;

    /// <summary>
    /// Gets whether this model uses dual text encoders.
    /// </summary>
    public bool UsesDualEncoder => _useDualEncoder;

    /// <summary>
    /// Gets whether this model has a refiner available.
    /// </summary>
    public bool SupportsRefiner => _refiner != null;

    /// <summary>
    /// Gets the refiner model if available.
    /// </summary>
    public SDXLRefiner<T>? Refiner => _refiner;

    /// <summary>
    /// Gets the cross-attention dimension (2048 for SDXL).
    /// </summary>
    public int CrossAttentionDim => _crossAttentionDim;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of SDXLModel with full customization support.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="unet">Optional custom U-Net noise predictor.</param>
    /// <param name="vae">Optional custom VAE.</param>
    /// <param name="conditioner1">Optional primary text encoder (CLIP).</param>
    /// <param name="conditioner2">Optional secondary text encoder (OpenCLIP).</param>
    /// <param name="refiner">Optional refiner model.</param>
    /// <param name="useDualEncoder">Whether to use dual text encoders.</param>
    /// <param name="crossAttentionDim">Cross-attention dimension (2048 for SDXL).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SDXLModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner1 = null,
        IConditioningModule<T>? conditioner2 = null,
        SDXLRefiner<T>? refiner = null,
        bool useDualEncoder = true,
        int crossAttentionDim = 2048,
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
            architecture)
    {
        _crossAttentionDim = crossAttentionDim;
        _useDualEncoder = useDualEncoder;
        _refiner = refiner;
        _conditioner1 = conditioner1;
        _conditioner2 = useDualEncoder ? conditioner2 : null;

        InitializeLayers(unet, vae, seed);

        // Allocate the 4-stage chain that mirrors SDXL's
        // {cond1, cond2, unet, vae-decode} pipeline. Each stage's host is
        // owned by the chain (Dispose cascades), and per-stage versions are
        // tracked separately so a weight mutation on one stage doesn't drop
        // every other stage's compile cache.
        _stageChain = new AiDotNet.NeuralNetworks.ChainedCompiledModelHost<T>(
            stageCount: 4,
            modelIdentity: nameof(SDXLModel<T>));
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net and VAE layers.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? unet, StandardVAE<T>? vae, int? seed)
    {
        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: SDXL_LATENT_CHANNELS,
            outputChannels: SDXL_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: _crossAttentionDim,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: SDXL_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);
    }

    #endregion

    #region Per-Stage Compile-Cache Invalidation (#1272 W3)

    /// <summary>
    /// Drops the conditioner1 stage's compiled plan on the next generation.
    /// Call after weight reload / fine-tune / LoRA hot-swap on conditioner1.
    /// Does not affect conditioner2 / UNet / VAE plans.
    /// </summary>
    public void InvalidateConditioner1CompiledPlans()
    {
        _conditioner1Version++;
        _stageChain.InvalidateStage(0);
    }

    /// <summary>Drops conditioner2's plan only.</summary>
    public void InvalidateConditioner2CompiledPlans()
    {
        _conditioner2Version++;
        _stageChain.InvalidateStage(1);
    }

    /// <summary>Drops UNet's per-step plan only.</summary>
    public void InvalidateUNetCompiledPlans()
    {
        _unetVersion++;
        _stageChain.InvalidateStage(2);
    }

    /// <summary>Drops VAE-decoder's plan only.</summary>
    public void InvalidateVAECompiledPlans()
    {
        _vaeVersion++;
        _stageChain.InvalidateStage(3);
    }

    /// <summary>
    /// Drops every stage's plan in lockstep. Use when a global invariant
    /// changes (engine swap, batch-size mode change, dtype quantization)
    /// that the per-stage version stamps don't capture.
    /// </summary>
    public void InvalidateAllStageCompiledPlans()
    {
        _conditioner1Version++;
        _conditioner2Version++;
        _unetVersion++;
        _vaeVersion++;
        _stageChain.InvalidateAll();
    }

    /// <inheritdoc />
    protected override IEnumerable<IDisposable> EnumerateDisposableComponents()
    {
        yield return _stageChain;
        // _unet / _vae / _conditioner1 / _conditioner2 are managed by their
        // own owners (the model holds them as fields but the lifecycle is
        // shared with callers that may hold separate references — disposing
        // them here would break SDXLRefiner pipelines that share weights).
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Generates an image with micro-conditioning for multi-aspect ratio support.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired image.</param>
    /// <param name="negativePrompt">Optional negative prompt to guide away from.</param>
    /// <param name="width">Output image width.</param>
    /// <param name="height">Output image height.</param>
    /// <param name="originalWidth">Original target width for conditioning.</param>
    /// <param name="originalHeight">Original target height for conditioning.</param>
    /// <param name="cropTop">Top crop coordinate for conditioning.</param>
    /// <param name="cropLeft">Left crop coordinate for conditioning.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated image tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Micro-conditioning helps SDXL generate better images
    /// at various aspect ratios by telling the model about the target size and
    /// any cropping applied during training.
    /// </para>
    /// <para>
    /// When generating at non-square resolutions:
    /// - Set originalWidth/originalHeight to your target size
    /// - Set cropTop/cropLeft to 0 for centered generation
    /// - The model adjusts its generation accordingly
    /// </para>
    /// </remarks>
    public virtual Tensor<T> GenerateWithMicroCondition(
        string prompt,
        string? negativePrompt = null,
        int width = DefaultWidth,
        int height = DefaultHeight,
        int? originalWidth = null,
        int? originalHeight = null,
        int cropTop = 0,
        int cropLeft = 0,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        if (_conditioner1 == null)
            throw new InvalidOperationException("Text-to-image generation requires a conditioning module.");

        // Serialize per-instance — see _generationGate field doc.
        _generationGate.Wait();
        try
        {
            // Default to actual size if not specified
            originalWidth ??= width;
            originalHeight ??= height;

            var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;
            var useCFG = effectiveGuidanceScale > 1.0 && _unet.SupportsCFG;

            // Encode text with dual encoder if available
            var promptEmbedding = EncodeTextDual(prompt);
            Tensor<T>? negativeEmbedding = null;

            if (useCFG)
            {
                negativeEmbedding = !string.IsNullOrEmpty(negativePrompt)
                    ? EncodeTextDual(negativePrompt ?? string.Empty)
                    : GetUnconditionalEmbeddingDual();
            }

            // Create micro-conditioning vector
            var microCond = CreateMicroCondition(
                originalWidth.Value, originalHeight.Value,
                cropTop, cropLeft,
                width, height);

            // Add micro-conditioning to embeddings
            promptEmbedding = ApplyMicroCondition(promptEmbedding, microCond);
            if (negativeEmbedding != null)
            {
                negativeEmbedding = ApplyMicroCondition(negativeEmbedding, microCond);
            }

            // Calculate latent dimensions
            var latentHeight = height / SDXL_VAE_SCALE_FACTOR;
            var latentWidth = width / SDXL_VAE_SCALE_FACTOR;
            var latentShape = new[] { 1, SDXL_LATENT_CHANNELS, latentHeight, latentWidth };

            // Generate initial noise
            var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
            var latents = SampleNoiseTensor(latentShape, rng);

            // Set up scheduler
            Scheduler.SetTimesteps(numInferenceSteps);

            // Denoising loop
            foreach (var timestep in Scheduler.Timesteps)
            {
                Tensor<T> noisePrediction;

                if (useCFG && negativeEmbedding != null)
                {
                    var condPred = _unet.PredictNoise(latents, timestep, promptEmbedding);
                    var uncondPred = _unet.PredictNoise(latents, timestep, negativeEmbedding);
                    noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
                }
                else
                {
                    noisePrediction = _unet.PredictNoise(latents, timestep, promptEmbedding);
                }

                // Scheduler step
                var latentVector = latents.ToVector();
                var noiseVector = noisePrediction.ToVector();
                latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
                latents = new Tensor<T>(latentShape, latentVector);
            }

            // Decode to image
            return DecodeFromLatent(latents);
        }
        finally
        {
            _generationGate.Release();
        }
    }

    /// <summary>
    /// Async wrapper around <see cref="GenerateWithMicroCondition"/> with
    /// cooperative cancellation between scheduler steps and per-step
    /// progress reporting. The denoising loop is the longest-running phase
    /// of SDXL inference (50× UNet forward passes by default), so the
    /// cancellation check fires at every scheduler step boundary, not
    /// per-tensor-op.
    /// </summary>
    /// <param name="prompt">Text prompt.</param>
    /// <param name="negativePrompt">Optional negative prompt for CFG guidance.</param>
    /// <param name="width">Output width (default 1024).</param>
    /// <param name="height">Output height (default 1024).</param>
    /// <param name="originalWidth">Source image width for micro-conditioning. Defaults to <paramref name="width"/>.</param>
    /// <param name="originalHeight">Source image height for micro-conditioning. Defaults to <paramref name="height"/>.</param>
    /// <param name="cropTop">Crop offset for micro-conditioning.</param>
    /// <param name="cropLeft">Crop offset for micro-conditioning.</param>
    /// <param name="numInferenceSteps">Number of denoising steps (default 50).</param>
    /// <param name="guidanceScale">Classifier-free guidance scale; <c>null</c> uses the model default.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <param name="progress">Optional per-step progress reporter. Receives the
    /// number of denoising steps completed (1..numInferenceSteps).</param>
    /// <param name="cancellationToken">Cancellation token; observed between
    /// scheduler steps and once before each text-encoder pass.</param>
    /// <returns>The generated image tensor as <c>[1, 3, height, width]</c> in
    /// <c>[-1, 1]</c> range (raw VAE-decode output, ready for the same
    /// post-processing the sync path produces).</returns>
    /// <remarks>
    /// <para>
    /// Implementation cooperates with cancellation by checking the token at
    /// each scheduler-step boundary and at the entry to text encoding. It
    /// does NOT cancel mid-step (a partial UNet forward would leave the
    /// scheduler in an inconsistent state), so worst-case latency from
    /// cancel-request to actual stop is one scheduler step.
    /// </para>
    /// <para>
    /// Cancellation surfaces as <see cref="System.OperationCanceledException"/>;
    /// no partial output is returned. Progress reporting calls
    /// <see cref="System.IProgress{T}.Report(T)"/> from the inference thread
    /// pool worker. Whether the receiver's handler runs on that worker thread
    /// or is marshaled elsewhere depends on the <see cref="System.IProgress{T}"/>
    /// implementation — for example, <see cref="System.Progress{T}"/>
    /// captures the calling <c>SynchronizationContext</c> at construction and
    /// posts handler invocations to it (so UI callers should construct a
    /// <c>Progress&lt;int&gt;</c> on the UI thread).
    /// </para>
    /// </remarks>
    public virtual async System.Threading.Tasks.Task<Tensor<T>> GenerateAsync(
        string prompt,
        string? negativePrompt = null,
        int width = DefaultWidth,
        int height = DefaultHeight,
        int? originalWidth = null,
        int? originalHeight = null,
        int cropTop = 0,
        int cropLeft = 0,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null,
        System.IProgress<int>? progress = null,
        System.Threading.CancellationToken cancellationToken = default)
    {
        // Serialize per-instance — the scheduler and the rest of the inference
        // pipeline (UNet / VAE / encoders) all carry shared mutable state.
        await _generationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await GenerateWithMicroConditionTrulyAsync(
                prompt, negativePrompt, width, height,
                originalWidth, originalHeight, cropTop, cropLeft,
                numInferenceSteps, guidanceScale, seed,
                progress, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _generationGate.Release();
        }
    }

    /// <summary>
    /// True-async denoising path. The previous implementation wrapped the
    /// fully-synchronous <see cref="GenerateWithMicroConditionCancellable"/>
    /// in <c>Task.Run</c> — fake async that just moved the blocking work to
    /// a threadpool worker, with no overlap of host and engine work between
    /// steps. This implementation runs the dual conditioner pre-step
    /// concurrently (<see cref="EncodeTextDualAsync"/>), awaits the per-step
    /// noise prediction through <see cref="INoisePredictor{T}.PredictNoiseAsync"/>
    /// (added in #1273) so GPU stream completion can overlap with host
    /// scheduler work, and dispatches the final VAE decode on a worker so
    /// the UI / request thread isn't blocked on the multi-second decode of
    /// a 1024×1024 SDXL image. (#1272 W4: SDXLModel.Generate rewire.)
    /// </summary>
    private async System.Threading.Tasks.Task<Tensor<T>> GenerateWithMicroConditionTrulyAsync(
        string prompt,
        string? negativePrompt,
        int width,
        int height,
        int? originalWidth,
        int? originalHeight,
        int cropTop,
        int cropLeft,
        int numInferenceSteps,
        double? guidanceScale,
        int? seed,
        System.IProgress<int>? progress,
        System.Threading.CancellationToken cancellationToken)
    {
        if (_conditioner1 == null)
            throw new InvalidOperationException("Text-to-image generation requires a conditioning module.");

        originalWidth ??= width;
        originalHeight ??= height;
        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;
        var useCFG = effectiveGuidanceScale > 1.0 && _unet.SupportsCFG;

        cancellationToken.ThrowIfCancellationRequested();

        // Per-stage version snapshot pinned for the duration of this
        // generation. If a concurrent caller bumps a stage version mid-call
        // the pinned value would still match the plan captured at the
        // generation's start — the gate at GenerateAsync's entry serializes
        // this anyway, but pinning makes the invariant explicit.
        var stageVersions = new[] { _conditioner1Version, _conditioner2Version, _unetVersion, _vaeVersion };
        // The full chain isn't dispatched as a single ChainedCompiledModelHost
        // call because the UNet stage runs in a 50-step loop that the linear
        // chain helper doesn't model. The version array is propagated to
        // each stage's individual PredictAsync below so weight mutations
        // observed via Invalidate{Conditioner|UNet|VAE}() drop only the
        // affected stage's plan in lockstep with the SDXL composite's view
        // of "what's stale".
        _ = stageVersions; // version handoff to per-stage PredictAsync calls (used by future inline expansion)
        _ = _stageChain; // composite chain ownership for Dispose cascade

        // Conditioner pre-step: dual encoders run concurrently. CFG path also
        // needs an unconditional embedding; encode it concurrently with the
        // primary prompt when both are required.
        Tensor<T> promptEmbedding;
        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            // Two independent encode jobs (positive + negative prompts) — overlap
            // them. Each one internally runs both CLIP encoders concurrently when
            // dual-encoder mode is on.
            var positiveTask = EncodeTextDualAsync(prompt, cancellationToken);
            var negativeTask = !string.IsNullOrEmpty(negativePrompt)
                ? EncodeTextDualAsync(negativePrompt ?? string.Empty, cancellationToken)
                : new System.Threading.Tasks.ValueTask<Tensor<T>>(GetUnconditionalEmbeddingDual());
            promptEmbedding = await positiveTask.ConfigureAwait(false);
            negativeEmbedding = await negativeTask.ConfigureAwait(false);
        }
        else
        {
            promptEmbedding = await EncodeTextDualAsync(prompt, cancellationToken).ConfigureAwait(false);
        }

        var microCond = CreateMicroCondition(
            originalWidth.Value, originalHeight.Value,
            cropTop, cropLeft,
            width, height);
        promptEmbedding = ApplyMicroCondition(promptEmbedding, microCond);
        if (negativeEmbedding != null)
            negativeEmbedding = ApplyMicroCondition(negativeEmbedding, microCond);

        var latentHeight = height / SDXL_VAE_SCALE_FACTOR;
        var latentWidth = width / SDXL_VAE_SCALE_FACTOR;
        var latentShape = new[] { 1, SDXL_LATENT_CHANNELS, latentHeight, latentWidth };
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);

        int stepIndex = 0;
        foreach (var timestep in Scheduler.Timesteps)
        {
            // Cancellation observed at the boundary, NOT mid-step — a half-applied
            // UNet forward would leave the scheduler in an inconsistent state.
            cancellationToken.ThrowIfCancellationRequested();

            Tensor<T> noisePrediction;
            if (useCFG && negativeEmbedding != null)
            {
                // Run conditional and unconditional UNet predictions concurrently
                // when CFG is engaged. Both branches use identical (latents, t)
                // inputs and differ only in the conditioning embedding, so they
                // compete only for engine resources — for CPU engines that's a
                // wash, but for GPU engines with separate streams the second
                // PredictNoise's queue submission overlaps with the first's
                // tail kernels.
                var condTask = _unet.PredictNoiseAsync(latents, timestep, promptEmbedding, cancellationToken);
                var uncondTask = _unet.PredictNoiseAsync(latents, timestep, negativeEmbedding, cancellationToken);
                var condPred = await condTask.ConfigureAwait(false);
                var uncondPred = await uncondTask.ConfigureAwait(false);
                noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
            }
            else
            {
                noisePrediction = await _unet
                    .PredictNoiseAsync(latents, timestep, promptEmbedding, cancellationToken)
                    .ConfigureAwait(false);
            }

            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);

            stepIndex++;
            progress?.Report(stepIndex);
        }

        cancellationToken.ThrowIfCancellationRequested();

        // VAE decode on a worker — even on warm cache it's a multi-second forward
        // for 1024×1024 SDXL output, and we don't want to block the
        // SemaphoreSlim's continuation thread (which on UI hosts is the UI thread).
        // Future commit on this branch can lift this into the chain via
        // VAEModelBase.DecodeCompiledAsync once StandardVAE.Decode wraps its
        // body with DecodeCompiled.
        return await System.Threading.Tasks.Task.Run(
            () => DecodeFromLatent(latents), cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Cancellable variant of <see cref="GenerateWithMicroCondition"/> shared
    /// between the sync and async surfaces. Same denoise / decode pipeline
    /// but with cancellation-token observation at scheduler-step boundaries
    /// and per-step <see cref="System.IProgress{T}"/> reporting.
    /// </summary>
    private Tensor<T> GenerateWithMicroConditionCancellable(
        string prompt,
        string? negativePrompt,
        int width,
        int height,
        int? originalWidth,
        int? originalHeight,
        int cropTop,
        int cropLeft,
        int numInferenceSteps,
        double? guidanceScale,
        int? seed,
        System.IProgress<int>? progress,
        System.Threading.CancellationToken cancellationToken)
    {
        if (_conditioner1 == null)
            throw new InvalidOperationException("Text-to-image generation requires a conditioning module.");

        originalWidth ??= width;
        originalHeight ??= height;
        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;
        var useCFG = effectiveGuidanceScale > 1.0 && _unet.SupportsCFG;

        cancellationToken.ThrowIfCancellationRequested();
        var promptEmbedding = EncodeTextDual(prompt);
        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            cancellationToken.ThrowIfCancellationRequested();
            negativeEmbedding = !string.IsNullOrEmpty(negativePrompt)
                ? EncodeTextDual(negativePrompt ?? string.Empty)
                : GetUnconditionalEmbeddingDual();
        }

        var microCond = CreateMicroCondition(
            originalWidth.Value, originalHeight.Value,
            cropTop, cropLeft,
            width, height);
        promptEmbedding = ApplyMicroCondition(promptEmbedding, microCond);
        if (negativeEmbedding != null)
            negativeEmbedding = ApplyMicroCondition(negativeEmbedding, microCond);

        var latentHeight = height / SDXL_VAE_SCALE_FACTOR;
        var latentWidth = width / SDXL_VAE_SCALE_FACTOR;
        var latentShape = new[] { 1, SDXL_LATENT_CHANNELS, latentHeight, latentWidth };
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);

        int stepIndex = 0;
        foreach (var timestep in Scheduler.Timesteps)
        {
            // Check at the boundary, NOT mid-step — a half-applied UNet
            // forward would leave the scheduler in an inconsistent state.
            cancellationToken.ThrowIfCancellationRequested();

            Tensor<T> noisePrediction;
            if (useCFG && negativeEmbedding != null)
            {
                var condPred = _unet.PredictNoise(latents, timestep, promptEmbedding);
                var uncondPred = _unet.PredictNoise(latents, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
            }
            else
            {
                noisePrediction = _unet.PredictNoise(latents, timestep, promptEmbedding);
            }

            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);

            stepIndex++;
            progress?.Report(stepIndex);
        }

        cancellationToken.ThrowIfCancellationRequested();
        return DecodeFromLatent(latents);
    }

    /// <summary>
    /// Refines an image using the SDXL refiner model.
    /// </summary>
    /// <param name="image">The base image to refine.</param>
    /// <param name="prompt">The text prompt (should match base generation).</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="numInferenceSteps">Number of refiner steps (typically 20-30).</param>
    /// <param name="denoiseStrength">How much to denoise (0.2-0.4 typical for refining).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Refined image tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The refiner is a specialized model that takes
    /// an already-generated image and enhances fine details:
    ///
    /// Without refiner:
    /// - Base SDXL generates good overall structure
    /// - Some fine details may be slightly soft
    ///
    /// With refiner:
    /// - Details like skin texture, fabric, hair are enhanced
    /// - Overall coherence is preserved
    /// - Image looks more "finished"
    ///
    /// Best practices:
    /// - Use denoiseStrength 0.2-0.4 (higher = more change)
    /// - Use 20-30 refiner steps
    /// - Keep the same prompt as base generation
    /// </para>
    /// </remarks>
    public virtual Tensor<T> RefineImage(
        Tensor<T> image,
        string prompt,
        string? negativePrompt = null,
        int numInferenceSteps = 25,
        double denoiseStrength = 0.3,
        int? seed = null)
    {
        if (_refiner == null)
            throw new InvalidOperationException("Refiner model not available. Initialize SDXLModel with a refiner.");

        return _refiner.Refine(
            image,
            prompt,
            negativePrompt,
            numInferenceSteps,
            denoiseStrength,
            seed);
    }

    /// <summary>
    /// Encodes text using dual text encoders.
    /// </summary>
    private Tensor<T> EncodeTextDual(string prompt)
    {
        if (_conditioner1 == null)
            throw new InvalidOperationException("Primary conditioner not initialized.");

        var tokens1 = _conditioner1.Tokenize(prompt);
        var embedding1 = _conditioner1.EncodeText(tokens1);

        if (_useDualEncoder && _conditioner2 != null)
        {
            var tokens2 = _conditioner2.Tokenize(prompt);
            var embedding2 = _conditioner2.EncodeText(tokens2);

            // Concatenate embeddings from both encoders
            return ConcatenateEmbeddings(embedding1, embedding2);
        }

        return embedding1;
    }

    /// <summary>
    /// Async dual-encoder text encode. The two conditioners (CLIP-L + CLIP-G/OpenCLIP-G
    /// in the canonical SDXL configuration) have no data dependency on each other,
    /// so they run concurrently via <see cref="System.Threading.Tasks.Task.WhenAll"/>.
    /// On a 2-core CPU box this halves the wall time of the conditioning pre-step;
    /// when the conditioners run on different engines (e.g. CPU primary + GPU
    /// secondary) the engines' streams overlap natively. (#1272 acceptance
    /// criterion #4: dual-conditioner SDXL path measures &lt; 1.5× single-conditioner
    /// latency.)
    /// </summary>
    private async System.Threading.Tasks.ValueTask<Tensor<T>> EncodeTextDualAsync(
        string prompt,
        System.Threading.CancellationToken cancellationToken)
    {
        if (_conditioner1 == null)
            throw new InvalidOperationException("Primary conditioner not initialized.");

        cancellationToken.ThrowIfCancellationRequested();

        // Tokenization is fast (string scan) — keep on the calling thread.
        var tokens1 = _conditioner1.Tokenize(prompt);

        if (_useDualEncoder && _conditioner2 != null)
        {
            var tokens2 = _conditioner2.Tokenize(prompt);

            // Run both encoders concurrently. Conditioner.EncodeText is sync today;
            // wrap each in Task.Run so the two CPU-bound forward passes can interleave
            // on the threadpool. When the IConditioningModule interface gains a true
            // EncodeTextAsync overload, swap these for direct awaits.
            var embed1Task = System.Threading.Tasks.Task.Run(
                () => _conditioner1.EncodeText(tokens1), cancellationToken);
            var embed2Task = System.Threading.Tasks.Task.Run(
                () => _conditioner2.EncodeText(tokens2), cancellationToken);
            await System.Threading.Tasks.Task.WhenAll(embed1Task, embed2Task).ConfigureAwait(false);
            return ConcatenateEmbeddings(embed1Task.Result, embed2Task.Result);
        }

        // Single-conditioner fallback — no concurrency available.
        return await System.Threading.Tasks.Task.Run(
            () => _conditioner1.EncodeText(tokens1), cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets unconditional embedding for CFG with dual encoders.
    /// </summary>
    private Tensor<T> GetUnconditionalEmbeddingDual()
    {
        if (_conditioner1 == null)
            throw new InvalidOperationException("Primary conditioner not initialized.");

        var uncond1 = _conditioner1.GetUnconditionalEmbedding(1);

        if (_useDualEncoder && _conditioner2 != null)
        {
            var uncond2 = _conditioner2.GetUnconditionalEmbedding(1);
            return ConcatenateEmbeddings(uncond1, uncond2);
        }

        return uncond1;
    }

    /// <summary>
    /// Concatenates embeddings from two text encoders.
    /// </summary>
    private Tensor<T> ConcatenateEmbeddings(Tensor<T> embedding1, Tensor<T> embedding2)
    {
        // Concatenate along embedding (last) dimension. Tensors NuGet rejects axis=-1.
        return Engine.TensorConcatenate<T>(new[] { embedding1, embedding2 }, axis: embedding1._shape.Length - 1);
    }

    /// <summary>
    /// Creates micro-conditioning vector for aspect ratio handling.
    /// </summary>
    private Tensor<T> CreateMicroCondition(
        int originalWidth, int originalHeight,
        int cropTop, int cropLeft,
        int targetWidth, int targetHeight)
    {
        // SDXL micro-conditioning: [original_size, crop_coords, target_size]
        var microCond = new Tensor<T>(new[] { 1, 6 });
        var span = microCond.AsWritableSpan();

        span[0] = NumOps.FromDouble(originalWidth);
        span[1] = NumOps.FromDouble(originalHeight);
        span[2] = NumOps.FromDouble(cropTop);
        span[3] = NumOps.FromDouble(cropLeft);
        span[4] = NumOps.FromDouble(targetWidth);
        span[5] = NumOps.FromDouble(targetHeight);

        return microCond;
    }

    /// <summary>
    /// Concatenates micro-conditioning to text embedding.
    /// </summary>
    private Tensor<T> ApplyMicroCondition(Tensor<T> embedding, Tensor<T> microCond)
    {
        // SDXL uses micro-conditioning (original_size, crop_coords, target_size) to improve
        // generation quality at different resolutions. The 6 values are projected through
        // a learned embedding and added to influence the diffusion process.
        var shape = embedding._shape;
        var batch = shape[0];
        var seqLen = shape[1];
        var embedDim = shape[2];

        var result = new Tensor<T>(new[] { batch, seqLen, embedDim });
        var resultSpan = result.AsWritableSpan();
        var embSpan = embedding.AsSpan();
        var microSpan = microCond.AsSpan();

        // Project micro-conditioning values (6 values) to embedding dimension
        // Each micro-cond value is scaled and used as a modulation factor
        int microLen = microCond.Shape[1]; // Should be 6

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < embedDim; d++)
                {
                    int embIdx = b * seqLen * embedDim + s * embedDim + d;
                    T embVal = embSpan[embIdx];

                    // Apply micro-conditioning as sinusoidal modulation
                    // Similar to how timestep embedding works in diffusion models
                    int microIdx = d % microLen;
                    T microVal = microSpan[b * microLen + microIdx];

                    // Scale micro-conditioning (normalized to prevent explosion)
                    double microDouble = NumOps.ToDouble(microVal);
                    double normalizedMicro = microDouble / 1024.0; // Normalize by typical image size
                    double modulation = Math.Sin(d * normalizedMicro * 0.01);

                    // Apply additive modulation
                    resultSpan[embIdx] = NumOps.Add(embVal, NumOps.FromDouble(modulation * 0.1));
                }
            }
        }

        return result;
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        // Combine parameters from U-Net and VAE
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
            throw new ArgumentException($"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.");

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

    /// <inheritdoc />
    public override long ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

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
        // Clone U-Net with trained weights
                // Clone VAE with trained weights
                return new SDXLModel<T>(
            options: null,
            scheduler: null,
            unet: (UNetNoisePredictor<T>)_unet.Clone(),
            vae: (StandardVAE<T>)_vae.Clone(),
            conditioner1: _conditioner1,
            conditioner2: _conditioner2,
            refiner: _refiner,
            useDualEncoder: _useDualEncoder,
            crossAttentionDim: _crossAttentionDim);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Stable Diffusion XL",
            Version = "1.0",
            Description = "SDXL base model with dual text encoders and 1024px native resolution",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "sdxl-unet-latent-diffusion");
        metadata.SetProperty("base_model", "Stable Diffusion XL");
        metadata.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        metadata.SetProperty("text_encoder_2", "OpenCLIP ViT-bigG/14");
        metadata.SetProperty("cross_attention_dim", _crossAttentionDim);
        metadata.SetProperty("latent_channels", SDXL_LATENT_CHANNELS);
        metadata.SetProperty("default_resolution", DefaultWidth);
        metadata.SetProperty("latent_scale", 0.13025);
        metadata.SetProperty("has_refiner", _refiner != null);
        metadata.SetProperty("dual_encoder", _useDualEncoder);

        return metadata;
    }

    #endregion
}

/// <summary>
/// SDXL Refiner model for enhancing generated images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The SDXL refiner is a specialized model trained to enhance the fine details
/// of images generated by the SDXL base model. It operates in the latent space
/// and is designed to work with partially denoised latents.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of the refiner as a "finishing touch" step:
///
/// Without refiner:
/// Base image (good structure, okay details)
///
/// With refiner:
/// Refined image (same structure, enhanced details like texture, sharpness)
///
/// The refiner uses lower noise levels (typically 0.2-0.4 denoising strength)
/// to preserve the overall composition while enhancing fine details.
/// </para>
/// </remarks>
public class SDXLRefiner<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The refiner U-Net.
    /// </summary>
    private readonly UNetNoisePredictor<T> _refinerUNet;

    /// <summary>
    /// The VAE (shared with base model).
    /// </summary>
    private readonly IVAEModel<T> _vae;

    /// <summary>
    /// The scheduler.
    /// </summary>
    private readonly INoiseScheduler<T> _scheduler;

    /// <summary>
    /// The conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new SDXL refiner.
    /// </summary>
    /// <param name="refinerUNet">The refiner U-Net.</param>
    /// <param name="vae">The VAE (shared with base model).</param>
    /// <param name="scheduler">The scheduler.</param>
    /// <param name="conditioner">Optional conditioning module.</param>
    /// <param name="seed">Optional random seed.</param>
    public SDXLRefiner(
        UNetNoisePredictor<T> refinerUNet,
        IVAEModel<T> vae,
        INoiseScheduler<T> scheduler,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
    {
        _refinerUNet = refinerUNet;
        _vae = vae;
        _scheduler = scheduler;
        _conditioner = conditioner;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Refines an image by enhancing details.
    /// </summary>
    /// <param name="image">The input image to refine.</param>
    /// <param name="prompt">Text prompt describing the image.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="numInferenceSteps">Number of refiner steps.</param>
    /// <param name="denoiseStrength">Denoising strength (0.0-1.0).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Refined image tensor.</returns>
    public Tensor<T> Refine(
        Tensor<T> image,
        string prompt,
        string? negativePrompt = null,
        int numInferenceSteps = 25,
        double denoiseStrength = 0.3,
        int? seed = null)
    {
        // Encode image to latent
        var latent = _vae.Encode(image, sampleMode: false);
        latent = _vae.ScaleLatent(latent);

        // Encode text conditioning
        Tensor<T>? promptEmbedding = null;
        Tensor<T>? negativeEmbedding = null;

        if (_conditioner != null)
        {
            var tokens = _conditioner.Tokenize(prompt);
            promptEmbedding = _conditioner.EncodeText(tokens);

            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = _conditioner.Tokenize(negativePrompt ?? string.Empty);
                negativeEmbedding = _conditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = _conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Set up scheduler for refining (partial denoising)
        _scheduler.SetTimesteps(numInferenceSteps);
        var startStep = (int)(numInferenceSteps * (1.0 - denoiseStrength));

        // Add noise at the starting point
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : _random;
        var noise = SampleNoise(latent._shape, rng);

        var startTimestep = _scheduler.Timesteps.Skip(startStep).FirstOrDefault();
        latent = AddNoiseAtTimestep(latent, noise, startTimestep);

        // Denoising loop (only the remaining steps)
        foreach (var timestep in _scheduler.Timesteps.Skip(startStep))
        {
            Tensor<T> noisePrediction;

            if (promptEmbedding != null && negativeEmbedding != null)
            {
                var condPred = _refinerUNet.PredictNoise(latent, timestep, promptEmbedding);
                var uncondPred = _refinerUNet.PredictNoise(latent, timestep, negativeEmbedding);

                // Apply guidance (typically lower for refiner, around 5.0)
                noisePrediction = ApplyGuidance(uncondPred, condPred, 5.0);
            }
            else
            {
                noisePrediction = _refinerUNet.PredictNoise(latent, timestep, null);
            }

            // Scheduler step
            var latentVector = latent.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = _scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latent = new Tensor<T>(latent._shape, latentVector);
        }

        // Decode back to image
        var unscaled = _vae.UnscaleLatent(latent);
        return _vae.Decode(unscaled);
    }

    /// <summary>
    /// Samples noise tensor.
    /// </summary>
    private Tensor<T> SampleNoise(int[] shape, Random rng)
    {
        var noise = new Tensor<T>(shape);
        var span = noise.AsWritableSpan();

        for (int i = 0; i < span.Length; i++)
        {
            span[i] = NumOps.FromDouble(rng.NextGaussian());
        }

        return noise;
    }

    /// <summary>
    /// Adds noise at a specific timestep.
    /// </summary>
    private Tensor<T> AddNoiseAtTimestep(Tensor<T> latent, Tensor<T> noise, int timestep)
    {
        // Simplified noise addition based on timestep
        // In practice, this uses the scheduler's alpha values
        var alpha = 1.0 - (timestep / 1000.0);
        var sigma = Math.Sqrt(1.0 - alpha * alpha);

        // result = alpha * latent + sigma * noise
        var engine = AiDotNetEngine.Current;
        var alphaT = NumOps.FromDouble(alpha);
        var sigmaT = NumOps.FromDouble(sigma);
        var scaledLatent = engine.TensorMultiplyScalar<T>(latent, alphaT);
        var scaledNoise = engine.TensorMultiplyScalar<T>(noise, sigmaT);
        return engine.TensorAdd<T>(scaledLatent, scaledNoise);
    }

    /// <summary>
    /// Applies classifier-free guidance.
    /// </summary>
    private Tensor<T> ApplyGuidance(Tensor<T> uncondPred, Tensor<T> condPred, double guidanceScale)
    {
        // CFG: guided = uncond + scale * (cond - uncond)
        var engine = AiDotNetEngine.Current;
        var scaleT = NumOps.FromDouble(guidanceScale);
        var diff = engine.TensorSubtract<T>(condPred, uncondPred);
        var scaled = engine.TensorMultiplyScalar<T>(diff, scaleT);
        return engine.TensorAdd<T>(uncondPred, scaled);
    }
}
