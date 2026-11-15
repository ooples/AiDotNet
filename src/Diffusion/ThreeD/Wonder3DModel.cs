using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.Attention;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.ThreeD.Meshing;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralRadianceFields.NeuS;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// Wonder3D (Long et al. 2023, "Wonder3D: Single Image to 3D using Cross-Domain Diffusion", arXiv 2310.15008): a
/// cross-domain multi-view diffusion that generates six orthographic normal maps and color images of an object from
/// one image, and a geometry-aware normal fusion that reconstructs its textured mesh from them.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>Cross-domain diffusion.</b> One Stable Diffusion image-variations U-Net produces all twelve images of an object
/// at once, the six views (front, front-right, right, back, left, front-left) in the normal domain and the same six
/// in the color domain. A view's domain and camera reach the U-Net through the domain switcher, a class embedding
/// added to the timestep embedding (<see cref="Wonder3DDomain"/>). Every transformer block runs multi-view
/// self-attention, where each view reads the keys and values of all six views of its domain, and cross-domain
/// attention, where each view's normal and color tokens attend to one another; both are built from the U-Net's own
/// attention blocks. The input conditions the U-Net as in SD image variations: its CLIP image embedding is the
/// cross-attention context and its VAE latent is concatenated to every view's noisy latent (8 input channels).
/// Sampling is DDIM, 50 steps, eta 1, classifier-free guidance 3 against a zero embedding and zero latent.
/// </para>
/// <para>
/// <b>Geometry-aware normal fusion.</b> The views are orthographic, so the reconstruction fits NeuS to them with
/// orthographic rays. The generated normal maps (in the front camera's frame) are rotated to world space and
/// supervise the rendered normals with Wonder3D's geometry-aware, outlier-dropping normal loss, alongside its color,
/// mask, Eikonal, sparsity and smoothness terms; the zero level set is meshed with marching cubes and colored by the
/// NeuS color network.
/// </para>
/// <para>
/// <b>Weights.</b> The architecture is the released one, but meaningful views need the released weights
/// (<c>flamehaze1115/wonder3d-v1.0</c>), loaded through the model's parameter vector, and the CLIP ViT-L/14 image
/// encoder of SD image variations, supplied as an <see cref="IVisualEncoder{T}"/> whose
/// <see cref="IVisualEncoder{T}.EncodeImage"/> returns the projected image embedding (<c>image_embeds</c>). The
/// reconstruction needs no weights: it is optimized per object.
/// </para>
/// <para>
/// <b>For Beginners:</b> Give Wonder3D a photo of an object on a white background. It imagines what the object
/// looks like from six directions, both its colors and the direction its surface faces at every pixel, then carves
/// a 3D shape that matches all of them.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var model = new Wonder3DModel&lt;float&gt;(imageEncoder: clipViTL14);
/// var mesh = model.ImageTo3D(image); // image: [1, 3, 256, 256], RGB in [0, 1]
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ThreeDGeneration)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Wonder3D: Single Image to 3D using Cross-Domain Diffusion", "https://arxiv.org/abs/2310.15008", Year = 2024, Authors = "Long et al.")]
public partial class Wonder3DModel<T> : ThreeDDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Registration order is serialization order: the U-Net (with its multi-view and cross-domain
    /// attention), the VAE, then the domain switcher's two linear layers.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_unet);
        RegisterParameterComponent(_vae);
        RegisterParameterComponent(_domainLinear1);
        RegisterParameterComponent(_domainLinear2);
    }

    #region Fields

    private const int DefaultPointBudget = 4096;

    private readonly Wonder3DOptions _options;

    private readonly UNetNoisePredictor<T> _unet;

    private readonly StandardVAE<T> _vae;

    /// <summary>The domain switcher's first layer, <c>Linear(10, d)</c> then SiLU (<c>class_embedding.linear_1</c>).</summary>
    private readonly DenseLayer<T> _domainLinear1;

    /// <summary>The domain switcher's second layer, <c>Linear(d, d)</c> (<c>class_embedding.linear_2</c>).</summary>
    private readonly DenseLayer<T> _domainLinear2;

    private readonly IVisualEncoder<T>? _imageEncoder;

    #endregion

    #region Constructor

    /// <summary>Creates Wonder3D.</summary>
    /// <param name="options">The configuration, or null for the released model's.</param>
    /// <param name="architecture">An optional architecture for the U-Net.</param>
    /// <param name="scheduler">The sampler, or null for DDIM with Stable Diffusion's scaled-linear schedule.</param>
    /// <param name="unet">A U-Net to use, or null to build the configured one. It must take twice the latent channels
    /// (the noisy latent and the conditioning latent) and produce the latent channels.</param>
    /// <param name="vae">A VAE to use, or null to build the configured one.</param>
    /// <param name="imageEncoder">The CLIP image encoder whose projected embedding conditions the U-Net (CLIP ViT-L/14
    /// for the released weights); needed by the image-conditioned methods when the U-Net cross-attends.</param>
    public Wonder3DModel(
        Wonder3DOptions? options = null,
        NeuralNetworkArchitecture<T>? architecture = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IVisualEncoder<T>? imageEncoder = null)
        : base(
            new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear, Seed = options?.Seed
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            DefaultPointBudget, architecture)
    {
        _options = options ?? new Wonder3DOptions();
        Validate(_options);
        var o = _options;
        int? seed = o.Seed;

        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: o.LatentChannels,
            baseChannels: o.VaeBaseChannels, channelMultipliers: (int[])o.VaeChannelMultipliers.Clone(),
            numResBlocksPerLevel: o.VaeResBlocksPerLevel, latentScaleFactor: o.VaeScaleFactor, seed: seed);
        if (_vae.LatentChannels != o.LatentChannels)
            throw new ArgumentException($"The VAE produces {_vae.LatentChannels} latent channels; the options say {o.LatentChannels}.",
                nameof(vae));

        _unet = unet ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: 2 * o.LatentChannels, outputChannels: o.LatentChannels,
            baseChannels: o.BaseChannels, channelMultipliers: (int[])o.ChannelMultipliers.Clone(),
            numResBlocks: o.ResBlocksPerLevel, attentionResolutions: (int[])o.AttentionLevels.Clone(),
            contextDim: o.CrossAttentionDim, numHeads: o.AttentionHeads, inputHeight: LatentSize(o), seed: seed);
        if (_unet.InputChannels != 2 * o.LatentChannels || _unet.OutputChannels != o.LatentChannels)
            throw new ArgumentException(
                $"Wonder3D's U-Net takes {2 * o.LatentChannels} channels (noisy and conditioning latents) and predicts " +
                $"{o.LatentChannels}; this one maps {_unet.InputChannels} to {_unet.OutputChannels}.", nameof(unet));

        int timeEmbeddingDim = _unet.TimeEmbeddingDim;
        _domainLinear1 = new DenseLayer<T>(timeEmbeddingDim, (IActivationFunction<T>)new SiLUActivation<T>(),
            InitializationStrategies<T>.Lazy);
        _domainLinear1.ResolveShapesOnly(new[] { Wonder3DDomainEmbedding.InputDim });
        _domainLinear2 = new DenseLayer<T>(timeEmbeddingDim, (IActivationFunction<T>?)null, InitializationStrategies<T>.Lazy);
        _domainLinear2.ResolveShapesOnly(new[] { timeEmbeddingDim });
        if (seed.HasValue)
        {
            _domainLinear1.RandomSeed = unchecked(seed.Value * 31 + 1);
            _domainLinear2.RandomSeed = unchecked(seed.Value * 31 + 2);
        }

        _imageEncoder = imageEncoder;
        AttentionBlockCount = DecorateAttention(seed);
    }

    private static int LatentSize(Wonder3DOptions o) => o.ImageSize >> (o.VaeChannelMultipliers.Length - 1);

    private static void Validate(Wonder3DOptions o)
    {
        void Positive(int value, string name)
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(name, $"{name} must be positive.");
        }

        void Fraction(double value, string name)
        {
            if (!(value > 0 && value <= 1)) throw new ArgumentOutOfRangeException(name, $"{name} must be in (0, 1].");
        }

        void NonNegative(double value, string name)
        {
            if (!(value >= 0) || double.IsInfinity(value))
                throw new ArgumentOutOfRangeException(name, $"{name} must be non-negative and finite.");
        }

        Positive(o.NumViews, nameof(o.NumViews));
        Positive(o.ImageSize, nameof(o.ImageSize));
        Positive(o.LatentChannels, nameof(o.LatentChannels));
        Positive(o.BaseChannels, nameof(o.BaseChannels));
        Positive(o.ResBlocksPerLevel, nameof(o.ResBlocksPerLevel));
        Positive(o.AttentionHeads, nameof(o.AttentionHeads));
        Positive(o.VaeBaseChannels, nameof(o.VaeBaseChannels));
        Positive(o.VaeResBlocksPerLevel, nameof(o.VaeResBlocksPerLevel));
        Positive(o.InferenceSteps, nameof(o.InferenceSteps));
        Positive(o.ReconstructionIterations, nameof(o.ReconstructionIterations));
        Positive(o.ReconstructionBatchSize, nameof(o.ReconstructionBatchSize));
        if (o.CrossAttentionDim < 0) throw new ArgumentOutOfRangeException(nameof(o.CrossAttentionDim));
        if (o.ChannelMultipliers is null || o.ChannelMultipliers.Length == 0)
            throw new ArgumentException("ChannelMultipliers needs at least one level.", nameof(o));
        if (o.VaeChannelMultipliers is null || o.VaeChannelMultipliers.Length == 0)
            throw new ArgumentException("VaeChannelMultipliers needs at least one level.", nameof(o));
        if (o.AttentionLevels is null) throw new ArgumentException("AttentionLevels cannot be null.", nameof(o));
        if (o.SdfSkipLayers is null) throw new ArgumentException("SdfSkipLayers cannot be null.", nameof(o));
        int factor = 1 << (o.VaeChannelMultipliers.Length - 1);
        if (o.ImageSize % factor != 0)
            throw new ArgumentException($"ImageSize {o.ImageSize} must be a multiple of the VAE's downscale {factor}.", nameof(o));
        if (o.ViewAzimuthsDegrees is null || o.ViewAzimuthsDegrees.Length != o.NumViews)
            throw new ArgumentException("ViewAzimuthsDegrees needs one azimuth per view.", nameof(o));
        if (o.ViewCameraEmbedding is null || o.ViewCameraEmbedding.Length != 3 * o.NumViews)
            throw new ArgumentException("ViewCameraEmbedding needs three values per view.", nameof(o));
        if (o.ViewWeights is null || o.ViewWeights.Length != o.NumViews)
            throw new ArgumentException("ViewWeights needs one weight per view.", nameof(o));
        foreach (double weight in o.ViewWeights) NonNegative(weight, nameof(o.ViewWeights));
        if (!(o.GuidanceScale > 0) || double.IsInfinity(o.GuidanceScale))
            throw new ArgumentOutOfRangeException(nameof(o.GuidanceScale), "The guidance scale must be positive and finite.");
        NonNegative(o.Eta, nameof(o.Eta));
        if (!(o.OrthographicHalfExtent > 0)) throw new ArgumentOutOfRangeException(nameof(o.OrthographicHalfExtent));
        if (!(o.CameraDistance > 0)) throw new ArgumentOutOfRangeException(nameof(o.CameraDistance));
        if (!(o.BackgroundThreshold > 0 && o.BackgroundThreshold <= 1))
            throw new ArgumentOutOfRangeException(nameof(o.BackgroundThreshold));
        Fraction(o.ColorKeepRatio, nameof(o.ColorKeepRatio));
        Fraction(o.NormalKeepRatio, nameof(o.NormalKeepRatio));
        Fraction(o.MaskKeepRatio, nameof(o.MaskKeepRatio));
        if (!(o.MeshResolution >= 2)) throw new ArgumentOutOfRangeException(nameof(o.MeshResolution), "MeshResolution must be at least 2.");
    }

    // Wraps every self-attention block of the U-Net with Wonder3D's multi-view and cross-domain attention. Adopts a
    // block that is already wrapped: a clone hands the constructor an already-decorated U-Net, and wrapping again
    // would stack a second cross-domain attention on the first.
    private int DecorateAttention(int? seed)
    {
        int created = 0;
        int numViews = _options.NumViews;
        int heads = _options.AttentionHeads;
        return _unet.DecorateAttentionBlocks((block, channels, isCrossAttention) =>
        {
            if (isCrossAttention) return block;
            if (block is Wonder3DAttentionBlock<T> adopted)
            {
                if (adopted.NumViews != numViews)
                    throw new ArgumentException(
                        $"The U-Net's Wonder3D attention was built for {adopted.NumViews} views; the options say {numViews}.");
                return block;
            }

            if (block is not DiffusionAttention<T> attention)
                throw new InvalidOperationException(
                    $"Wonder3D's multi-view attention reuses the U-Net's DiffusionAttention blocks; got {block.GetType().Name}.");
            int? blockSeed = seed.HasValue ? unchecked(seed.Value * 7919 + 104729 + created) : null;
            created++;
            return new Wonder3DAttentionBlock<T>(attention, channels, numViews, heads, blockSeed);
        });
    }

    #endregion

    #region Properties

    /// <summary>The configuration.</summary>
    public Wonder3DOptions Configuration => _options;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    /// <remarks>Wonder3D has no text conditioning: its condition is the input image (see the constructor's image
    /// encoder).</remarks>
    public override IConditioningModule<T>? Conditioner => null;

    /// <inheritdoc />
    public override int LatentChannels => _options.LatentChannels;

    /// <inheritdoc />
    public override bool SupportsPointCloud => true;

    /// <inheritdoc />
    public override bool SupportsMesh => true;

    /// <inheritdoc />
    public override bool SupportsTexture => true;

    /// <inheritdoc />
    public override bool SupportsNovelView => true;

    /// <inheritdoc />
    public override bool SupportsScoreDistillation => false;

    /// <summary>Gets the number of views generated in each domain.</summary>
    public int NumViews => _options.NumViews;

    /// <summary>The number of U-Net attention blocks carrying multi-view and cross-domain attention.</summary>
    internal int AttentionBlockCount { get; }

    /// <summary>The domain switcher's first layer (<c>Linear(10, d)</c> then SiLU).</summary>
    internal DenseLayer<T> DomainLinear1 => _domainLinear1;

    /// <summary>The domain switcher's second layer (<c>Linear(d, d)</c>).</summary>
    internal DenseLayer<T> DomainLinear2 => _domainLinear2;

    #endregion

    #region Cross-domain diffusion

    /// <summary>
    /// The domain-switcher embeddings of <paramref name="objects"/> objects' joint batch, <c>[2·objects·V, d]</c>:
    /// the normal views of every object, then their color views.
    /// </summary>
    internal Tensor<T> ClassEmbeddings(int objects)
    {
        int views = _options.NumViews;
        int count = 2 * objects * views;
        var rows = new double[count * Wonder3DDomainEmbedding.InputDim];
        var table = _options.ViewCameraEmbedding;
        for (int i = 0; i < count; i++)
        {
            int view = i % views;
            var domain = i < count / 2 ? Wonder3DDomain.Normal : Wonder3DDomain.Color;
            var encoded = Wonder3DDomainEmbedding.Encode(table[3 * view], table[3 * view + 1], table[3 * view + 2], domain);
            Array.Copy(encoded, 0, rows, i * encoded.Length, encoded.Length);
        }

        return EmbedDomains(NeuSHost<T>.FromDoubles(rows, count, Wonder3DDomainEmbedding.InputDim));
    }

    /// <summary>The domain switcher on encoded rows <c>[N, 10]</c>: <c>Linear → SiLU → Linear</c>, <c>[N, d]</c>.</summary>
    internal Tensor<T> EmbedDomains(Tensor<T> encoded) => _domainLinear2.Forward(_domainLinear1.Forward(encoded));

    /// <summary>
    /// One U-Net evaluation on a joint batch: the noisy latents concatenated with the conditioning latents along the
    /// channels, the image embedding as cross-attention context, and the domain-switcher embedding added to the time
    /// embedding.
    /// </summary>
    internal Tensor<T> PredictJointNoise(Tensor<T> latents, int timestep, Tensor<T> conditionLatents,
        Tensor<T>? imageEmbedding, Tensor<T> classEmbedding)
    {
        var input = Engine.TensorConcatenate(new[] { latents, conditionLatents }, 1);
        return _unet.PredictNoiseWithAddedTimeEmbedding(input, timestep, imageEmbedding, classEmbedding);
    }

    /// <summary>
    /// Wonder3D's unconditional noise prediction for a joint batch <c>[2·k·V, C, h, w]</c> (the normal views of k
    /// objects, then their color views): a zero conditioning latent and a zero image embedding, the classifier-free
    /// guidance branch the released pipeline runs.
    /// </summary>
    /// <param name="noisySample">The joint batch of noisy latents.</param>
    /// <param name="timestep">The timestep.</param>
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep)
    {
        if (noisySample is null) throw new ArgumentNullException(nameof(noisySample));
        if (noisySample.Shape.Length != 4 || noisySample.Shape[1] != _options.LatentChannels)
            throw new ArgumentException(
                $"Wonder3D predicts noise for [2·k·{NumViews}, {_options.LatentChannels}, h, w] joint batches.", nameof(noisySample));
        int count = noisySample.Shape[0];
        int group = 2 * NumViews;
        if (count % group != 0)
            throw new ArgumentException(
                $"A Wonder3D batch holds every object's {NumViews} views in both domains, a multiple of {group}; got {count}.",
                nameof(noisySample));
        var zeros = new Tensor<T>(noisySample._shape);
        return PredictJointNoise(noisySample, timestep, zeros, UnconditionalContext(count), ClassEmbeddings(count / group));
    }

    /// <summary>
    /// A joint batch is one object's views noised at ONE timestep, as Wonder3D trains (its views share the
    /// object's timestep), so the base trains it as a single sample.
    /// </summary>
    protected override bool IsBatchedTrainingSample(Tensor<T> sample) => false;

    private Tensor<T>? UnconditionalContext(int count)
        => _unet.ContextDimension > 0 ? new Tensor<T>(new[] { count, 1, _unet.ContextDimension }) : null;

    /// <summary>
    /// Generates the six views of the object in <paramref name="image"/> in both domains.
    /// </summary>
    /// <param name="image">The input, <c>[1, 3, H, W]</c> or <c>[3, H, W]</c>, RGB in [0, 1] on a white background;
    /// resized to <see cref="Wonder3DOptions.ImageSize"/> when it differs.</param>
    /// <param name="numInferenceSteps">DDIM steps, or null for the configured 50.</param>
    /// <param name="guidanceScale">The guidance scale, or null for the configured 3.</param>
    /// <param name="seed">The sampling seed, or null.</param>
    /// <returns><c>[2V, 3, S, S]</c> in [0, 1]: the normal maps of the V views (camera normals of the front view,
    /// encoded as <c>(n + 1) / 2</c>), then their color images.</returns>
    public Tensor<T> GenerateViews(Tensor<T> image, int? numInferenceSteps = null, double? guidanceScale = null, int? seed = null)
    {
        var input = PrepareImage(image);
        var o = _options;
        int count = 2 * o.NumViews;
        int steps = numInferenceSteps ?? o.InferenceSteps;
        if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(numInferenceSteps), "At least one step is needed.");
        double scale = guidanceScale ?? o.GuidanceScale;

        // SD image variations: the VAE sees [-1, 1], latent_dist.mode() scaled; CLIP sees the image itself.
        var vaeInput = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(input, NumOps.FromDouble(2.0)), NumOps.FromDouble(-1.0));
        var latent = EncodeToLatent(vaeInput, sampleMode: false);
        var conditionLatents = Repeat(latent, count);
        var context = ImageContext(input, count);
        var unconditionalLatents = new Tensor<T>(conditionLatents._shape);
        var unconditionalContext = UnconditionalContext(count);
        var classEmbedding = ClassEmbeddings(1);

        var latentShape = conditionLatents._shape;
        var rng = CreateInferenceRng(seed);
        var latents = SampleNoiseTensor(latentShape, rng);
        var eta = NumOps.FromDouble(o.Eta);
        Scheduler.SetTimesteps(steps);
        foreach (var timestep in Scheduler.Timesteps)
        {
            var conditional = PredictJointNoise(latents, timestep, conditionLatents, context, classEmbedding);
            // do_classifier_free_guidance = guidance_scale != 1.0
            var noise = scale != 1.0
                ? ApplyGuidance(PredictJointNoise(latents, timestep, unconditionalLatents, unconditionalContext, classEmbedding),
                    conditional, scale)
                : conditional;
            var stepNoise = o.Eta > 0 ? SampleNoiseTensor(latentShape, rng).ToVector() : null;
            latents = new Tensor<T>(latentShape, Scheduler.Step(noise.ToVector(), timestep, latents.ToVector(), eta, stepNoise));
        }

        var decoded = new Tensor<T>[count];
        for (int i = 0; i < count; i++)
            decoded[i] = DecodeFromLatent(Engine.TensorNarrow(latents, 0, i, 1).Contiguous());
        var images = Engine.TensorConcatenate(decoded, 0);
        // VaeImageProcessor.postprocess: (x / 2 + 0.5).clamp(0, 1).
        var unit = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(images, NumOps.FromDouble(0.5)), NumOps.FromDouble(0.5));
        return Engine.TensorClamp(unit, NumOps.Zero, NumOps.One);
    }

    private Tensor<T> PrepareImage(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var batch = image.Shape.Length == 3
            ? Engine.Reshape(image, new[] { 1, image.Shape[0], image.Shape[1], image.Shape[2] })
            : image;
        if (batch.Shape.Length != 4 || batch.Shape[0] != 1 || batch.Shape[1] != 3)
            throw new ArgumentException("Wonder3D takes one RGB image, [1, 3, H, W] or [3, H, W].", nameof(image));
        int size = _options.ImageSize;
        return batch.Shape[2] == size && batch.Shape[3] == size
            ? batch
            : Engine.Interpolate(batch, new[] { size, size }, InterpolateMode.Bilinear, false);
    }

    // The CLIP image embedding of the input as the U-Net's context, [count, 1, d]; null when the U-Net has no
    // cross-attention.
    private Tensor<T>? ImageContext(Tensor<T> image, int count)
    {
        int width = _unet.ContextDimension;
        if (width <= 0) return null;
        var encoder = _imageEncoder ?? throw new InvalidOperationException(
            "Wonder3D conditions its U-Net on the CLIP image embedding of the input (SD image variations: CLIP ViT-L/14, " +
            "image_embeds). Pass an IVisualEncoder<T> that returns it as the imageEncoder constructor argument.");
        var resized = encoder.ImageSize > 0 && (encoder.ImageSize != image.Shape[2] || encoder.ImageSize != image.Shape[3])
            ? Engine.Interpolate(image, new[] { encoder.ImageSize, encoder.ImageSize }, InterpolateMode.Bilinear, false)
            : image;
        var embedding = encoder.EncodeImage(resized);
        if (embedding.Length != width)
            throw new InvalidOperationException(
                $"The image encoder returned {embedding.Length} values; the U-Net cross-attends to one {width}-wide image embedding.");
        return Repeat(Engine.Reshape(embedding, new[] { 1, 1, width }), count);
    }

    private Tensor<T> Repeat(Tensor<T> single, int count)
    {
        var copies = new Tensor<T>[count];
        for (int i = 0; i < count; i++) copies[i] = single;
        return Engine.TensorConcatenate(copies, 0);
    }

    #endregion

    #region Reconstruction

    /// <summary>
    /// The orthographic camera of view <paramref name="view"/>: on the horizontal circle at
    /// <see cref="Wonder3DOptions.CameraDistance"/>, looking at the origin with z up, the front view at −y.
    /// </summary>
    internal NeuSCamera ViewCamera(int view, int width, int height)
    {
        double azimuth = _options.ViewAzimuthsDegrees[view] * Math.PI / 180.0;
        return OrthographicCamera(azimuth, 0.0, width, height);
    }

    private NeuSCamera OrthographicCamera(double azimuth, double elevation, int width, int height)
    {
        double d = _options.CameraDistance;
        var center = new[]
        {
            d * Math.Cos(elevation) * Math.Sin(azimuth), -d * Math.Cos(elevation) * Math.Cos(azimuth), d * Math.Sin(elevation),
        };
        return NeuSCamera.Orthographic(center, new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 0.0, 1.0 },
            _options.OrthographicHalfExtent, width, height);
    }

    /// <summary>The NeuS configuration of the geometry-aware normal fusion.</summary>
    internal NeuSOptions ReconstructionOptions(int? seed)
    {
        var o = _options;
        return new NeuSOptions
        {
            SdfHiddenDim = o.SdfHiddenDim,
            SdfHiddenLayers = o.SdfHiddenLayers,
            SdfFeatureDim = o.SdfFeatureDim,
            SdfSkipLayers = (int[])o.SdfSkipLayers.Clone(),
            SdfFrequencies = o.SdfFrequencies,
            ColorHiddenDim = o.ColorHiddenDim,
            ColorHiddenLayers = o.ColorHiddenLayers,
            ViewFrequencies = o.ViewFrequencies,
            Samples = o.RaySamples,
            ImportanceSamples = o.ImportanceSamples,
            UpSampleSteps = o.UpSampleSteps,
            // background_color: black, with the targets composited onto it (apply_mask).
            BackgroundColor = new[] { 0.0, 0.0, 0.0 },
            LearningRate = o.ReconstructionLearningRate,
            LearningRateAlpha = o.ReconstructionLearningRateAlpha,
            WarmUpIterations = 0,
            Iterations = o.ReconstructionIterations,
            BatchSize = o.ReconstructionBatchSize,
            AnnealIterations = o.ReconstructionAnnealIterations,
            ColorL1Weight = 0.0,
            ColorMseWeight = o.ColorMseWeight,
            ColorOnForegroundOnly = false,
            ColorKeepRatio = o.ColorKeepRatio,
            EikonalWeight = o.EikonalWeight,
            MaskWeight = o.MaskWeight,
            MaskKeepRatio = o.MaskKeepRatio,
            NormalWeight = o.NormalWeight,
            NormalKeepRatio = o.NormalKeepRatio,
            GeometryAwareNormals = o.GeometryAwareNormals,
            NormalVisibleCosine = o.NormalVisibleCosine,
            SparsityWeight = o.SparsityWeight,
            SparsityScale = o.SparsityScale,
            NormalSmoothWeight = o.NormalSmoothWeight,
            Seed = seed,
        };
    }

    /// <summary>
    /// The NeuS views of generated views <c>[2V, 3, H, W]</c>: each view's orthographic camera, its color composited
    /// onto black, its foreground mask (where the normal map is not white background), its world normals and its
    /// view weight.
    /// </summary>
    internal IReadOnlyList<NeuSView<T>> ReconstructionViews(Tensor<T> views)
    {
        if (views is null) throw new ArgumentNullException(nameof(views));
        int v = NumViews;
        if (views.Shape.Length != 4 || views.Shape[0] != 2 * v || views.Shape[1] != 3)
            throw new ArgumentException($"The views must be [{2 * v}, 3, H, W]: {v} normal maps, then {v} color images.", nameof(views));
        int h = views.Shape[2], w = views.Shape[3];
        var data = NeuSHost<T>.ToDoubles(views);
        // Wonder3D's normals are in the front camera's frame (normal_system 'front'), OpenGL axes.
        var front = ViewCamera(0, w, h).Rotation;
        double threshold = _options.BackgroundThreshold;
        var result = new List<NeuSView<T>>(v);
        for (int k = 0; k < v; k++)
        {
            var colors = new double[h * w * 3];
            var mask = new double[h * w];
            var normals = new double[h * w * 3];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int pixel = y * w + x;
                    double nx = data[((k * 3) * h + y) * w + x];
                    double ny = data[((k * 3 + 1) * h + y) * w + x];
                    double nz = data[((k * 3 + 2) * h + y) * w + x];
                    bool background = nx > threshold && ny > threshold && nz > threshold;
                    if (background) continue;
                    mask[pixel] = 1.0;
                    for (int c = 0; c < 3; c++)
                        colors[pixel * 3 + c] = data[(((v + k) * 3 + c) * h + y) * w + x];
                    var world = FrontNormalToWorld(front, 2.0 * nx - 1.0, 2.0 * ny - 1.0, 2.0 * nz - 1.0);
                    Array.Copy(world, 0, normals, pixel * 3, 3);
                }

            result.Add(new NeuSView<T>(
                NeuSHost<T>.FromDoubles(colors, h, w, 3), ViewCamera(k, w, h), NeuSHost<T>.FromDoubles(mask, h, w),
                NeuSHost<T>.FromDoubles(normals, h, w, 3), _options.ViewWeights[k]));
        }

        return result;
    }

    /// <summary>
    /// A front-camera normal in OpenGL axes (x right, y up, z toward the viewer), rotated to world space through
    /// the front camera's OpenCV camera-to-world rotation (<c>normal_opengl2opencv</c> then
    /// <c>camNormal2worldNormal</c>), normalized.
    /// </summary>
    internal static double[] FrontNormalToWorld(double[] frontRotation, double x, double y, double z)
    {
        double cx = x, cy = -y, cz = -z;
        var world = new double[3];
        double norm = 0;
        for (int r = 0; r < 3; r++)
        {
            world[r] = frontRotation[r * 3] * cx + frontRotation[r * 3 + 1] * cy + frontRotation[r * 3 + 2] * cz;
            norm += world[r] * world[r];
        }

        norm = Math.Sqrt(norm);
        if (norm > 0)
            for (int r = 0; r < 3; r++) world[r] /= norm;
        return world;
    }

    /// <summary>Fits NeuS to generated views with the geometry-aware normal fusion.</summary>
    internal NeuSField<T> ReconstructField(Tensor<T> views, int? seed)
    {
        var reconstructor = new NeuSReconstructor<T>(ReconstructionOptions(seed));
        return reconstructor.Fit(ReconstructionViews(views), _options.ReconstructionIterations);
    }

    /// <summary>
    /// The mesh of a field's zero level set on the unit cube, colored by its color network seen along the inward
    /// normal.
    /// </summary>
    internal Mesh3D<T> ExtractMesh(NeuSField<T> field, int resolution)
    {
        if (field is null) throw new ArgumentNullException(nameof(field));
        if (resolution < 2) throw new ArgumentOutOfRangeException(nameof(resolution), "The grid needs at least two samples per axis.");
        var grid = field.SampleGrid(resolution, SdfGridBounds.UnitCube);
        var mesh = MarchingCubes.Extract(grid, 0.0, IsoSurfaceInterior.BelowIsoLevel, -1.0, -1.0, -1.0, 2.0 / (resolution - 1));
        if (mesh.Vertices.Shape[0] > 0)
        {
            var normals = mesh.Normals ?? field.QueryNormals(mesh.Vertices);
            mesh.Colors = field.QueryColor(mesh.Vertices, Engine.TensorNegate(normals));
        }

        return mesh;
    }

    /// <summary>
    /// Reconstructs the textured mesh of generated views <c>[2V, 3, H, W]</c> (as <see cref="GenerateViews"/> returns).
    /// </summary>
    /// <param name="views">The normal maps, then the color images, in [0, 1].</param>
    /// <param name="resolution">The marching-cubes resolution, or null for the configured 192.</param>
    /// <param name="seed">The reconstruction seed, or null.</param>
    public Mesh3D<T> ReconstructMesh(Tensor<T> views, int? resolution = null, int? seed = null)
        => ExtractMesh(ReconstructField(views, seed), resolution ?? _options.MeshResolution);

    /// <summary>Generates the textured mesh of the object in <paramref name="image"/>.</summary>
    /// <param name="image">The input, <c>[1, 3, H, W]</c> or <c>[3, H, W]</c>, RGB in [0, 1].</param>
    /// <param name="resolution">The marching-cubes resolution, or null for the configured 192.</param>
    /// <param name="numInferenceSteps">DDIM steps, or null for the configured 50.</param>
    /// <param name="guidanceScale">The guidance scale, or null for the configured 3.</param>
    /// <param name="seed">The seed of the sampling and of the reconstruction, or null.</param>
    public Mesh3D<T> GenerateMeshFromImage(Tensor<T> image, int? resolution = null, int? numInferenceSteps = null,
        double? guidanceScale = null, int? seed = null)
        => ReconstructMesh(GenerateViews(image, numInferenceSteps, guidanceScale, seed), resolution, seed);

    /// <summary>
    /// Generates a colored point cloud of the object in <paramref name="image"/>: points drawn uniformly by area
    /// on the reconstructed surface, colored by the reconstruction.
    /// </summary>
    /// <param name="image">The input, <c>[1, 3, H, W]</c> or <c>[3, H, W]</c>, RGB in [0, 1].</param>
    /// <param name="numPoints">The point count, or null for <see cref="ThreeDDiffusionModelBase{T}.DefaultPointCount"/>.</param>
    /// <param name="numInferenceSteps">DDIM steps, or null for the configured 50.</param>
    /// <param name="guidanceScale">The guidance scale, or null for the configured 3.</param>
    /// <param name="seed">The seed of the sampling, the reconstruction and the point draw, or null.</param>
    /// <returns><c>[1, N, 6]</c>: xyz then rgb.</returns>
    public Tensor<T> GeneratePointCloudFromImage(Tensor<T> image, int? numPoints = null, int? numInferenceSteps = null,
        double? guidanceScale = null, int? seed = null)
    {
        int count = numPoints ?? DefaultPointCount;
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(numPoints), "At least one point is needed.");
        var field = ReconstructField(GenerateViews(image, numInferenceSteps, guidanceScale, seed), seed);
        var mesh = ExtractMesh(field, _options.MeshResolution);
        return SampleSurface(field, mesh, count, CreateInferenceRng(seed));
    }

    /// <summary>
    /// <paramref name="count"/> points drawn uniformly by area on <paramref name="mesh"/> (a face by its area, then a
    /// uniform barycentric point), colored by <paramref name="field"/> seen along the face's inward normal.
    /// </summary>
    internal Tensor<T> SampleSurface(NeuSField<T> field, Mesh3D<T> mesh, int count, Random random)
    {
        int faces = mesh.Faces.GetLength(0);
        if (faces == 0 || mesh.Vertices.Shape[0] == 0)
            throw new InvalidOperationException("The reconstructed surface is empty, so there are no points to sample.");
        var vertices = NeuSHost<T>.ToDoubles(mesh.Vertices);
        var cumulative = new double[faces];
        double total = 0;
        var faceNormals = new double[faces * 3];
        for (int f = 0; f < faces; f++)
        {
            var (a, b, c) = Corners(vertices, mesh.Faces, f);
            double ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
            double vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
            double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
            double length = Math.Sqrt(nx * nx + ny * ny + nz * nz);
            total += 0.5 * length;
            cumulative[f] = total;
            if (length > 0)
            {
                faceNormals[f * 3] = nx / length;
                faceNormals[f * 3 + 1] = ny / length;
                faceNormals[f * 3 + 2] = nz / length;
            }
        }

        if (!(total > 0)) throw new InvalidOperationException("The reconstructed surface has no area to sample.");
        var points = new double[count * 3];
        var inward = new double[count * 3];
        for (int i = 0; i < count; i++)
        {
            double target = random.NextDouble() * total;
            int f = Array.BinarySearch(cumulative, target);
            if (f < 0) f = ~f;
            if (f >= faces) f = faces - 1;
            var (a, b, c) = Corners(vertices, mesh.Faces, f);
            // Uniform on the triangle: sqrt(r1) (Turk 1990, Osada et al. 2002).
            double r1 = Math.Sqrt(random.NextDouble()), r2 = random.NextDouble();
            double wa = 1.0 - r1, wb = r1 * (1.0 - r2), wc = r1 * r2;
            for (int d = 0; d < 3; d++)
            {
                points[i * 3 + d] = wa * a[d] + wb * b[d] + wc * c[d];
                inward[i * 3 + d] = -faceNormals[f * 3 + d];
            }
        }

        var colors = NeuSHost<T>.ToDoubles(field.QueryColor(NeuSHost<T>.FromDoubles(points, count, 3),
            NeuSHost<T>.FromDoubles(inward, count, 3)));
        var cloud = new double[count * 6];
        for (int i = 0; i < count; i++)
        {
            Array.Copy(points, i * 3, cloud, i * 6, 3);
            Array.Copy(colors, i * 3, cloud, i * 6 + 3, 3);
        }

        return NeuSHost<T>.FromDoubles(cloud, 1, count, 6);
    }

    private static (double[] A, double[] B, double[] C) Corners(double[] vertices, int[,] faces, int face)
    {
        double[] Vertex(int index) => new[] { vertices[index * 3], vertices[index * 3 + 1], vertices[index * 3 + 2] };
        return (Vertex(faces[face, 0]), Vertex(faces[face, 1]), Vertex(faces[face, 2]));
    }

    #endregion

    #region I3DDiffusionModel

    /// <summary>
    /// Image-to-3D with Wonder3D: the cross-domain diffusion's six normal maps and color images, fused by the
    /// geometry-aware NeuS reconstruction into a textured mesh.
    /// </summary>
    /// <param name="inputImage">The input, <c>[1, 3, H, W]</c> or <c>[3, H, W]</c>, RGB in [0, 1].</param>
    /// <param name="numViews">Not used: Wonder3D always generates its configured views (six).</param>
    /// <param name="numInferenceSteps">DDIM steps.</param>
    /// <param name="guidanceScale">The guidance scale.</param>
    /// <param name="seed">The seed, or null.</param>
    public override Mesh3D<T> ImageTo3D(Tensor<T> inputImage, int numViews = 4, int numInferenceSteps = 50,
        double guidanceScale = 3.0, int? seed = null)
        => GenerateMeshFromImage(inputImage, null, numInferenceSteps, guidanceScale, seed);

    /// <summary>
    /// Renders the reconstruction of <paramref name="inputImage"/> from each target angle with an orthographic
    /// camera like Wonder3D's own views.
    /// </summary>
    /// <param name="inputImage">The input, <c>[1, 3, H, W]</c> or <c>[3, H, W]</c>, RGB in [0, 1].</param>
    /// <param name="targetAngles">The (azimuth, elevation) of each view in radians, azimuth from the front toward the
    /// right.</param>
    /// <param name="numInferenceSteps">DDIM steps.</param>
    /// <param name="guidanceScale">The guidance scale.</param>
    /// <param name="seed">The seed, or null.</param>
    /// <returns>One <c>[1, 3, S, S]</c> image per angle, colors in [0, 1].</returns>
    public override Tensor<T>[] SynthesizeNovelViews(Tensor<T> inputImage, (double azimuth, double elevation)[] targetAngles,
        int numInferenceSteps = 50, double guidanceScale = 3.0, int? seed = null)
    {
        if (targetAngles is null) throw new ArgumentNullException(nameof(targetAngles));
        var field = ReconstructField(GenerateViews(inputImage, numInferenceSteps, guidanceScale, seed), seed);
        int size = _options.ImageSize;
        var images = new Tensor<T>[targetAngles.Length];
        for (int i = 0; i < targetAngles.Length; i++)
        {
            var camera = OrthographicCamera(targetAngles[i].azimuth, targetAngles[i].elevation, size, size);
            var rendered = field.RenderImage(camera, size, size);
            images[i] = Engine.Reshape(Engine.TensorPermute(rendered, new[] { 2, 0, 1 }), new[] { 1, 3, size, size });
        }

        return images;
    }

    /// <inheritdoc />
    /// <remarks>Wonder3D has no text conditioning; use <see cref="GeneratePointCloudFromImage"/>
    /// with an image.</remarks>
    public override Tensor<T> GeneratePointCloud(string prompt, string? negativePrompt = null,
        int? numPoints = null, int numInferenceSteps = 50, double guidanceScale = 7.5, int? seed = null)
        => throw new NotSupportedException(
            "Wonder3D generates from an image (SD image-variations conditioning) and has no text conditioning; " +
            "call GeneratePointCloudFromImage(image) instead.");

    /// <inheritdoc />
    /// <remarks>Wonder3D has no text conditioning; use <see cref="GenerateMeshFromImage"/>
    /// or <see cref="ImageTo3D"/> with an image.</remarks>
    public override Mesh3D<T> GenerateMesh(string prompt, string? negativePrompt = null,
        int resolution = 256, int numInferenceSteps = 50, double guidanceScale = 7.5, int? seed = null)
        => throw new NotSupportedException(
            "Wonder3D generates from an image (SD image-variations conditioning) and has no text conditioning; " +
            "call GenerateMeshFromImage(image) or ImageTo3D instead.");

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = _options;
        var metadata = new ModelMetadata<T>
        {
            Name = "Wonder3D", Version = "1.0",
            Description = "Wonder3D cross-domain multi-view diffusion of normal maps and colors with geometry-aware NeuS fusion",
            FeatureCount = (int)Math.Min(int.MaxValue, ParameterCount), Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "cross-domain-multiview-unet");
        metadata.SetProperty("num_views", o.NumViews);
        metadata.SetProperty("image_size", o.ImageSize);
        metadata.SetProperty("cross_attention_dim", o.CrossAttentionDim);
        metadata.SetProperty("base_channels", o.BaseChannels);
        metadata.SetProperty("unet_in_channels", 2 * o.LatentChannels);
        metadata.SetProperty("class_embedding_input_dim", Wonder3DDomainEmbedding.InputDim);
        metadata.SetProperty("generates_normals", true);
        metadata.SetProperty("multiview_attention", true);
        metadata.SetProperty("cross_domain_attention", true);
        metadata.SetProperty("reconstruction", "NeuS geometry-aware normal fusion (orthographic)");
        metadata.SetProperty("scheduler", "DDIM");
        metadata.SetProperty("inference_steps", o.InferenceSteps);
        metadata.SetProperty("guidance_scale", o.GuidanceScale);
        metadata.SetProperty("eta", o.Eta);
        metadata.SetProperty("mesh_resolution", o.MeshResolution);
        return metadata;
    }

    #endregion
}
