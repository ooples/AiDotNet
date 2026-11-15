using AiDotNet.Attributes;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.ThreeD.Meshing;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// TripoSR (Tochilkin et al. 2024): feed-forward single-image 3D reconstruction with an LRM-style transformer that
/// predicts a triplane NeRF from one image.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The pipeline follows the paper and the official VAST-AI-Research/TripoSR code, which builds on LRM (Hong et al.
/// 2023):
/// </para>
/// <list type="number">
/// <item><description><b>Image encoder.</b> A DINO ViT-B/16 (HF <c>ViTModel</c>, no pooler) encodes the 512x512 image,
/// ImageNet-normalized, to 1 + 32x32 tokens of width 768. The learned position embeddings (stored at DINO's 14x14
/// pretraining grid) are resized bicubically to the input grid. Unlike LRM, the encoder is camera-free: no camera
/// parameters modulate it.</description></item>
/// <item><description><b>Triplane transformer.</b> 3x32x32 learnable triplane tokens of width 1024 pass a
/// <c>Transformer1D</c>: GroupNorm(32), a projection to 16 heads x 64, 16 blocks of self-attention, cross-attention
/// to the image tokens and a GEGLU feed-forward, a projection back, and a residual.</description></item>
/// <item><description><b>Upsampler.</b> A shared stride-2 <c>ConvTranspose2d</c> turns the token planes into
/// triplanes [3, 40, 64, 64].</description></item>
/// <item><description><b>Triplane NeRF decoder.</b> A point in [-0.87, 0.87]^3 samples each plane bilinearly; the
/// three features are concatenated and a 9-hidden-layer SiLU MLP of width 64 decodes a density
/// <c>exp(raw - 1)</c> and a colour <c>sigmoid(features)</c>.</description></item>
/// <item><description><b>Rendering.</b> Rays are clipped to the cube, sampled 128 times and composited with
/// <see cref="NeuralRadianceFields.Helpers.VolumeRenderer{T}"/>.</description></item>
/// </list>
/// <para>
/// Geometry: <see cref="TriplaneField{T}.SampleDensityGrid"/> samples the density on a dense grid, the point cloud is
/// the occupied samples (density above 25, TripoSR's surface threshold) with their decoded colours, and the mesh is
/// the marching-cubes surface of that grid at the same threshold, with vertex colours decoded at the vertices
/// (TripoSR's <c>extract_mesh</c>: resolution 256, threshold 25, radius 0.87).
/// </para>
/// <para>
/// <b>Weights.</b> The architecture is complete, but meaningful reconstructions need the released weights
/// (stabilityai/TripoSR <c>model.ckpt</c>); load them with <see cref="LoadTorchWeights"/>, which reads the official
/// state-dict names. Freshly initialized weights produce a valid, input-dependent but untrained field. Training is not
/// provided: TripoSR is trained with rendering losses against multi-view renders, which the diffusion
/// <c>Train</c> step (a denoising step) cannot express, and it has no noise predictor or VAE.
/// </para>
/// <para>
/// <b>For Beginners:</b> Give TripoSR one photo of an object and it predicts the whole 3D object in one pass: no
/// diffusion steps. You can ask the result for a voxel grid, a coloured point cloud, a coloured mesh, or pictures
/// from new viewpoints.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var model = new TripoSRModel&lt;float&gt;();
/// model.LoadTorchWeights((name, shape) =&gt; checkpoint.Read(name));   // stabilityai/TripoSR model.ckpt
/// var image = LoadImage("chair.png");                                 // [1, 3, H, W] in [0, 1]
/// var field = model.Reconstruct(image);
/// var mesh = model.GenerateMesh(image);                               // marching cubes at density 25
/// var cloud = model.GeneratePointCloud(image);                        // [1, P, 6] xyz + rgb
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ThreeDGeneration)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TripoSR: Fast 3D Object Reconstruction from a Single Image", "https://arxiv.org/abs/2403.02151", Year = 2024, Authors = "Tochilkin et al.")]
public partial class TripoSRModel<T> : ThreeDDiffusionModelBase<T>
{
    #region Fields

    private readonly TripoSROptions _options;

    /// <summary>The whole network; the parameter registry reaches it through the generated registration.</summary>
    private readonly TripoSRNetwork<T> _network;

    #endregion

    #region Constructor

    /// <summary>
    /// Builds TripoSR with the given configuration (the official configuration when null). Weights are allocated
    /// lazily, on first use, and initialized from <see cref="ModelOptions.Seed"/> when it is set.
    /// </summary>
    /// <param name="options">The configuration; the official TripoSR configuration when null.</param>
    /// <param name="architecture">An optional architecture; only its random seed is used.</param>
    public TripoSRModel(TripoSROptions? options = null, NeuralNetworkArchitecture<T>? architecture = null)
        : base(
            new DiffusionModelOptions<T> { TrainTimesteps = 1, Seed = options?.Seed },
            new DDIMScheduler<T>(SchedulerConfig<T>.CreateDefault()),
            ValidatedPointBudget(options),
            architecture)
    {
        _options = options ?? new TripoSROptions();
        Validate(_options);
        LayerInitializationSeedScope.ResetForModelConstruction(architecture?.RandomSeed ?? _options.Seed);
        _network = new TripoSRNetwork<T>(_options);
    }

    // The base's default point count only labels the model; TripoSR's clouds hold every occupied grid sample.
    private static int ValidatedPointBudget(TripoSROptions? options)
    {
        int resolution = (options ?? new TripoSROptions()).GridResolution;
        return resolution > 1 ? resolution * resolution : 4096;
    }

    private static void Validate(TripoSROptions o)
    {
        void Positive(int value, string name)
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(name, $"{name} must be positive.");
        }

        Positive(o.ImageSize, nameof(o.ImageSize));
        Positive(o.PatchSize, nameof(o.PatchSize));
        Positive(o.EncoderLayers, nameof(o.EncoderLayers));
        Positive(o.EncoderHeads, nameof(o.EncoderHeads));
        Positive(o.EncoderHeadDim, nameof(o.EncoderHeadDim));
        Positive(o.EncoderMlpRatio, nameof(o.EncoderMlpRatio));
        Positive(o.EncoderPositionGridSize, nameof(o.EncoderPositionGridSize));
        Positive(o.TriplaneTokenResolution, nameof(o.TriplaneTokenResolution));
        Positive(o.TriplaneTokenChannels, nameof(o.TriplaneTokenChannels));
        Positive(o.TransformerLayers, nameof(o.TransformerLayers));
        Positive(o.TransformerHeads, nameof(o.TransformerHeads));
        Positive(o.TransformerHeadDim, nameof(o.TransformerHeadDim));
        Positive(o.TransformerNormGroups, nameof(o.TransformerNormGroups));
        Positive(o.TriplaneChannels, nameof(o.TriplaneChannels));
        Positive(o.DecoderHiddenWidth, nameof(o.DecoderHiddenWidth));
        Positive(o.DecoderHiddenLayers, nameof(o.DecoderHiddenLayers));
        Positive(o.SamplesPerRay, nameof(o.SamplesPerRay));
        Positive(o.QueryChunkSize, nameof(o.QueryChunkSize));
        Positive(o.RenderResolution, nameof(o.RenderResolution));
        if (o.ImageSize % o.PatchSize != 0)
            throw new ArgumentException($"ImageSize {o.ImageSize} must be a multiple of PatchSize {o.PatchSize}.", nameof(o));
        if (o.TriplaneTokenChannels % o.TransformerNormGroups != 0)
            throw new ArgumentException(
                $"TriplaneTokenChannels {o.TriplaneTokenChannels} must split into {o.TransformerNormGroups} norm groups.", nameof(o));
        if (o.GridResolution < 2)
            throw new ArgumentOutOfRangeException(nameof(o.GridResolution), "GridResolution must be at least 2.");
        if (!(o.Radius > 0) || double.IsInfinity(o.Radius))
            throw new ArgumentOutOfRangeException(nameof(o.Radius), "Radius must be positive and finite.");
        if (!(o.EncoderLayerNormEpsilon >= 0))
            throw new ArgumentOutOfRangeException(nameof(o.EncoderLayerNormEpsilon), "The layer-norm epsilon cannot be negative.");
        if (double.IsNaN(o.DensityThreshold) || double.IsNaN(o.DensityBias))
            throw new ArgumentException("DensityThreshold and DensityBias must be numbers.", nameof(o));
        if (!(o.CameraDistance > 0))
            throw new ArgumentOutOfRangeException(nameof(o.CameraDistance), "CameraDistance must be positive.");
        if (!(o.FieldOfViewDegrees > 0 && o.FieldOfViewDegrees < 180))
            throw new ArgumentOutOfRangeException(nameof(o.FieldOfViewDegrees), "FieldOfViewDegrees must be in (0, 180).");
        if (o.ImageMean is null || o.ImageMean.Length != 3 || o.ImageStd is null || o.ImageStd.Length != 3)
            throw new ArgumentException("ImageMean and ImageStd must each hold three values (RGB).", nameof(o));
        foreach (double std in o.ImageStd)
            if (!(std > 0)) throw new ArgumentException("ImageStd values must be positive.", nameof(o));
        if (!TsrActivations<T>.IsSupported(o.DecoderActivation))
            throw new ArgumentException($"The decoder does not support the {o.DecoderActivation} activation.", nameof(o));
        if (!Enum.IsDefined(typeof(TriplaneFeatureReduction), o.FeatureReduction))
            throw new ArgumentException($"Unknown feature reduction {o.FeatureReduction}.", nameof(o));
        if (!Enum.IsDefined(typeof(TriplaneDensityActivation), o.DensityActivation))
            throw new ArgumentException($"Unknown density activation {o.DensityActivation}.", nameof(o));
    }

    #endregion

    #region Properties

    /// <summary>The configuration this model was built with.</summary>
    public TripoSROptions Configuration => _options;

    /// <summary>TripoSR is feed-forward: it has no noise predictor.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    [ExternalState]
    public override INoisePredictor<T> NoisePredictor => throw new NotSupportedException(
        "TripoSR is a feed-forward reconstruction model with no noise predictor; use Reconstruct, Predict or ImageTo3D.");

    /// <summary>TripoSR encodes images with a DINO ViT, not a VAE.</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    [ExternalState]
    public override IVAEModel<T> VAE => throw new NotSupportedException(
        "TripoSR has no VAE: its image encoder is a DINO ViT and its 3D representation is a triplane.");

    /// <summary>TripoSR has no text conditioning; it is conditioned on an image.</summary>
    public override IConditioningModule<T>? Conditioner => null;

    /// <summary>The channels of each triplane, TripoSR's scene code.</summary>
    public override int LatentChannels => _options.TriplaneChannels;

    /// <inheritdoc />
    public override bool SupportsPointCloud => true;

    /// <inheritdoc />
    public override bool SupportsMesh => true;

    /// <summary>Colour is decoded per point and per vertex, but TripoSR bakes no texture image.</summary>
    public override bool SupportsTexture => false;

    /// <inheritdoc />
    public override bool SupportsNovelView => true;

    /// <inheritdoc />
    public override bool SupportsScoreDistillation => false;

    #endregion

    #region Reconstruction

    /// <summary>
    /// The scene codes (triplanes) of RGB images: [B, 3, C, 2R, 2R] for images [B, 3, H, W] (or one [3, H, W]) with
    /// values in [0, 1]. Images whose side differs from <see cref="TripoSROptions.ImageSize"/> are resized bilinearly.
    /// </summary>
    /// <param name="images">The images, foreground-centred as TripoSR expects.</param>
    public Tensor<T> EncodeSceneCodes(Tensor<T> images)
    {
        var batch = AsImageBatch(images);
        return _network.SceneCodes(batch);
    }

    /// <summary>Decodes one image to its triplane field.</summary>
    /// <param name="image">The image [1, 3, H, W] or [3, H, W], values in [0, 1].</param>
    public TriplaneField<T> Reconstruct(Tensor<T> image)
    {
        var batch = AsImageBatch(image);
        if (batch.Shape[0] != 1)
            throw new ArgumentException("Reconstruct takes one image; use EncodeSceneCodes for a batch.", nameof(image));
        var codes = _network.SceneCodes(batch);
        var planes = Engine.Reshape(codes, new[] { 3, codes.Shape[2], codes.Shape[3], codes.Shape[4] });
        return new TriplaneField<T>(planes, _network.Decoder, _options);
    }

    /// <summary>
    /// The activated density of the image's field on a dense grid [resolution, resolution, resolution] spanning
    /// <c>linspace(-bounds, bounds, resolution)</c> per axis (see <see cref="TriplaneField{T}.SampleDensityGrid"/>).
    /// </summary>
    /// <param name="image">The image [1, 3, H, W] or [3, H, W], values in [0, 1].</param>
    /// <param name="resolution">Samples per axis; <see cref="TripoSROptions.GridResolution"/> when null.</param>
    /// <param name="bounds">The half-extent; <see cref="TripoSROptions.Radius"/> when null.</param>
    public Tensor<T> SampleDensityGrid(Tensor<T> image, int? resolution = null, double? bounds = null)
        => Reconstruct(image).SampleDensityGrid(
            resolution ?? _options.GridResolution, bounds ?? _options.Radius);

    /// <summary>
    /// The image's point cloud [1, P, 6] (XYZ, RGB): the grid samples whose density exceeds the threshold, with the
    /// colour the decoder gives each of them.
    /// </summary>
    /// <param name="image">The image [1, 3, H, W] or [3, H, W], values in [0, 1].</param>
    /// <param name="maxPoints">Keep at most this many points, the densest; every occupied sample when null.</param>
    /// <param name="resolution">Grid samples per axis; <see cref="TripoSROptions.GridResolution"/> when null.</param>
    /// <param name="threshold">The occupancy threshold; <see cref="TripoSROptions.DensityThreshold"/> when null.</param>
    public Tensor<T> GeneratePointCloud(Tensor<T> image, int? maxPoints = null, int? resolution = null, double? threshold = null)
        => Reconstruct(image).SamplePointCloud(
            resolution ?? _options.GridResolution, _options.Radius,
            threshold ?? _options.DensityThreshold, maxPoints);

    /// <summary>
    /// The image's mesh, as TripoSR's <c>extract_mesh</c> builds it: marching cubes on the density grid over
    /// [-radius, radius]^3 at the threshold (inside where the density is above it), with each vertex coloured by
    /// the decoder at the vertex. A field with no occupied sample gives an empty mesh.
    /// </summary>
    /// <param name="image">The image [1, 3, H, W] or [3, H, W], values in [0, 1].</param>
    /// <param name="resolution">Grid samples per axis; <see cref="TripoSROptions.GridResolution"/> (256) when null.</param>
    /// <param name="threshold">The iso-level; <see cref="TripoSROptions.DensityThreshold"/> (25) when null.</param>
    public Mesh3D<T> GenerateMesh(Tensor<T> image, int? resolution = null, double? threshold = null)
        => ExtractMesh(Reconstruct(image), resolution ?? _options.GridResolution,
            threshold ?? _options.DensityThreshold);

    /// <summary>
    /// The coloured marching-cubes mesh of <paramref name="field"/> on a grid of <paramref name="resolution"/>
    /// samples per axis over the field's cube, at density <paramref name="threshold"/>.
    /// </summary>
    /// <param name="field">The decoded field.</param>
    /// <param name="resolution">Grid samples per axis, at least 2.</param>
    /// <param name="threshold">The iso-level; the inside is where the density exceeds it.</param>
    public Mesh3D<T> ExtractMesh(TriplaneField<T> field, int resolution, double threshold)
    {
        if (field is null) throw new ArgumentNullException(nameof(field));
        var grid = field.SampleDensityGrid(resolution, field.Radius);
        var mesh = MarchingCubes.ExtractInCube(grid, threshold, IsoSurfaceInterior.AboveIsoLevel, field.Radius);
        if (mesh.Vertices.Shape[0] > 0)
            mesh.Colors = Engine.TensorNarrow(field.Query(mesh.Vertices), 1, 1, 3).Contiguous();
        return mesh;
    }

    /// <summary>
    /// Renders the image's field from a camera orbiting the origin (TripoSR's <c>get_spherical_cameras</c>, at the
    /// configured distance, field of view and resolution).
    /// </summary>
    /// <param name="image">The image [1, 3, H, W] or [3, H, W], values in [0, 1].</param>
    /// <param name="azimuthDegrees">The camera azimuth, in degrees.</param>
    /// <param name="elevationDegrees">The camera elevation, in degrees.</param>
    /// <returns>The view [1, 3, RenderResolution, RenderResolution].</returns>
    public Tensor<T> RenderView(Tensor<T> image, double azimuthDegrees, double elevationDegrees = 0.0)
        => RenderField(Reconstruct(image), azimuthDegrees, elevationDegrees);

    private Tensor<T> RenderField(TriplaneField<T> field, double azimuthDegrees, double elevationDegrees)
        => field.RenderView(azimuthDegrees, elevationDegrees, _options.CameraDistance,
            _options.FieldOfViewDegrees, _options.RenderResolution, _options.RenderResolution);

    private static Tensor<T> AsImageBatch(Tensor<T> images)
    {
        if (images is null) throw new ArgumentNullException(nameof(images));
        var batch = images.Rank == 3
            ? AiDotNetEngine.Current.Reshape(images, new[] { 1, images.Shape[0], images.Shape[1], images.Shape[2] })
            : images;
        if (batch.Rank != 4 || batch.Shape[1] != 3)
            throw new ArgumentException("Images must be RGB, [B, 3, H, W] or [3, H, W].", nameof(images));
        return batch;
    }

    #endregion

    #region I3DDiffusionModel overrides

    /// <summary>
    /// The scene codes of <paramref name="input"/> images [B, 3, H, W] (see <see cref="EncodeSceneCodes"/>): TripoSR's
    /// forward pass is one feed-forward step, not a denoising loop.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input) => EncodeSceneCodes(input);

    /// <summary>
    /// Image-to-3D in one feed-forward pass: the coloured mesh of the image overload of <c>GenerateMesh</c>
    /// at the configured resolution and threshold.
    /// </summary>
    /// <param name="inputImage">The image [1, 3, H, W] or [3, H, W], values in [0, 1].</param>
    /// <param name="numViews">Not used: TripoSR reconstructs from the single view directly.</param>
    /// <param name="numInferenceSteps">Not used: there is no iterative sampling.</param>
    /// <param name="guidanceScale">Not used: there is no guidance.</param>
    /// <param name="seed">Not used: inference is deterministic.</param>
    public override Mesh3D<T> ImageTo3D(Tensor<T> inputImage, int numViews = 4, int numInferenceSteps = 50,
        double guidanceScale = 3.0, int? seed = null)
        => GenerateMesh(inputImage);

    /// <summary>
    /// Renders the image's reconstructed field from each target angle (azimuth and elevation in radians, as
    /// <c>GenerateViewAngles</c> gives them).
    /// </summary>
    /// <param name="inputImage">The image [1, 3, H, W] or [3, H, W], values in [0, 1].</param>
    /// <param name="targetAngles">The (azimuth, elevation) of each view, in radians.</param>
    /// <param name="numInferenceSteps">Not used: views are rendered, not sampled.</param>
    /// <param name="guidanceScale">Not used.</param>
    /// <param name="seed">Not used: rendering is deterministic.</param>
    public override Tensor<T>[] SynthesizeNovelViews(Tensor<T> inputImage, (double azimuth, double elevation)[] targetAngles,
        int numInferenceSteps = 50, double guidanceScale = 3.0, int? seed = null)
    {
        if (targetAngles is null) throw new ArgumentNullException(nameof(targetAngles));
        var field = Reconstruct(inputImage);
        var views = new Tensor<T>[targetAngles.Length];
        for (int v = 0; v < targetAngles.Length; v++)
            views[v] = RenderField(field, targetAngles[v].azimuth * 180.0 / Math.PI, targetAngles[v].elevation * 180.0 / Math.PI);
        return views;
    }

    /// <summary>TripoSR is conditioned on an image, not text.</summary>
    /// <exception cref="NotSupportedException">Always; pass an image to the image overload instead.</exception>
    public override Tensor<T> GeneratePointCloud(string prompt, string? negativePrompt = null,
        int? numPoints = null, int numInferenceSteps = 1, double guidanceScale = 1.0, int? seed = null)
        => throw new NotSupportedException(
            "TripoSR reconstructs from an image and has no text conditioning; call GeneratePointCloud(image) instead.");

    /// <summary>TripoSR is conditioned on an image, not text.</summary>
    /// <exception cref="NotSupportedException">Always; pass an image to the image overload or to ImageTo3D.</exception>
    public override Mesh3D<T> GenerateMesh(string prompt, string? negativePrompt = null,
        int resolution = 256, int numInferenceSteps = 1, double guidanceScale = 1.0, int? seed = null)
        => throw new NotSupportedException(
            "TripoSR reconstructs from an image and has no text conditioning; call GenerateMesh(image) or ImageTo3D instead.");

    /// <inheritdoc />
    public override int[] GetInputShape() => new[] { 1, 3, _options.ImageSize, _options.ImageSize };

    /// <inheritdoc />
    public override int[] GetOutputShape() => new[]
    {
        1, 3, _options.TriplaneChannels, _options.TriplaneResolution, _options.TriplaneResolution
    };

    #endregion

    #region Weights

    /// <summary>
    /// Loads the released TripoSR weights by their official state-dict names (<c>image_tokenizer.model.*</c>,
    /// <c>tokenizer.embeddings</c>, <c>backbone.*</c>, <c>post_processor.upsample.*</c>, <c>decoder.layers.*</c>).
    /// </summary>
    /// <param name="read">Returns the values of the named tensor, row-major, given its name and expected shape.</param>
    /// <returns>The names read, in order.</returns>
    public IReadOnlyCollection<string> LoadTorchWeights(Func<string, int[], double[]> read)
        => KlModule<T>.Load(_network.NamedTensors(string.Empty), read);

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = _options;
        var metadata = new ModelMetadata<T>
        {
            Name = "TripoSR",
            Version = "1.0",
            Description = "TripoSR feed-forward single-image 3D reconstruction: DINO ViT, triplane transformer, triplane NeRF",
            FeatureCount = (int)Math.Min(int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "dino-vit + transformer1d + triplane-nerf");
        metadata.SetProperty("image_size", o.ImageSize);
        metadata.SetProperty("encoder_width", o.EncoderWidth);
        metadata.SetProperty("encoder_layers", o.EncoderLayers);
        metadata.SetProperty("transformer_width", o.TransformerWidth);
        metadata.SetProperty("transformer_layers", o.TransformerLayers);
        metadata.SetProperty("transformer_heads", o.TransformerHeads);
        metadata.SetProperty("triplane_shape", $"3x{o.TriplaneChannels}x{o.TriplaneResolution}x{o.TriplaneResolution}");
        metadata.SetProperty("decoder_hidden_layers", o.DecoderHiddenLayers);
        metadata.SetProperty("density_threshold", o.DensityThreshold);
        metadata.SetProperty("feed_forward", true);
        return metadata;
    }

    #endregion
}
