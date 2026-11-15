using AiDotNet.Attributes;
using AiDotNet.Diffusion.ThreeD.Meshing;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralRadianceFields.NeuS;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// One-2-3-45 (Liu et al. 2023): single image to textured 3D mesh with Zero123 multi-view prediction, elevation
/// estimation and a feed-forward SparseNeuS reconstruction, without per-shape optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The pipeline follows the official release (One-2-3-45/One-2-3-45, <c>run.py</c>):
/// <list type="number">
/// <item><description>Stage 1, part 1: Zero123 predicts 4 views at the input's elevation, azimuths +0, +90, +180 and
/// +270 (<c>predict_stage1_gradio</c>, views 0 to 3).</description></item>
/// <item><description>Zero123 predicts 4 nearby views of view 0 (polar ∓10°, azimuth ∓10°), and the input's polar
/// angle is estimated from them: matches between the 4 views are triangulated under each candidate elevation and
/// the hypothesis with the least confidence-weighted reprojection error wins (coarse 30 to 140 by 10, then 1°).</description></item>
/// <item><description>Stage 1, part 2: 4 more views on a ring 30° below the input (or above it when the estimated polar
/// angle exceeds 75°), azimuths +30, +120, +210, +300.</description></item>
/// <item><description>Stage 2: 4 nearby views for each of the other 7 first-stage views, so 8 × 4 = 32 source views,
/// each placed at its absolute pose (input azimuth 30°, radius 1.2, focal 280 px at 256 px).</description></item>
/// <item><description>SparseNeuS (the lod-0 network of <c>one2345_lod0_val_demo.conf</c>: 96³ volume) encodes the
/// 32 views in the reference camera's frame, normalized by the frusta's bounding box; marching cubes extracts the
/// SDF's zero level on a 256³ grid over [−1, 1]³; the rendering network colours each vertex as seen from the
/// reference camera; and the mesh is mapped back to the object frame.</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Weights.</b> Zero123 and SparseNeuS need the released checkpoints (<c>zero123-xl.ckpt</c> and the One-2-3-45
/// SparseNeuS checkpoint) to produce meaningful shapes. Load SparseNeuS with <see cref="LoadReconstructionParameters"/>
/// using the reference state-dict names (<see cref="GetReconstructionParameters"/> lists them). With random weights
/// the pipeline still runs end to end and its output depends on the input, but it is not a reconstruction of it.
/// </para>
/// <para>
/// <b>Elevation matching.</b> As in the official code, the nearby views are matched with LoFTR
/// (<see cref="LoFTRMatcher{T}"/>, the full architecture at the configuration One-2-3-45 runs). Load its released
/// <c>indoor_ds_new.ckpt</c> weights with <see cref="LoFTRMatcher{T}.LoadParameters"/> and pass the matcher as
/// <c>featureMatcher</c>; untrained, the elevation estimate is not meaningful. Any other
/// <see cref="IViewFeatureMatcher{T}"/>, such as <see cref="NormalizedCrossCorrelationMatcher{T}"/>, can be
/// supplied instead.
/// </para>
/// <para>
/// <b>Text to 3D.</b> As in the paper, text goes through a text-to-image model first; pass one as
/// <c>textToImage</c> to use <see cref="GenerateMesh"/> and <see cref="GeneratePointCloud"/>.
/// </para>
/// <para>
/// <b>For Beginners:</b> give it one picture of an object (values in [−1, 1], shape [1, 3, H, W]) and it imagines
/// 32 other viewpoints, works out how high the original camera was, and builds a coloured 3D mesh from all the
/// views in one pass.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var one2345 = new One2345Model&lt;float&gt;();
/// one2345.LoadReconstructionParameters(sparseNeuSCheckpoint);
/// var mesh = one2345.ImageTo3D(image, seed: 42);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ThreeDGeneration)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization", "https://arxiv.org/abs/2306.16928", Year = 2023, Authors = "Liu et al.")]
public partial class One2345Model<T> : ThreeDDiffusionModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>Zero123 (U-Net, VAE, image and pose encoders), then the SparseNeuS reconstruction network.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_zero123);
        RegisterParameterComponent(_reconstructionParameters);
    }

    #region Constants

    private const int LATENT_CHANNELS = 4;
    private const int DEFAULT_POINT_COUNT = 4096;
    private const int DEFAULT_IMAGE_SIZE = 256;
    private const int SOURCE_VIEW_COUNT = 32;

    #endregion

    #region Fields

    private readonly Zero123Model<T> _zero123;
    private readonly One2345Options _pipelineOptions;
    private readonly SparseNeuS<T> _reconstruction;
    private readonly SparseNeuSParameterSource<T> _reconstructionParameters;
    private readonly IViewFeatureMatcher<T> _featureMatcher;
    private readonly ILatentDiffusionModel<T>? _textToImage;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _zero123.NoisePredictor;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _zero123.VAE;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => null;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
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

    /// <summary>The pipeline configuration.</summary>
    public One2345Options PipelineOptions => _pipelineOptions;

    /// <summary>The elevation matcher (a <see cref="LoFTRMatcher{T}"/> unless another was supplied); load the
    /// LoFTR checkpoint into it before estimating elevations.</summary>
    public IViewFeatureMatcher<T> FeatureMatcher => _featureMatcher;

    #endregion

    #region Constructor

    /// <summary>Creates a One-2-3-45 pipeline.</summary>
    /// <param name="architecture">Optional architecture for the Zero123 U-Net.</param>
    /// <param name="options">Diffusion options for Zero123 (SD 1.5 scaled-linear betas by default).</param>
    /// <param name="scheduler">Zero123's scheduler (DDIM by default, as the official sampler).</param>
    /// <param name="unet">An optional Zero123 U-Net (8 input channels); ignored when <paramref name="zero123"/> is given.</param>
    /// <param name="vae">An optional Zero123 VAE; ignored when <paramref name="zero123"/> is given.</param>
    /// <param name="defaultPointCount">Points sampled by <see cref="GeneratePointCloud"/> by default.</param>
    /// <param name="seed">Seeds weight initialization.</param>
    /// <param name="pipelineOptions">The pipeline configuration; the official values by default.</param>
    /// <param name="zero123">The multi-view model; a default Zero123 is built from the other arguments when null.</param>
    /// <param name="textToImage">The text-to-image model used for text-to-3D, or null.</param>
    /// <param name="featureMatcher">The elevation matcher; a <see cref="LoFTRMatcher{T}"/> with the official
    /// configuration (seeded by <paramref name="seed"/>) by default.</param>
    public One2345Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        int defaultPointCount = DEFAULT_POINT_COUNT,
        int? seed = null,
        One2345Options? pipelineOptions = null,
        Zero123Model<T>? zero123 = null,
        ILatentDiffusionModel<T>? textToImage = null,
        IViewFeatureMatcher<T>? featureMatcher = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.00085,
                BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            defaultPointCount, architecture)
    {
        _pipelineOptions = pipelineOptions ?? new One2345Options();
        _pipelineOptions.Validate();
        _zero123 = zero123 ?? new Zero123Model<T>(architecture, options, scheduler, unet, vae, DEFAULT_IMAGE_SIZE, seed);
        _textToImage = textToImage;
        _featureMatcher = featureMatcher ?? new LoFTRMatcher<T>(new LoFTROptions { Seed = seed });
        _reconstruction = new SparseNeuS<T>(CreateReconstructionOptions(_pipelineOptions, seed));
        _reconstructionParameters = new SparseNeuSParameterSource<T>(_reconstruction);
    }

    private static SparseNeuSOptions CreateReconstructionOptions(One2345Options o, int? seed)
    {
        int r = o.VolumeResolution;
        return new SparseNeuSOptions
        {
            VolumeDims = new[] { r, r, r },
            VolumeOrigin = new[] { -1.0, -1.0, -1.0 },
            VoxelSize = 2.0 / (r - 1),
            CompressedFeatureDim = o.CompressedFeatureDim,
            GeometryFeatureDim = o.GeometryFeatureDim,
            SdfHiddenDim = o.SdfHiddenDim,
            SdfLayers = o.SdfLayers,
            SdfFrequencies = o.SdfFrequencies,
            InitialVariance = o.InitialVariance,
            AntiAliasPooling = o.AntiAliasPooling,
            GridChunkSize = o.QueryChunkSize,
            Seed = seed,
        };
    }

    #endregion

    #region Reconstruction weights

    /// <summary>
    /// The SparseNeuS weights under the reference state-dict names (<c>pyramid_feature_network_geometry_lod0.*</c>,
    /// <c>sdf_network_lod0.*</c>, <c>variance_network_lod0.variance</c>, <c>rendering_network_lod0.*</c>).
    /// </summary>
    /// <returns>Each parameter's name and tensor.</returns>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> GetReconstructionParameters() => _reconstruction.NamedParameters();

    /// <summary>Loads SparseNeuS weights by reference name, e.g. converted from the One-2-3-45 checkpoint.</summary>
    /// <param name="parameters">Name to tensor; each name must be one of <see cref="GetReconstructionParameters"/>'s.</param>
    /// <exception cref="ArgumentException">A name is unknown or a shape does not match.</exception>
    public void LoadReconstructionParameters(IReadOnlyDictionary<string, Tensor<T>> parameters)
    {
        _reconstruction.LoadParameters(parameters);
        InvalidateTrainableParametersCache();
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Image to textured mesh with the full One-2-3-45 pipeline (see the class remarks).
    /// </summary>
    /// <param name="inputImage">The image, <c>[1, 3, H, W]</c> or <c>[3, H, W]</c> in [−1, 1]; H and W multiples of 8
    /// (256 in the paper), the object centered on a white background.</param>
    /// <param name="numViews">The source-view count of the fixed layout, 8 × 4 = 32; any other value is rejected.</param>
    /// <param name="numInferenceSteps">DDIM steps of the second-stage views (50 officially); the first stage uses
    /// <see cref="One2345Options.Stage1Steps"/>.</param>
    /// <param name="guidanceScale">Zero123's guidance scale (3 officially).</param>
    /// <param name="seed">Seeds the Zero123 sampling; each view gets its own derived seed.</param>
    /// <returns>The mesh in the object frame, with per-vertex colours in [0, 1] and normals.</returns>
    /// <exception cref="ArgumentException"><paramref name="numViews"/> is not 32.</exception>
    public override Mesh3D<T> ImageTo3D(
        Tensor<T> inputImage,
        int numViews = SOURCE_VIEW_COUNT,
        int numInferenceSteps = 50,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        if (numViews != SOURCE_VIEW_COUNT)
            throw new ArgumentException(
                $"One-2-3-45 reconstructs from its fixed layout of {SOURCE_VIEW_COUNT} source views (8 first-stage views " +
                $"× 4 nearby views); {numViews} was requested.", nameof(numViews));
        return Reconstruct(inputImage, _pipelineOptions.Stage1Steps, numInferenceSteps, guidanceScale,
            _pipelineOptions.MeshResolution, seed).Mesh;
    }

    /// <summary>
    /// Text to mesh: the text-to-image model renders the prompt, then <see cref="ImageTo3D"/>'s pipeline runs with
    /// the configured step counts and guidance.
    /// </summary>
    /// <param name="prompt">The prompt.</param>
    /// <param name="negativePrompt">The negative prompt.</param>
    /// <param name="resolution">The marching-cubes resolution (256 officially).</param>
    /// <param name="numInferenceSteps">The text-to-image model's steps.</param>
    /// <param name="guidanceScale">The text-to-image model's guidance scale.</param>
    /// <param name="seed">Seeds both the image and the views.</param>
    /// <returns>The coloured mesh.</returns>
    /// <exception cref="InvalidOperationException">No text-to-image model was supplied.</exception>
    public override Mesh3D<T> GenerateMesh(string prompt, string? negativePrompt = null,
        int resolution = 256, int numInferenceSteps = 50, double guidanceScale = 3.0, int? seed = null)
    {
        var image = RenderPrompt(prompt, negativePrompt, numInferenceSteps, guidanceScale, seed);
        return Reconstruct(image, _pipelineOptions.Stage1Steps, _pipelineOptions.Stage2Steps, _pipelineOptions.GuidanceScale,
            resolution, seed).Mesh;
    }

    /// <summary>
    /// Text to coloured point cloud: <see cref="GenerateMesh"/>'s surface, sampled uniformly by area with the
    /// vertex colours interpolated barycentrically.
    /// </summary>
    /// <param name="prompt">The prompt.</param>
    /// <param name="negativePrompt">The negative prompt.</param>
    /// <param name="numPoints">Points to sample; <see cref="ThreeDDiffusionModelBase{T}.DefaultPointCount"/> by default.</param>
    /// <param name="numInferenceSteps">The text-to-image model's steps.</param>
    /// <param name="guidanceScale">The text-to-image model's guidance scale.</param>
    /// <param name="seed">Seeds the image, the views and the sampling.</param>
    /// <returns><c>[1, N, 6]</c>: x, y, z, r, g, b.</returns>
    /// <exception cref="InvalidOperationException">No text-to-image model was supplied, or the mesh has no surface.</exception>
    public override Tensor<T> GeneratePointCloud(string prompt, string? negativePrompt = null,
        int? numPoints = null, int numInferenceSteps = 50, double guidanceScale = 3.0, int? seed = null)
    {
        var mesh = GenerateMesh(prompt, negativePrompt, _pipelineOptions.MeshResolution, numInferenceSteps, guidanceScale, seed);
        return SampleSurface(mesh, numPoints ?? DefaultPointCount, seed);
    }

    private Tensor<T> RenderPrompt(string prompt, string? negativePrompt, int numInferenceSteps, double guidanceScale, int? seed)
    {
        if (prompt is null) throw new ArgumentNullException(nameof(prompt));
        if (_textToImage is null)
            throw new InvalidOperationException(
                "One-2-3-45 reconstructs from an image. Text-to-3D renders the prompt with a text-to-image model first: " +
                "pass one as the textToImage constructor argument, or call ImageTo3D with an image.");
        int size = _pipelineOptions.TextToImageSize;
        return _textToImage.GenerateFromText(prompt, negativePrompt, size, size, numInferenceSteps, guidanceScale, seed);
    }

    #endregion

    #region Pipeline

    /// <summary>Runs the full pipeline and keeps the intermediate results.</summary>
    internal One2345Result<T> Reconstruct(Tensor<T> inputImage, int stage1Steps, int stage2Steps, double guidanceScale,
        int meshResolution, int? seed)
    {
        if (inputImage is null) throw new ArgumentNullException(nameof(inputImage));
        if (meshResolution < 2) throw new ArgumentOutOfRangeException(nameof(meshResolution));
        var o = _pipelineOptions;
        var input = ToBatch(inputImage);
        int height = input.Shape[2], width = input.Shape[3];
        if (height % 8 != 0 || width % 8 != 0)
            throw new ArgumentException("The image height and width must be multiples of 8.", nameof(inputImage));

        // Stage 1, views 0-3 at the input's elevation; the polar angle is not known yet and does not change them.
        var stage1Images = new Tensor<T>[8];
        var provisional = One2345CameraLayout.Stage1(o, o.FallbackPolar);
        for (int i = 0; i < 4; i++)
            stage1Images[i] = PredictView(input, provisional[i], stage1Steps, guidanceScale, ViewSeed(seed, i));

        // Stage 2 of view 0, then the elevation from its four neighbours.
        var stage2Images = new Tensor<T>[8][];
        stage2Images[0] = PredictNeighbours(stage1Images[0], provisional[0], 0, stage2Steps, guidanceScale, seed);
        double polar = EstimatePolar(stage2Images[0], width, height);

        // Stage 1, views 4-7 on the ring the estimate selects; stage 2 of views 1-7.
        var stage1 = One2345CameraLayout.Stage1(o, polar);
        for (int i = 4; i < 8; i++)
            stage1Images[i] = PredictView(input, stage1[i], stage1Steps, guidanceScale, ViewSeed(seed, i));
        for (int i = 1; i < 8; i++)
            stage2Images[i] = PredictNeighbours(stage1Images[i], stage1[i], i, stage2Steps, guidanceScale, seed);

        // Cameras: the reference (first-stage view 0, at the input's pose) and the 32 source views.
        var reference = One2345CameraLayout.Camera(o, stage1[0], width, height);
        var cameras = new List<NeuSCamera> { reference };
        var sourceImages = new List<Tensor<T>>();
        for (int i = 0; i < 8; i++)
        {
            var neighbours = One2345CameraLayout.Stage2(o, stage1[i]);
            for (int j = 0; j < 4; j++)
            {
                cameras.Add(One2345CameraLayout.Camera(o, neighbours[j], width, height));
                sourceImages.Add(stage2Images[i][j]);
            }
        }

        var frame = One2345CameraLayout.Normalize(o, reference, cameras, width, height);
        var views = new List<NeuSView<T>>();
        for (int k = 0; k < sourceImages.Count; k++)
            views.Add(new NeuSView<T>(ToImage(ToUnit(sourceImages[k]), height, width), frame.Cameras[k + 1]));

        var scene = _reconstruction.Reconstruct(views);
        var bounds = SdfGridBounds.UnitCube;
        var grid = scene.SampleGrid(meshResolution, bounds);
        var normalizedMesh = MarchingCubes.Extract(grid, 0.0, IsoSurfaceInterior.BelowIsoLevel,
            bounds.Min[0], bounds.Min[1], bounds.Min[2], bounds.Spacing(meshResolution, 0));
        var mesh = ToObjectFrame(normalizedMesh, scene, frame);
        return new One2345Result<T>(polar, stage1, stage1Images, stage2Images, views, frame, grid, mesh);
    }

    private static int? ViewSeed(int? seed, int index) => seed.HasValue ? unchecked(seed.Value + index) : (int?)null;

    private Tensor<T> PredictView(Tensor<T> source, One2345ViewPose pose, int steps, double guidanceScale, int? seed)
    {
        var view = _zero123.GenerateNovelView(source, pose.PolarDelta, pose.AzimuthDelta, 1.0, steps, guidanceScale, seed);
        return ToBatch(view);
    }

    /// <summary><c>infer_stage_2</c>: whiten the near-white pixels of a first-stage view, then predict its 4 neighbours.</summary>
    private Tensor<T>[] PredictNeighbours(Tensor<T> parentImage, One2345ViewPose parent, int parentIndex, int steps,
        double guidanceScale, int? seed)
    {
        var e = Engine;
        var unit = ToUnit(parentImage);
        var whiteMask = e.TensorGreaterThan(unit, NumOps.FromDouble(_pipelineOptions.WhiteCleanupThreshold - 0.5 / 255.0));
        var cleaned = e.TensorWhere(whiteMask, Tensor<T>.CreateDefault(unit.Shape.ToArray(), NumOps.One), unit);
        var source = e.TensorAddScalar(e.TensorMultiplyScalar(cleaned, NumOps.FromDouble(2.0)), NumOps.FromDouble(-1.0));
        var poses = One2345CameraLayout.Stage2(_pipelineOptions, parent);
        var result = new Tensor<T>[4];
        for (int j = 0; j < 4; j++)
            result[j] = PredictView(source, poses[j], steps, guidanceScale, ViewSeed(seed, 8 + 4 * parentIndex + j));
        return result;
    }

    /// <summary>
    /// <c>estimate_elev</c>: match the 4 nearby views pairwise, drop matches on the background (<c>mask_out_bkgd</c>),
    /// and search the polar angle.
    /// </summary>
    private double EstimatePolar(Tensor<T>[] neighbours, int width, int height)
    {
        var images = neighbours.Select(n => ToImage(ToUnit(n), height, width)).ToArray();
        var masks = images.Select(Foreground).ToArray();
        var matches = new Dictionary<(int, int), IReadOnlyList<FeatureCorrespondence>>();
        for (int i = 0; i < 4; i++)
            for (int j = i + 1; j < 4; j++)
            {
                var found = _featureMatcher.Match(images[i], images[j]);
                matches[(i, j)] = found.Where(m => IsForeground(masks[i], m.X0, m.Y0, width, height)
                    && IsForeground(masks[j], m.X1, m.Y1, width, height)).ToList();
            }

        double focal = _pipelineOptions.FocalLengthRatio * width;
        return One2345ElevationEstimator.Estimate(_pipelineOptions, matches, focal, width / 2.0, height / 2.0);
    }

    private bool[] Foreground(Tensor<T> image)
    {
        var values = NeuSHost<T>.ToDoubles(image);
        var mask = new bool[values.Length / 3];
        double threshold = _pipelineOptions.BackgroundThreshold;
        for (int p = 0; p < mask.Length; p++)
            mask[p] = !(values[p * 3] > threshold && values[p * 3 + 1] > threshold && values[p * 3 + 2] > threshold);
        return mask;
    }

    private static bool IsForeground(bool[] mask, double x, double y, int width, int height)
    {
        int ix = (int)x, iy = (int)y;
        return ix >= 0 && iy >= 0 && ix < width && iy < height && mask[iy * width + ix];
    }

    /// <summary>Colours the normalized mesh with the rendering network, then maps it to the object frame.</summary>
    private Mesh3D<T> ToObjectFrame(Mesh3D<T> mesh, SparseNeuSScene<T> scene, One2345SceneFrame frame)
    {
        int count = mesh.Vertices.Shape[0];
        if (count == 0)
        {
            mesh.Colors = new Tensor<T>(new[] { 0, 3 });
            return mesh;
        }

        var normalized = NeuSHost<T>.ToDoubles(mesh.Vertices);
        mesh.Colors = scene.PointColors(normalized, frame.QueryCenter);
        var world = new double[normalized.Length];
        var point = new double[3];
        for (int v = 0; v < count; v++)
        {
            Array.Copy(normalized, v * 3, point, 0, 3);
            Array.Copy(frame.ToWorld(point), 0, world, v * 3, 3);
        }

        mesh.Vertices = NeuSHost<T>.FromDoubles(world, count, 3);
        if (mesh.Normals is not null && mesh.Normals.Shape[0] == count)
        {
            var normals = NeuSHost<T>.ToDoubles(mesh.Normals);
            var rotated = new double[normals.Length];
            for (int v = 0; v < count; v++)
            {
                Array.Copy(normals, v * 3, point, 0, 3);
                Array.Copy(One2345CameraLayout.Rotate(frame.ReferenceRotation, point), 0, rotated, v * 3, 3);
            }

            mesh.Normals = NeuSHost<T>.FromDoubles(rotated, count, 3);
        }

        return mesh;
    }

    /// <summary>Area-weighted uniform surface samples with barycentric vertex colours, <c>[1, N, 6]</c>.</summary>
    internal static Tensor<T> SampleSurface(Mesh3D<T> mesh, int numPoints, int? seed)
    {
        if (numPoints <= 0) throw new ArgumentOutOfRangeException(nameof(numPoints));
        int faces = mesh.Faces.GetLength(0);
        if (faces == 0)
            throw new InvalidOperationException("The reconstructed SDF has no zero crossing in [-1, 1]^3, so there is no surface to sample.");
        var vertices = NeuSHost<T>.ToDoubles(mesh.Vertices);
        var colors = mesh.Colors is null ? null : NeuSHost<T>.ToDoubles(mesh.Colors);
        var cumulative = new double[faces];
        double total = 0;
        for (int f = 0; f < faces; f++)
        {
            int a = mesh.Faces[f, 0] * 3, b = mesh.Faces[f, 1] * 3, c = mesh.Faces[f, 2] * 3;
            double ux = vertices[b] - vertices[a], uy = vertices[b + 1] - vertices[a + 1], uz = vertices[b + 2] - vertices[a + 2];
            double vx = vertices[c] - vertices[a], vy = vertices[c + 1] - vertices[a + 1], vz = vertices[c + 2] - vertices[a + 2];
            double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
            total += 0.5 * Math.Sqrt(cx * cx + cy * cy + cz * cz);
            cumulative[f] = total;
        }

        if (!(total > 0)) throw new InvalidOperationException("The reconstructed surface has zero area.");
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var cloud = new double[numPoints * 6];
        for (int p = 0; p < numPoints; p++)
        {
            double target = random.NextDouble() * total;
            int lo = 0, hi = faces - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (cumulative[mid] < target) lo = mid + 1;
                else hi = mid;
            }

            double r1 = Math.Sqrt(random.NextDouble()), r2 = random.NextDouble();
            double wa = 1 - r1, wb = r1 * (1 - r2), wc = r1 * r2;
            int ia = mesh.Faces[lo, 0] * 3, ib = mesh.Faces[lo, 1] * 3, ic = mesh.Faces[lo, 2] * 3;
            for (int k = 0; k < 3; k++)
            {
                cloud[p * 6 + k] = wa * vertices[ia + k] + wb * vertices[ib + k] + wc * vertices[ic + k];
                cloud[p * 6 + 3 + k] = colors is null ? 0.0 : wa * colors[ia + k] + wb * colors[ib + k] + wc * colors[ic + k];
            }
        }

        return NeuSHost<T>.FromDoubles(cloud, 1, numPoints, 6);
    }

    private static Tensor<T> ToBatch(Tensor<T> image)
    {
        if (image.Shape.Length == 3 && image.Shape[0] == 3)
            return image.Reshape(1, 3, image.Shape[1], image.Shape[2]);
        if (image.Shape.Length != 4 || image.Shape[0] != 1 || image.Shape[1] != 3)
            throw new ArgumentException($"The image must be [1, 3, H, W] or [3, H, W]; got [{string.Join(", ", image.Shape.ToArray())}].",
                nameof(image));
        return image;
    }

    /// <summary>[−1, 1] to [0, 1], clamped (<c>torch.clamp((x + 1) / 2, 0, 1)</c>).</summary>
    private Tensor<T> ToUnit(Tensor<T> image)
    {
        var e = Engine;
        var shifted = e.TensorMultiplyScalar(e.TensorAddScalar(image, NumOps.One), NumOps.FromDouble(0.5));
        return e.TensorClamp(shifted, NumOps.Zero, NumOps.One);
    }

    /// <summary><c>[1, 3, H, W]</c> to <c>[H, W, 3]</c>, copied out of the engine's buffers because the views are
    /// held while many later operations run.</summary>
    private Tensor<T> ToImage(Tensor<T> batch, int height, int width)
    {
        var e = Engine;
        var permuted = e.TensorPermute(e.Reshape(batch, new[] { 3, height, width }), new[] { 1, 2, 0 });
        return permuted.Clone();
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = _pipelineOptions;
        var metadata = new ModelMetadata<T>
        {
            Name = "One-2-3-45", Version = "1.0",
            Description = "One-2-3-45: Zero123 multi-view prediction, elevation estimation and SparseNeuS reconstruction",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount), Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "zero123-plus-sparseneus");
        metadata.SetProperty("source_views", 32);
        metadata.SetProperty("stage1_steps", o.Stage1Steps);
        metadata.SetProperty("stage2_steps", o.Stage2Steps);
        metadata.SetProperty("volume_resolution", o.VolumeResolution);
        metadata.SetProperty("mesh_resolution", o.MeshResolution);
        metadata.SetProperty("reconstruction", "SparseNeuS");
        metadata.SetProperty("viewpoint_diffusion", "Zero123");
        metadata.SetProperty("scheduler", "DDIM");
        metadata.SetProperty("default_points", DefaultPointCount);
        return metadata;
    }

    #endregion
}

/// <summary>The intermediate and final results of one One-2-3-45 run.</summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class One2345Result<T>
{
    public One2345Result(double polar, One2345ViewPose[] stage1Poses, Tensor<T>[] stage1Images, Tensor<T>[][] stage2Images,
        IReadOnlyList<NeuSView<T>> sourceViews, One2345SceneFrame frame, Tensor<T> sdfGrid, Mesh3D<T> mesh)
    {
        Polar = polar;
        Stage1Poses = stage1Poses;
        Stage1Images = stage1Images;
        Stage2Images = stage2Images;
        SourceViews = sourceViews;
        Frame = frame;
        SdfGrid = sdfGrid;
        Mesh = mesh;
    }

    /// <summary>The estimated polar angle of the input, in degrees (elevation is 90 minus this).</summary>
    public double Polar { get; }

    public One2345ViewPose[] Stage1Poses { get; }

    public Tensor<T>[] Stage1Images { get; }

    public Tensor<T>[][] Stage2Images { get; }

    public IReadOnlyList<NeuSView<T>> SourceViews { get; }

    public One2345SceneFrame Frame { get; }

    public Tensor<T> SdfGrid { get; }

    public Mesh3D<T> Mesh { get; }
}

/// <summary>Exposes the SparseNeuS weights (and batch-norm statistics) as one flat parameter vector.</summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class SparseNeuSParameterSource<T> : IParameterSource<T>
{
    private readonly SparseNeuS<T> _network;

    public SparseNeuSParameterSource(SparseNeuS<T> network)
    {
        _network = network ?? throw new ArgumentNullException(nameof(network));
    }

    public long ParameterCount => _network.NamedParameters().Sum(p => (long)p.Value.Length);

    public Vector<T> GetParameters()
    {
        var values = new List<T>();
        foreach (var entry in _network.NamedParameters()) values.AddRange(entry.Value.ToVector().ToArray());
        return new Vector<T>(values.ToArray());
    }

    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} SparseNeuS parameters; got {parameters.Length}.", nameof(parameters));
        var flat = parameters.ToArray();
        var loaded = new Dictionary<string, Tensor<T>>();
        int offset = 0;
        foreach (var entry in _network.NamedParameters().ToList())
        {
            int length = entry.Value.Length;
            var slice = new T[length];
            Array.Copy(flat, offset, slice, 0, length);
            loaded[entry.Key] = new Tensor<T>(entry.Value.Shape.ToArray(), new Vector<T>(slice));
            offset += length;
        }

        _network.LoadParameters(loaded);
    }
}
