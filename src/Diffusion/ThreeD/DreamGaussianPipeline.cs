using AiDotNet.Diffusion.ThreeD.Meshing;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// The outputs of a DreamGaussian run: the optimised Gaussians, their coloured centres, the extracted mesh and, when
/// the texture stage ran, the UV-textured mesh.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class DreamGaussianResult<T>
{
    internal DreamGaussianResult(
        GaussianSplatting<T> gaussians, Tensor<T> pointCloud, Mesh3D<T> mesh, Mesh3D<T>? texturedMesh,
        IReadOnlyList<double> referenceLossHistory)
    {
        Gaussians = gaussians;
        PointCloud = pointCloud;
        Mesh = mesh;
        TexturedMesh = texturedMesh;
        ReferenceLossHistory = referenceLossHistory;
    }

    /// <summary>The stage-1 Gaussians.</summary>
    public GaussianSplatting<T> Gaussians { get; }

    /// <summary>The Gaussian centres with their colours, [1, N, 6] (x, y, z, r, g, b).</summary>
    public Tensor<T> PointCloud { get; }

    /// <summary>The mesh extracted from the Gaussians' density (vertices, faces, normals).</summary>
    public Mesh3D<T> Mesh { get; }

    /// <summary>
    /// The mesh with its UV atlas and the refined texture (<see cref="Mesh3D{T}.UVCoordinates"/> [V, 2],
    /// <see cref="Mesh3D{T}.TextureImage"/> [S, S, 3]); null when the texture stage was off or the mesh is empty.
    /// </summary>
    public Mesh3D<T>? TexturedMesh { get; }

    /// <summary>
    /// Stage 1's reference-view loss per step, <c>w_rgb mse(rgb) + w_mask mse(alpha)</c> before the warm-up factor
    /// (image-to-3D only; empty for text-to-3D).
    /// </summary>
    public IReadOnlyList<double> ReferenceLossHistory { get; }

    /// <summary>The textured mesh when there is one, otherwise the extracted mesh.</summary>
    public Mesh3D<T> FinalMesh => TexturedMesh ?? Mesh;
}

/// <summary>
/// The DreamGaussian optimisation (Tang et al., ICLR 2024): stage 1 (<c>main.py</c>) fits 3D Gaussians to a guidance
/// prior with SDS, plus the reference-view losses in image-to-3D; the mesh is extracted from their density; stage 2
/// (<c>main2.py</c>) bakes the Gaussians' colours into a UV texture and refines it against SDEdit-denoised renders.
/// </summary>
internal sealed class DreamGaussianPipeline<T>
{
    /// <summary>The opacity ceiling of the reference's <c>reset_opacity</c>.</summary>
    private const double OpacityResetCeiling = 0.01;

    private readonly DreamGaussianOptions _settings;
    private readonly DreamGaussianMode _mode;
    private readonly IScoreDistillationGuidance<T> _guidance;
    private readonly double[]? _referenceImage;
    private readonly double[]? _referenceMask;
    private readonly Random _random;
    private readonly List<double> _referenceLosses = new();

    /// <param name="settings">The settings.</param>
    /// <param name="mode">Image- or text-to-3D.</param>
    /// <param name="guidance">The 2D prior.</param>
    /// <param name="referenceImage">Image-to-3D: the input on white, [S, S, 3] with S = <see cref="DreamGaussianOptions.ReferenceSize"/>.</param>
    /// <param name="referenceMask">Image-to-3D: its foreground mask, [S, S].</param>
    /// <param name="random">The single random source of the run.</param>
    public DreamGaussianPipeline(
        DreamGaussianOptions settings, DreamGaussianMode mode, IScoreDistillationGuidance<T> guidance,
        double[]? referenceImage, double[]? referenceMask, Random random)
    {
        _settings = settings ?? throw new ArgumentNullException(nameof(settings));
        _guidance = guidance ?? throw new ArgumentNullException(nameof(guidance));
        _random = random ?? throw new ArgumentNullException(nameof(random));
        _mode = mode;
        settings.Validate();
        if (mode == DreamGaussianMode.ImageTo3D)
        {
            int pixels = settings.ReferenceSize * settings.ReferenceSize;
            if (referenceImage is null || referenceImage.Length != 3 * pixels)
                throw new ArgumentException($"Image-to-3D needs a reference image of {3 * pixels} values.", nameof(referenceImage));
            if (referenceMask is null || referenceMask.Length != pixels)
                throw new ArgumentException($"Image-to-3D needs a reference mask of {pixels} values.", nameof(referenceMask));
        }

        _referenceImage = referenceImage;
        _referenceMask = referenceMask;
    }

    /// <summary>The reference-view loss of every stage-1 step so far.</summary>
    public IReadOnlyList<double> ReferenceLossHistory => _referenceLosses;

    /// <summary>Runs both stages.</summary>
    public DreamGaussianResult<T> Run()
    {
        var scene = TrainGaussians();
        var centers = scene.ExportColoredCenters();
        var pointCloud = new Tensor<T>(new[] { 1, centers.Shape[0], 6 }, centers.ToVector());
        var mesh = GaussianMeshExtractor.ExtractMesh(scene.ToGaussianCloud(), _settings.MeshExtraction);
        Mesh3D<T>? textured = _settings.RefineTexture ? TextureMesh(scene, mesh) : null;
        return new DreamGaussianResult<T>(scene, pointCloud, mesh, textured, _referenceLosses.ToArray());
    }

    /// <summary>Runs stage 1 only; the result's meshes are empty.</summary>
    public DreamGaussianResult<T> RunGaussiansOnly()
    {
        var scene = TrainGaussians();
        var centers = scene.ExportColoredCenters();
        var pointCloud = new Tensor<T>(new[] { 1, centers.Shape[0], 6 }, centers.ToVector());
        return new DreamGaussianResult<T>(scene, pointCloud, new Mesh3D<T>(), null, _referenceLosses.ToArray());
    }

    // ---- Stage 1 ---------------------------------------------------------------------------------------------------

    /// <summary>The reference's stage-1 loop (<c>main.py train_step</c>, <see cref="DreamGaussianOptions.Iterations"/> times).</summary>
    public GaussianSplatting<T> TrainGaussians()
    {
        var s = _settings;
        // sh_degree 0: the colour parameter is the DC coefficient, so the feature learning rate is in the
        // reference's units (a raw-RGB parameter would move C0 times faster per step).
        var scene = GaussianSplatting<T>.CreateInSphere(
            new GaussianSplattingOptions { UseSphericalHarmonics = true, ShDegree = 0 },
            s.InitialGaussianCount, s.InitialRadius, s.InitialOpacity, _random);

        var adam = new GaussianAdamSettings
        {
            PositionLearningRateInit = s.PositionLearningRateInit,
            PositionLearningRateFinal = s.PositionLearningRateFinal,
            PositionLearningRateDelayMultiplier = s.PositionLearningRateDelayMultiplier,
            PositionLearningRateMaxSteps = s.PositionLearningRateMaxSteps(_mode),
            FeatureLearningRate = s.FeatureLearningRate,
            OpacityLearningRate = s.OpacityLearningRate,
            ScalingLearningRate = s.ScalingLearningRate,
            RotationLearningRate = s.RotationLearningRate,
        };
        var densification = new GaussianDensificationSettings
        {
            GradientThreshold = s.DensifyGradientThreshold,
            MinOpacity = s.PruneMinOpacity,
            SceneExtent = s.SceneExtent,
            MaxScreenSize = s.MaxScreenSize,
            PercentDense = s.PercentDense,
        };

        double radius = s.CameraRadius(_mode);
        double elevation = s.ReferenceElevation;
        var referenceCamera = _mode == DreamGaussianMode.ImageTo3D
            ? GaussianSplatCamera.FromOrbit(elevation, 0.0, radius, s.ReferenceSize, s.ReferenceSize, s.FovYDegrees)
            : null;
        // Avoid elevations beyond +-80 degrees while always covering [min_ver, max_ver].
        double minVer = Math.Max(Math.Min(s.MinElevation, s.MinElevation - elevation), -80.0 - elevation);
        double maxVer = Math.Min(Math.Max(s.MaxElevation, s.MaxElevation - elevation), 80.0 - elevation);
        int densifyStart = s.DensifyStartIteration(_mode);
        int densifyInterval = s.DensificationInterval(_mode);
        double guidanceWeight = s.GuidanceWeight(_mode);

        for (int step = 1; step <= s.Iterations; step++)
        {
            double ratio = Math.Min(1.0, (double)step / s.Iterations);
            var gradients = new GaussianSplatGradients(scene.GenerativeCount);

            if (referenceCamera is not null)
                gradients.Add(ReferenceViewGradients(scene, referenceCamera, ratio));

            int resolution = s.RenderResolution(ratio);
            var renders = new GaussianSplatRender[s.BatchSize];
            var views = new GuidanceViewOffset[s.BatchSize];
            for (int b = 0; b < s.BatchSize; b++)
            {
                int ver = RandomInteger(minVer, maxVer);
                int hor = _random.Next(-180, 180);
                views[b] = new GuidanceViewOffset(ver, hor, 0.0);
                var camera = GaussianSplatCamera.FromOrbit(elevation + ver, hor, radius, resolution, resolution, s.FovYDegrees);
                double shade = _random.NextDouble() > s.InvertBackgroundProbability ? 1.0 : 0.0;
                renders[b] = scene.RenderDifferentiable(camera, new[] { shade, shade, shade });
            }

            GaussianSplatGradients? lastViewGradients = null;
            if (guidanceWeight != 0.0)
            {
                var images = ToBatch(renders.Select(r => r.Image).ToArray(), resolution);
                int timestep = s.AnnealTimestep
                    ? LatentScoreDistillation<T>.AnnealedTimestep(ratio, _guidance.TrainTimesteps, _guidance.MinStep, _guidance.MaxStep)
                    : _random.Next(_guidance.MinStep, _guidance.MaxStep + 1);
                var imageGradients = _guidance.ComputeImageGradient(images, views, timestep, _random);
                for (int b = 0; b < s.BatchSize; b++)
                {
                    var hwc = FromBatch(imageGradients, b, guidanceWeight);
                    var viewGradients = scene.BackwardDifferentiable(renders[b], hwc, null);
                    gradients.Add(viewGradients);
                    lastViewGradients = viewGradients;
                }
            }

            scene.GenerativeAdamStep(gradients, adam, step);

            if (step >= densifyStart && step <= s.DensifyEndIteration)
            {
                // The statistics come from the last novel view (the reference's `out`), whose centres received only
                // the guidance gradient.
                scene.AccumulateDensificationStatistics(
                    renders[s.BatchSize - 1], lastViewGradients ?? new GaussianSplatGradients(renders[s.BatchSize - 1].GaussianCount));
                if (densifyInterval > 0 && step % densifyInterval == 0)
                    scene.DensifyAndPruneGenerative(densification, _random);
                if (s.OpacityResetInterval > 0 && step % s.OpacityResetInterval == 0)
                    scene.ResetGenerativeOpacity(OpacityResetCeiling);
            }
        }

        return scene;
    }

    /// <summary>
    /// The reference-view losses <c>10000 r mse(image, ref) + 1000 r mse(alpha, mask)</c> (r the warm-up ratio) and
    /// their Gaussian gradients.
    /// </summary>
    internal GaussianSplatGradients ReferenceViewGradients(GaussianSplatting<T> scene, GaussianSplatCamera camera, double ratio)
    {
        var s = _settings;
        var image = _referenceImage ?? throw new InvalidOperationException("No reference image.");
        var mask = _referenceMask ?? throw new InvalidOperationException("No reference mask.");
        var render = scene.RenderDifferentiable(camera, new[] { 1.0, 1.0, 1.0 });
        int pixels = render.Width * render.Height;
        double warmup = s.WarmupReferenceLoss ? ratio : 1.0;
        var imageGradient = new double[3 * pixels];
        var alphaGradient = new double[pixels];
        double rgbError = 0.0, maskError = 0.0;
        for (int i = 0; i < 3 * pixels; i++)
        {
            double d = render.Image[i] - image[i];
            rgbError += d * d;
            imageGradient[i] = warmup * s.ReferenceRgbLossWeight * 2.0 * d / (3.0 * pixels);
        }

        for (int p = 0; p < pixels; p++)
        {
            double d = render.Alpha[p] - mask[p];
            maskError += d * d;
            alphaGradient[p] = warmup * s.ReferenceMaskLossWeight * 2.0 * d / pixels;
        }

        _referenceLosses.Add((s.ReferenceRgbLossWeight * rgbError / (3.0 * pixels)) + (s.ReferenceMaskLossWeight * maskError / pixels));
        return scene.BackwardDifferentiable(render, imageGradient, alphaGradient);
    }

    // ---- Stage 2 ---------------------------------------------------------------------------------------------------

    /// <summary>
    /// Unwraps the mesh, bakes the Gaussians into its texture (<c>save_model('geo+tex')</c>) and refines the texture
    /// (<c>main2.py</c>, <see cref="DreamGaussianOptions.RefineIterations"/> steps).
    /// </summary>
    /// <returns>The textured mesh, or null for an empty mesh.</returns>
    public Mesh3D<T>? TextureMesh(GaussianSplatting<T> scene, Mesh3D<T> mesh)
    {
        if (scene is null) throw new ArgumentNullException(nameof(scene));
        if (mesh is null) throw new ArgumentNullException(nameof(mesh));
        var s = _settings;
        int faceCount = mesh.Faces.GetLength(0);
        if (faceCount == 0) return null;
        var vertices = mesh.Vertices.Cast<double>().ToArray();
        var faces = new int[3 * faceCount];
        for (int f = 0; f < faceCount; f++)
        {
            for (int k = 0; k < 3; k++)
                faces[(3 * f) + k] = mesh.Faces[f, k];
        }

        var geometry = DreamGaussianTexturing.Prepare(
            vertices, faces, new UvAtlasOptions { TextureResolution = s.TextureSize, PaddingTexels = s.UvPaddingTexels });
        double radius = s.CameraRadius(_mode);

        // Bake: render the Gaussians (white background) from the bake views and back-project.
        var bakeViews = new List<(MeshTextureView View, double[] Image)>();
        for (int v = 0; v < s.BakeElevations.Length; v++)
        {
            var camera = GaussianSplatCamera.FromOrbit(
                s.BakeElevations[v], s.BakeAzimuths[v], radius, s.BakeRenderResolution, s.BakeRenderResolution, s.FovYDegrees);
            var render = scene.RenderDifferentiable(camera, new[] { 1.0, 1.0, 1.0 });
            bakeViews.Add((DreamGaussianTexturing.SampleView(geometry, camera), render.Image));
        }

        var (albedo, baked) = DreamGaussianTexturing.Bake(bakeViews, s.TextureSize, s.BakeMinResolution, s.ViewCosineThreshold);
        albedo = DreamGaussianTexturing.Inpaint(albedo, baked, s.TextureSize, s.InpaintDilation, s.InpaintSearchErosion);
        var raw = RefineTexture(geometry, DreamGaussianTexturing.TruncatedInverseSigmoid(albedo));
        return BuildTexturedMesh(geometry, DreamGaussianTexturing.Sigmoid(raw));
    }

    /// <summary>The reference's stage-2 loop on the raw (pre-sigmoid) texture.</summary>
    internal double[] RefineTexture(UvTexturedMesh geometry, double[] raw)
    {
        var s = _settings;
        int size = s.TextureSize;
        double radius = s.CameraRadius(_mode);
        double elevation = s.ReferenceElevation;
        double minVer = Math.Max(Math.Min(s.MinElevation, s.MinElevation - elevation), -80.0 - elevation);
        double maxVer = Math.Min(Math.Max(s.MaxElevation, s.MaxElevation - elevation), 80.0 - elevation);
        double guidanceWeight = s.GuidanceWeight(_mode);
        var firstMoment = new double[raw.Length];
        var secondMoment = new double[raw.Length];

        for (int step = 1; step <= s.RefineIterations; step++)
        {
            double ratio = Math.Min(1.0, (double)step / s.RefineIterations);
            var gradient = new double[raw.Length];

            if (_mode == DreamGaussianMode.ImageTo3D && _referenceImage is not null)
            {
                // A random super-sampling per render; mse(image * valid, ref * valid) with valid = alpha > 0 and
                // viewcos > threshold, both resized with the image.
                var referenceView = DreamGaussianTexturing.SampleSupersampledView(
                    geometry, (w, h) => GaussianSplatCamera.FromOrbit(elevation, 0.0, radius, w, h, s.FovYDegrees),
                    s.ReferenceSize, s.ReferenceSize, DreamGaussianTexturing.DrawSupersampling(_random));
                var image = DreamGaussianTexturing.Render(referenceView, raw, size, 1.0);
                int pixels = referenceView.Width * referenceView.Height;
                var imageGradient = new double[3 * pixels];
                for (int p = 0; p < pixels; p++)
                {
                    if (!(referenceView.Alpha[p] > 0.0) || !(referenceView.ViewCosine[p] > s.ViewCosineThreshold)) continue;
                    for (int c = 0; c < 3; c++)
                        imageGradient[(3 * p) + c] = 2.0 * (image[(3 * p) + c] - _referenceImage[(3 * p) + c]) / (3.0 * pixels);
                }

                DreamGaussianTexturing.Backward(referenceView, raw, size, imageGradient, gradient);
            }

            int resolution = s.RefineRenderResolution;
            var views = new SupersampledTextureView[s.BatchSize];
            var offsets = new GuidanceViewOffset[s.BatchSize];
            var images = new double[s.BatchSize][];
            for (int b = 0; b < s.BatchSize; b++)
            {
                int ver = RandomInteger(minVer, maxVer);
                int hor = _random.Next(-180, 180);
                offsets[b] = new GuidanceViewOffset(ver, hor, 0.0);
                views[b] = DreamGaussianTexturing.SampleSupersampledView(
                    geometry, (w, h) => GaussianSplatCamera.FromOrbit(elevation + ver, hor, radius, w, h, s.FovYDegrees),
                    resolution, resolution, DreamGaussianTexturing.DrawSupersampling(_random));
                images[b] = DreamGaussianTexturing.Render(views[b], raw, size, 1.0);
            }

            if (guidanceWeight != 0.0)
            {
                double strength = s.RefineStrengthStart + (s.RefineStrengthRange * ratio);
                var refined = _guidance.Refine(ToBatch(images, resolution), offsets, strength, s.RefineDenoisingSteps, _random);
                refined = LatentScoreDistillation<T>.Resize(refined, resolution);
                double scale = guidanceWeight * 2.0 / (s.BatchSize * 3.0 * resolution * resolution);
                for (int b = 0; b < s.BatchSize; b++)
                {
                    var target = FromBatch(refined, b, 1.0);
                    var imageGradient = new double[images[b].Length];
                    for (int i = 0; i < imageGradient.Length; i++)
                        imageGradient[i] = scale * (images[b][i] - target[i]);
                    DreamGaussianTexturing.Backward(views[b], raw, size, imageGradient, gradient);
                }
            }

            // torch.optim.Adam on raw_albedo.
            double bias1 = 1.0 - Math.Pow(s.TextureAdamBeta1, step);
            double bias2 = 1.0 - Math.Pow(s.TextureAdamBeta2, step);
            for (int i = 0; i < raw.Length; i++)
            {
                double g = gradient[i];
                firstMoment[i] = (s.TextureAdamBeta1 * firstMoment[i]) + ((1.0 - s.TextureAdamBeta1) * g);
                secondMoment[i] = (s.TextureAdamBeta2 * secondMoment[i]) + ((1.0 - s.TextureAdamBeta2) * g * g);
                double denominator = (Math.Sqrt(secondMoment[i]) / Math.Sqrt(bias2)) + s.TextureAdamEpsilon;
                raw[i] -= s.TextureLearningRate / bias1 * firstMoment[i] / denominator;
            }
        }

        return raw;
    }

    private static Mesh3D<T> BuildTexturedMesh(UvTexturedMesh geometry, double[] albedo)
    {
        var atlas = geometry.Atlas;
        int vertexCount = atlas.SourceVertex.Length;
        int faceCount = atlas.Faces.Length / 3;
        var faces = new int[faceCount, 3];
        for (int f = 0; f < faceCount; f++)
        {
            for (int k = 0; k < 3; k++)
                faces[f, k] = atlas.Faces[(3 * f) + k];
        }

        int size = (int)Math.Round(Math.Sqrt(albedo.Length / 3.0));
        return new Mesh3D<T>(ToTensor(atlas.Vertices, vertexCount, 3), faces)
        {
            Normals = ToTensor(geometry.Normals, vertexCount, 3),
            UVCoordinates = ToTensor(atlas.Uvs, vertexCount, 2),
            TextureImage = new Tensor<double>(new[] { size, size, 3 }, new Vector<double>(albedo)).Cast<T>(),
        };
    }

    // ---- Helpers ---------------------------------------------------------------------------------------------------

    /// <summary><c>np.random.randint(low, high)</c>: an integer in [low, high).</summary>
    private int RandomInteger(double low, double high)
    {
        int lo = (int)low, hi = (int)high;
        return hi > lo ? _random.Next(lo, hi) : lo;
    }

    private static Tensor<T> ToTensor(double[] values, int rows, int columns)
        => new Tensor<double>(new[] { rows, columns }, new Vector<double>(values)).Cast<T>();

    /// <summary>Stacks [H, W, 3] images into [B, 3, H, W].</summary>
    internal static Tensor<T> ToBatch(IReadOnlyList<double[]> images, int resolution)
    {
        int pixels = resolution * resolution;
        var data = new double[images.Count * 3 * pixels];
        for (int b = 0; b < images.Count; b++)
        {
            var image = images[b];
            for (int p = 0; p < pixels; p++)
            {
                for (int c = 0; c < 3; c++)
                    data[(((b * 3) + c) * pixels) + p] = image[(3 * p) + c];
            }
        }

        return new Tensor<double>(new[] { images.Count, 3, resolution, resolution }, new Vector<double>(data)).Cast<T>();
    }

    /// <summary>Image <paramref name="index"/> of a [B, 3, H, W] batch as [H, W, 3], times <paramref name="scale"/>.</summary>
    internal static double[] FromBatch(Tensor<T> batch, int index, double scale)
    {
        int height = batch.Shape[2], width = batch.Shape[3];
        int pixels = height * width;
        var data = batch.Cast<double>().ToArray();
        var image = new double[3 * pixels];
        int offset = index * 3 * pixels;
        for (int p = 0; p < pixels; p++)
        {
            for (int c = 0; c < 3; c++)
                image[(3 * p) + c] = scale * data[offset + (c * pixels) + p];
        }

        return image;
    }
}
