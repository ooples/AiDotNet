using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.ThreeD.Meshing;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// LGM (Large Multi-View Gaussian Model): feed-forward reconstruction of 3D Gaussians from four generated views.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The pipeline of Tang et al. (ECCV 2024, arXiv 2402.05054) and the official 3DTopia/LGM code:
/// </para>
/// <list type="number">
/// <item><description><b>Four views.</b> A multi-view diffusion model (<see cref="MVDreamModel{T}"/>; ImageDream-style
/// image conditioning for image input, text conditioning for text input) generates the object from four orthogonal
/// cameras at azimuths 0, 90, 180 and 270 degrees.</description></item>
/// <item><description><b>Ray-augmented input.</b> Each view is resized to <see cref="LGMConfig.InputSize"/>,
/// normalized with the ImageNet statistics, and concatenated with its per-pixel Plucker ray embedding
/// <c>(o x d, d)</c>, 9 channels per view.</description></item>
/// <item><description><b>Asymmetric U-Net</b> (<see cref="LGMAsymmetricUNet{T}"/>) with multi-view self-attention
/// maps the four 9-channel views to four 14-channel splatter maps at half resolution (128 x 128 by default), followed
/// by a 1x1 convolution.</description></item>
/// <item><description><b>Gaussians.</b> Every splatter-map pixel is one Gaussian: position <c>clamp(x, -1, 1)</c>,
/// opacity <c>sigmoid</c>, scale <c>0.1 softplus</c>, rotation <c>normalize</c> (a unit quaternion) and RGB
/// <c>0.5 tanh + 0.5</c> (<see cref="LGMGaussians{T}"/>). They become a <see cref="GaussianSplatting{T}"/> scene through
/// <see cref="ToGaussianSplatting"/>, which renders them from any camera.</description></item>
/// </list>
/// <para>
/// <b>Weights.</b> The architecture is the paper's; meaningful reconstructions need the released LGM checkpoint, which
/// loads through <see cref="LGMAsymmetricUNet{T}.SetParameters"/> (the U-Net layers in <c>Layers</c> order) and
/// <see cref="GaussianProjection"/>, and a pretrained multi-view model. With freshly initialized weights the
/// outputs are well-formed Gaussians of the right shapes and ranges, but not the object.
/// </para>
/// <para>
/// <b>Meshes.</b> <see cref="GaussiansToMesh"/> (and through it <see cref="ImageTo3D"/> and
/// <see cref="GenerateMesh"/>) extracts the surface from the Gaussians' density field with
/// <see cref="GaussianMeshExtractor"/>: the density <c>sum_i alpha_i exp(-1/2 d^T Sigma_i^-1 d)</c> sampled on a grid
/// and its level set taken by marching cubes, as DreamGaussian does. The official LGM converter refines that further
/// (it distils the Gaussians into a NeRF and then fits and textures the mesh); those refinement stages are not part of
/// this model.
/// </para>
/// <para>
/// <b>For Beginners:</b> LGM turns one picture (or a sentence) into a 3D object in a single pass. First a diffusion
/// model imagines what the object looks like from the front, right, back and left. Then a network looks at those four
/// pictures together, along with where each pixel's camera ray points, and predicts tens of thousands of small
/// coloured 3D blobs that, drawn together, form the object.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var lgm = new LGMModel&lt;float&gt;();
/// // views: [4, 3, 256, 256] RGB in [0, 1] from azimuths 0, 90, 180, 270 at elevation 0
/// LGMGaussians&lt;float&gt; gaussians = lgm.ReconstructGaussians(views);
/// var scene = lgm.ToGaussianSplatting(gaussians);
/// Tensor&lt;float&gt; pointCloud = lgm.GaussiansToPointCloud(gaussians); // [1, M, 6] xyz + rgb
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ThreeDGeneration)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation", "https://arxiv.org/abs/2402.05054", Year = 2024, Authors = "Tang et al.")]
public partial class LGMModel<T> : ThreeDDiffusionModelBase<T>
{
    /// <summary>Input channels per view: RGB (3) plus the Plucker ray embedding (6).</summary>
    public const int InputChannelsPerView = 9;

    /// <summary>Gaussian parameters per splatter-map pixel: position 3, opacity 1, scale 3, rotation 4, RGB 3.</summary>
    public const int GaussianChannels = 14;

    private readonly LGMConfig _config;
    // The multi-view model is a collaborator with its own parameter surface and checkpoint. Its denoiser and VAE reach
    // this model's surface through the NoisePredictor and VAE slots; registering the whole model as well would count
    // them twice.
    [ExternalState]
    private readonly MVDreamModel<T> _multiViewModel;
    private readonly LGMAsymmetricUNet<T> _unet;
    private readonly ConvolutionalLayer<T> _gaussianProjection;

    /// <inheritdoc />
    /// <remarks>After the base's NoisePredictor and VAE slots (the multi-view model's denoiser and VAE) come the U-Net,
    /// its layers in forward order, then the 1x1 Gaussian projection.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_unet);
        RegisterParameterComponent(_gaussianProjection);
    }

    /// <summary>
    /// Creates an LGM model.
    /// </summary>
    /// <param name="config">Reconstructor configuration; null for the paper's "big" configuration.</param>
    /// <param name="multiViewModel">The multi-view diffusion model that generates the four input views; null for a
    /// default <see cref="MVDreamModel{T}"/>. Give it text and image conditioners for text and image input.</param>
    /// <param name="architecture">Optional architecture for the base model.</param>
    /// <param name="options">Diffusion options for the base model.</param>
    /// <param name="scheduler">Scheduler for the base model.</param>
    /// <param name="seed">Seed for the initial weights; null for a non-reproducible draw.</param>
    public LGMModel(
        LGMConfig? config = null,
        MVDreamModel<T>? multiViewModel = null,
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
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
            GaussianCountFor(config ?? LGMConfig.Default),
            architecture)
    {
        _config = config ?? LGMConfig.Default;
        _multiViewModel = multiViewModel ?? new MVDreamModel<T>(seed: seed);
        _unet = new LGMAsymmetricUNet<T>(_config, InputChannelsPerView, GaussianChannels, seed);
        // The U-Net reset the layer seed scope; the projection continues the same sequence.
        _gaussianProjection = ConvolutionalLayer<T>.WithInputDepth(
            inputDepth: GaussianChannels,
            outputDepth: GaussianChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>(),
            initializationStrategy: new TorchDefaultInitializationStrategy<T>(),
            biasMode: BiasMode.Always);
        _gaussianProjection.SetTrainingMode(false);
    }

    #region Properties

    /// <summary>The reconstructor configuration.</summary>
    public LGMConfig Config => _config;

    /// <summary>The multi-view diffusion model that generates the four input views. Its own parameter surface
    /// (its GetParameters) also holds its multi-view attention and camera embedding.</summary>
    [ExternalState]
    public MVDreamModel<T> MultiViewModel => _multiViewModel;

    /// <summary>The asymmetric U-Net that predicts the splatter maps.</summary>
    public LGMAsymmetricUNet<T> GaussianUNet => _unet;

    /// <summary>The 1x1 convolution after the U-Net (<c>self.conv</c> in the official model).</summary>
    public ConvolutionalLayer<T> GaussianProjection => _gaussianProjection;

    /// <summary>Side length of each view's splatter map (128 by default).</summary>
    public int SplatSize => _unet.OutputSize;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _multiViewModel.NoisePredictor;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _multiViewModel.VAE;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _multiViewModel.Conditioner;

    /// <inheritdoc />
    public override int LatentChannels => _multiViewModel.LatentChannels;

    /// <inheritdoc />
    public override bool SupportsPointCloud => true;

    /// <inheritdoc />
    public override bool SupportsMesh => true;

    /// <inheritdoc />
    public override bool SupportsTexture => false;

    /// <inheritdoc />
    /// <remarks>Novel views of an LGM result are renders of its Gaussians; use <see cref="ToGaussianSplatting"/>.</remarks>
    public override bool SupportsNovelView => false;

    /// <inheritdoc />
    public override bool SupportsScoreDistillation => false;

    #endregion

    #region Cameras and input

    /// <summary>
    /// The Plucker ray embedding of the input cameras, <c>[views, 6, InputSize, InputSize]</c>: channels 0-2 are the
    /// moment <c>o x d</c>, channels 3-5 the unit direction <c>d</c>.
    /// </summary>
    /// <param name="elevation">Camera elevation in degrees (kiui convention, negative looks down); null for
    /// <see cref="LGMConfig.InputElevation"/>.</param>
    public Tensor<T> ComputeRayEmbeddings(double? elevation = null)
    {
        int size = _config.InputSize;
        var rays = PluckerRays(_config.NumInputViews, size, _config.FieldOfViewY, _config.CameraRadius,
            elevation ?? _config.InputElevation);
        return new Tensor<double>(rays, new[] { _config.NumInputViews, 6, size, size }).Cast<T>();
    }

    /// <summary>
    /// The per-pixel Plucker coordinates of the LGM input cameras, as <c>prepare_default_rays</c> builds them.
    /// </summary>
    /// <remarks>
    /// <para>
    /// View <c>v</c> sits on an orbit at azimuth <c>360 v / views</c> degrees (kiui <c>orbit_camera</c>, OpenGL
    /// convention): position <c>r (cos e sin a, -sin e, cos e cos a)</c>, looking at the origin with +y up, so its
    /// rotation's columns are <c>right = normalize(up x forward)</c>, <c>up = normalize(forward x right)</c> and
    /// <c>forward = normalize(position)</c>. Pixel <c>(x, y)</c> has camera direction
    /// <c>((x - W/2 + 0.5) / f, -(y - H/2 + 0.5) / f, -1)</c> with <c>f = H / (2 tan(fovy / 2))</c>, rotated to world
    /// space and normalized (<c>get_rays</c>). The embedding is <c>(o x d, d)</c> with <c>o</c> the camera position.
    /// </para>
    /// <para>Camera geometry: computed once per call in double and cast to <typeparamref name="T"/>.</para>
    /// </remarks>
    /// <returns>Row-major <c>[views, 6, size, size]</c> values.</returns>
    internal static double[] PluckerRays(int views, int size, double fovyDegrees, double radius, double elevationDegrees)
    {
        if (views <= 0) throw new ArgumentOutOfRangeException(nameof(views));
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size));
        if (!(fovyDegrees > 0.0 && fovyDegrees < 180.0)) throw new ArgumentOutOfRangeException(nameof(fovyDegrees));

        var rays = new double[views * 6 * size * size];
        double focal = size * 0.5 / System.Math.Tan(0.5 * fovyDegrees * System.Math.PI / 180.0);
        double centre = size * 0.5;
        int plane = size * size;
        double elevation = elevationDegrees * System.Math.PI / 180.0;

        for (int v = 0; v < views; v++)
        {
            double azimuth = 2.0 * System.Math.PI * v / views;
            double ox = radius * System.Math.Cos(elevation) * System.Math.Sin(azimuth);
            double oy = -radius * System.Math.Sin(elevation);
            double oz = radius * System.Math.Cos(elevation) * System.Math.Cos(azimuth);

            var (fx, fy, fz) = Normalize(ox, oy, oz);
            var (rx, ry, rz) = Normalize(Cross(0.0, 1.0, 0.0, fx, fy, fz));
            var (ux, uy, uz) = Normalize(Cross(fx, fy, fz, rx, ry, rz));

            int viewBase = v * 6 * plane;
            for (int y = 0; y < size; y++)
            {
                double cy = -(y - centre + 0.5) / focal;
                for (int x = 0; x < size; x++)
                {
                    double cx = (x - centre + 0.5) / focal;
                    // rays_d = camera_dirs @ R^T with R = [right | up | forward], camera z = -1.
                    var (dx, dy, dz) = Normalize(
                        rx * cx + ux * cy - fx,
                        ry * cx + uy * cy - fy,
                        rz * cx + uz * cy - fz);
                    var (mx, my, mz) = Cross(ox, oy, oz, dx, dy, dz);

                    int pixel = y * size + x;
                    rays[viewBase + 0 * plane + pixel] = mx;
                    rays[viewBase + 1 * plane + pixel] = my;
                    rays[viewBase + 2 * plane + pixel] = mz;
                    rays[viewBase + 3 * plane + pixel] = dx;
                    rays[viewBase + 4 * plane + pixel] = dy;
                    rays[viewBase + 5 * plane + pixel] = dz;
                }
            }
        }

        return rays;
    }

    /// <summary>
    /// Builds the U-Net input from four views: resize to <see cref="LGMConfig.InputSize"/> (bilinear), ImageNet
    /// normalization, and the Plucker embedding concatenated on the channel axis.
    /// </summary>
    /// <param name="views">The views, <c>[NumInputViews, 3, H, W]</c>, RGB in [0, 1], ordered by azimuth 0, 90, 180,
    /// 270.</param>
    /// <param name="elevation">Camera elevation in degrees (kiui convention); null for
    /// <see cref="LGMConfig.InputElevation"/>.</param>
    /// <returns><c>[NumInputViews, 9, InputSize, InputSize]</c>.</returns>
    public Tensor<T> BuildUNetInput(Tensor<T> views, double? elevation = null)
    {
        if (views is null) throw new ArgumentNullException(nameof(views));
        if (views.Rank != 4 || views.Shape[0] != _config.NumInputViews || views.Shape[1] != 3)
            throw new ArgumentException(
                $"Expected views of shape [{_config.NumInputViews}, 3, H, W], got [{string.Join(", ", views.Shape.ToArray())}].",
                nameof(views));

        int size = _config.InputSize;
        var images = views;
        if (views.Shape[2] != size || views.Shape[3] != size)
            images = Engine.Interpolate(views, new[] { size, size }, InterpolateMode.Bilinear, false);

        var stats = new[] { 1, 3, 1, 1 };
        var mean = Engine.TensorBroadcastTo(
            new Tensor<double>(ChannelStatistic(_config.ImageMean, nameof(LGMConfig.ImageMean)), stats).Cast<T>(),
            images.Shape.ToArray());
        var std = Engine.TensorBroadcastTo(
            new Tensor<double>(ChannelStatistic(_config.ImageStd, nameof(LGMConfig.ImageStd)), stats).Cast<T>(),
            images.Shape.ToArray());
        var normalized = Engine.TensorDivide(Engine.TensorSubtract(images, mean), std);

        return Engine.TensorConcatenate(new[] { normalized, ComputeRayEmbeddings(elevation) }, 1);
    }

    #endregion

    #region Gaussian reconstruction

    /// <summary>
    /// Reconstructs the Gaussians of an object from its four views (the official <c>forward_gaussians</c>).
    /// </summary>
    /// <param name="views">The views, <c>[NumInputViews, 3, H, W]</c>, RGB in [0, 1], azimuths 0, 90, 180, 270.</param>
    /// <param name="elevation">Camera elevation in degrees (kiui convention); null for
    /// <see cref="LGMConfig.InputElevation"/>.</param>
    /// <returns><c>NumInputViews x SplatSize^2</c> Gaussians.</returns>
    public LGMGaussians<T> ReconstructGaussians(Tensor<T> views, double? elevation = null)
    {
        var input = BuildUNetInput(views, elevation);
        var splatterMaps = _gaussianProjection.Forward(_unet.Forward(input));
        return ActivateGaussians(splatterMaps);
    }

    /// <summary>
    /// Turns splatter maps into Gaussians: <c>[V, 14, S, S]</c> is flattened view by view, pixel by pixel, and each
    /// group of channels goes through its activation.
    /// </summary>
    /// <param name="splatterMaps">Raw U-Net output after the 1x1 projection, <c>[V, 14, S, S]</c>.</param>
    public LGMGaussians<T> ActivateGaussians(Tensor<T> splatterMaps)
    {
        if (splatterMaps is null) throw new ArgumentNullException(nameof(splatterMaps));
        if (splatterMaps.Rank != 4 || splatterMaps.Shape[1] != GaussianChannels)
            throw new ArgumentException($"Expected splatter maps of shape [V, {GaussianChannels}, S, S].",
                nameof(splatterMaps));

        int count = splatterMaps.Shape[0] * splatterMaps.Shape[2] * splatterMaps.Shape[3];
        // x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14) for one object: [V, 14, S, S] -> [V*S*S, 14].
        var flat = Engine.Reshape(
            Engine.TensorPermute(splatterMaps, new[] { 0, 2, 3, 1 }).Contiguous(),
            new[] { count, GaussianChannels });

        T bound = NumOps.FromDouble(_config.PositionBound);
        var positions = Engine.TensorClamp(Slice(flat, 0, 3), NumOps.Negate(bound), bound);
        var opacities = Engine.TensorSigmoid(Slice(flat, 3, 1));
        var scales = Engine.TensorMultiplyScalar(StableSoftplus(Slice(flat, 4, 3)), NumOps.FromDouble(_config.ScaleFactor));
        var rotations = Engine.TensorNormalize(Slice(flat, 7, 4), 1, NormalizeEpsilon());
        var half = NumOps.FromDouble(0.5);
        var colors = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(Engine.TensorTanh(Slice(flat, 11, 3)), half), half);
        return new LGMGaussians<T>(positions, opacities, scales, rotations, colors);
    }

    /// <summary>
    /// Generates the four views from one image with the multi-view model, then reconstructs their Gaussians.
    /// </summary>
    /// <param name="inputImage">The image, <c>[1, 3, H, W]</c>, in the multi-view model's VAE range, seen from
    /// azimuth 0.</param>
    /// <param name="numInferenceSteps">Multi-view denoising steps (30 in the official demo).</param>
    /// <param name="guidanceScale">Multi-view guidance scale (5.0 in the official demo).</param>
    /// <param name="seed">Seed of the multi-view sampler.</param>
    public LGMGaussians<T> ImageToGaussians(Tensor<T> inputImage, int numInferenceSteps = 30,
        double guidanceScale = 5.0, int? seed = null)
    {
        if (inputImage is null) throw new ArgumentNullException(nameof(inputImage));
        // The official pipeline generates the views with ImageDream at elevation 0 (LGMConfig.InputElevation). The
        // multi-view model's elevation is positive looking down; LGM's (kiui) is negative looking down. Its view v is at
        // azimuth 360 v / views measured from the input's viewpoint, which is LGM's camera order, so no reordering is
        // needed (the official code reorders ImageDream's output, whose azimuth origin differs from kiui's).
        var views = _multiViewModel.GenerateNovelViewsFromImage(
            inputImage, _config.NumInputViews, numInferenceSteps, guidanceScale, seed, -_config.InputElevation);
        return ReconstructGaussians(StackViews(views), _config.InputElevation);
    }

    /// <summary>
    /// Generates the four views from a text prompt with the multi-view model, then reconstructs their Gaussians.
    /// </summary>
    /// <param name="prompt">Text description of the object.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="numInferenceSteps">Multi-view denoising steps (30 in the official demo).</param>
    /// <param name="guidanceScale">Multi-view guidance scale (7.5 in the official demo).</param>
    /// <param name="seed">Seed of the multi-view sampler.</param>
    public LGMGaussians<T> TextToGaussians(string prompt, string? negativePrompt = null, int numInferenceSteps = 30,
        double guidanceScale = 7.5, int? seed = null)
    {
        if (prompt is null) throw new ArgumentNullException(nameof(prompt));
        // MVDream's elevation is positive looking down; LGM's (kiui) is negative looking down.
        var views = _multiViewModel.GenerateMultiView(prompt, negativePrompt, _config.NumInputViews,
            numInferenceSteps, guidanceScale, -_config.InputElevation, seed);
        return ReconstructGaussians(StackViews(views), _config.InputElevation);
    }

    /// <summary>
    /// A renderable <see cref="GaussianSplatting{T}"/> scene holding exactly these Gaussians.
    /// </summary>
    /// <param name="gaussians">The Gaussians.</param>
    /// <param name="options">Scene options; null for RGB colour (no spherical harmonics, as LGM predicts) and no
    /// densification.</param>
    public GaussianSplatting<T> ToGaussianSplatting(LGMGaussians<T> gaussians, GaussianSplattingOptions? options = null)
    {
        if (gaussians is null) throw new ArgumentNullException(nameof(gaussians));
        var sceneOptions = options ?? new GaussianSplattingOptions
        {
            UseSphericalHarmonics = false,
            ShDegree = 0,
            EnableDensification = false
        };
        return GaussianSplatting<T>.FromGaussians(sceneOptions, gaussians.Positions, gaussians.Scales,
            gaussians.Rotations, gaussians.Opacities, gaussians.Colors);
    }

    /// <summary>
    /// The Gaussian centres with their colours, <c>[1, M, 6]</c> (xyz, rgb), keeping Gaussians whose opacity is at
    /// least <see cref="LGMConfig.OpacityThreshold"/>.
    /// </summary>
    /// <param name="gaussians">The Gaussians.</param>
    /// <param name="maxPoints">Keep at most this many, the most opaque first; null keeps every visible Gaussian. The
    /// kept points stay in Gaussian order.</param>
    public Tensor<T> GaussiansToPointCloud(LGMGaussians<T> gaussians, int? maxPoints = null)
    {
        var (positions, colors) = VisibleGaussians(gaussians, maxPoints);
        var points = Engine.TensorConcatenate(new[] { positions, colors }, 1);
        return Engine.Reshape(points, new[] { 1, positions.Shape[0], 6 });
    }

    /// <summary>
    /// The mesh of the Gaussians' density level set (<see cref="GaussianMeshExtractor.ExtractMesh{T}"/>), in the
    /// Gaussians' coordinates, wound outward with per-vertex normals.
    /// </summary>
    /// <param name="gaussians">The Gaussians.</param>
    /// <param name="options">Grid and threshold settings; null for <see cref="LGMConfig.MeshExtraction"/>.</param>
    public Mesh3D<T> GaussiansToMesh(LGMGaussians<T> gaussians, GaussianMeshExtractionOptions? options = null)
    {
        return GaussianMeshExtractor.ExtractMesh(ToGaussianCloud(gaussians), options ?? _config.MeshExtraction);
    }

    /// <summary>
    /// The Gaussians as a <see cref="GaussianCloud{T}"/>, the meshing components' input. Scales are floored at 1e-6
    /// (a softplus can underflow to zero) and opacities flattened to [N].
    /// </summary>
    /// <param name="gaussians">The Gaussians.</param>
    public GaussianCloud<T> ToGaussianCloud(LGMGaussians<T> gaussians)
    {
        if (gaussians is null) throw new ArgumentNullException(nameof(gaussians));
        var scales = Engine.TensorClampMin(gaussians.Scales, NumOps.FromDouble(1e-6));
        var opacities = Engine.Reshape(gaussians.Opacities, new[] { gaussians.Count });
        return new GaussianCloud<T>(gaussians.Positions, scales, gaussians.Rotations, opacities);
    }

    #endregion

    #region I3DDiffusionModel

    /// <inheritdoc />
    /// <remarks>
    /// Text -> four MVDream views -> Gaussians -> their centres and colours, <c>[1, M, 6]</c>. <paramref name="numPoints"/>
    /// caps M (most opaque first); the default keeps every Gaussian at or above the opacity threshold, up to
    /// <c>4 x SplatSize^2</c>. Requires a text conditioner on the multi-view model.
    /// </remarks>
    public override Tensor<T> GeneratePointCloud(
        string prompt, string? negativePrompt = null,
        int? numPoints = null, int numInferenceSteps = 30,
        double guidanceScale = 7.5, int? seed = null)
    {
        var gaussians = TextToGaussians(prompt, negativePrompt, numInferenceSteps, guidanceScale, seed);
        return GaussiansToPointCloud(gaussians, numPoints);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Image -> four MVDream views -> Gaussians -> the mesh of their density level set (<see cref="GaussiansToMesh"/>,
    /// with <see cref="LGMConfig.MeshExtraction"/>). <paramref name="numViews"/> must equal
    /// <see cref="LGMConfig.NumInputViews"/>.
    /// </remarks>
    public override Mesh3D<T> ImageTo3D(
        Tensor<T> inputImage,
        int numViews = 4,
        int numInferenceSteps = 30,
        double guidanceScale = 5.0,
        int? seed = null)
    {
        if (numViews != _config.NumInputViews)
            throw new ArgumentOutOfRangeException(nameof(numViews),
                $"LGM reconstructs from exactly {_config.NumInputViews} views.");
        var gaussians = ImageToGaussians(inputImage, numInferenceSteps, guidanceScale, seed);
        return GaussiansToMesh(gaussians);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Text -> four MVDream views -> Gaussians -> the mesh of their density level set, sampled on a
    /// <paramref name="resolution"/>^3 grid (128 in DreamGaussian; it must be a multiple of
    /// <see cref="GaussianMeshExtractionOptions.BlockCount"/>). The other extraction settings come from
    /// <see cref="LGMConfig.MeshExtraction"/>. Requires a text conditioner on the multi-view model.
    /// </remarks>
    public override Mesh3D<T> GenerateMesh(
        string prompt, string? negativePrompt = null,
        int resolution = 128, int numInferenceSteps = 30,
        double guidanceScale = 7.5, int? seed = null)
    {
        var gaussians = TextToGaussians(prompt, negativePrompt, numInferenceSteps, guidanceScale, seed);
        var settings = _config.MeshExtraction ?? new GaussianMeshExtractionOptions();
        return GaussiansToMesh(gaussians, new GaussianMeshExtractionOptions
        {
            Resolution = resolution,
            BlockCount = settings.BlockCount,
            RelaxRatio = settings.RelaxRatio,
            DensityThreshold = settings.DensityThreshold,
            OpacityCullThreshold = settings.OpacityCullThreshold,
            NormalizedHalfExtent = settings.NormalizedHalfExtent
        });
    }

    #endregion

    #region Helpers

    /// <summary>The Gaussian count of a configuration: views x splatter size squared (4 x 128^2 by default).</summary>
    internal static int GaussianCountFor(LGMConfig config)
    {
        if (config is null) throw new ArgumentNullException(nameof(config));
        int downs = config.DownChannels is null ? 0 : config.DownChannels.Length;
        int ups = config.UpChannels is null ? 0 : config.UpChannels.Length;
        int size = config.InputSize;
        for (int i = 0; i < downs - 1; i++) size = (size + 1) / 2;
        for (int i = 0; i < ups - 1; i++) size *= 2;
        return checked(config.NumInputViews * size * size);
    }

    private (Tensor<T> Positions, Tensor<T> Colors) VisibleGaussians(LGMGaussians<T> gaussians, int? maxPoints)
    {
        if (gaussians is null) throw new ArgumentNullException(nameof(gaussians));
        if (maxPoints.HasValue && maxPoints.Value <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxPoints), "maxPoints must be positive.");

        // Selection is host logic over one opacity per Gaussian; the gathers below stay engine ops.
        double[] opacity = gaussians.Opacities.Cast<double>().ToArray();
        var visible = new List<int>(opacity.Length);
        for (int i = 0; i < opacity.Length; i++)
            if (opacity[i] >= _config.OpacityThreshold) visible.Add(i);

        if (maxPoints.HasValue && visible.Count > maxPoints.Value)
        {
            // Most opaque first, the lower index winning ties, so the cut is deterministic; then back to Gaussian
            // order.
            visible.Sort((a, b) =>
            {
                int byOpacity = opacity[b].CompareTo(opacity[a]);
                return byOpacity != 0 ? byOpacity : a.CompareTo(b);
            });
            visible.RemoveRange(maxPoints.Value, visible.Count - maxPoints.Value);
            visible.Sort();
        }

        if (visible.Count == 0)
            return (new Tensor<T>(new[] { 0, 3 }), new Tensor<T>(new[] { 0, 3 }));

        var indices = new Tensor<int>(visible.ToArray(), new[] { visible.Count });
        return (Engine.TensorIndexSelect(gaussians.Positions, indices, 0),
            Engine.TensorIndexSelect(gaussians.Colors, indices, 0));
    }

    private Tensor<T> Slice(Tensor<T> flat, int start, int width)
        => Engine.TensorNarrow(flat, 1, start, width).Contiguous();

    /// <summary>
    /// <c>softplus(x) = max(x, 0) + log(1 + exp(-|x|))</c>: the exponent is never positive, so it cannot overflow in
    /// any precision (a direct <c>log(1 + exp(x))</c> overflows Half for x above about 11).
    /// </summary>
    private Tensor<T> StableSoftplus(Tensor<T> x)
        => Engine.TensorAdd(Engine.TensorReLU(x), Engine.TensorLog1p(Engine.TensorExp(Engine.TensorNegate(Engine.TensorAbs(x)))));

    /// <summary>The quaternion normalization floor: torch's 1e-12 where the type holds it, else its smallest normal
    /// step (1e-12 rounds to zero in Half).</summary>
    private static T NormalizeEpsilon()
    {
        T epsilon = NumOps.FromDouble(1e-12);
        return NumOps.Equals(epsilon, NumOps.Zero) ? NumOps.FromDouble(1e-7) : epsilon;
    }

    /// <summary>Stacks multi-view decoder outputs ([1, 3, H, W] each, in [-1, 1]) into [V, 3, H, W] RGB in [0, 1].</summary>
    private Tensor<T> StackViews(Tensor<T>[] views)
    {
        if (views is null || views.Length != _config.NumInputViews)
            throw new InvalidOperationException(
                $"The multi-view model returned {(views is null ? 0 : views.Length)} views; LGM needs {_config.NumInputViews}.");
        var stacked = Engine.TensorConcatenate(views, 0);
        var half = NumOps.FromDouble(0.5);
        // The diffusers post-processing of a VAE decode: (x / 2 + 0.5).clamp(0, 1).
        return Engine.TensorClamp(Engine.TensorAddScalar(Engine.TensorMultiplyScalar(stacked, half), half),
            NumOps.Zero, NumOps.One);
    }

    private static double[] ChannelStatistic(double[] values, string name)
    {
        if (values is null || values.Length != 3)
            throw new InvalidOperationException($"LGMConfig.{name} must hold three values (R, G, B).");
        return values;
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "LGM",
            Version = "1.0",
            Description = "LGM: multi-view diffusion views + Plucker rays -> asymmetric U-Net -> 3D Gaussians",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "asymmetric-unet-multiview-attention");
        metadata.SetProperty("representation", "3d-gaussian-splatting");
        metadata.SetProperty("multiview_model", "MVDream");
        metadata.SetProperty("num_input_views", _config.NumInputViews);
        metadata.SetProperty("input_size", _config.InputSize);
        metadata.SetProperty("input_channels_per_view", InputChannelsPerView);
        metadata.SetProperty("gaussian_channels", GaussianChannels);
        metadata.SetProperty("splat_size", SplatSize);
        metadata.SetProperty("gaussian_count", DefaultPointCount);
        metadata.SetProperty("feed_forward", true);

        return metadata;
    }

    #endregion
}
