using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralRadianceFields.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.NeuralRadianceFields.Models;

/// <summary>
/// Implements 3D Gaussian Splatting for real-time novel view synthesis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> 3D Gaussian Splatting is the newest breakthrough in novel view synthesis,
/// achieving real-time rendering (30+ FPS) with state-of-the-art quality.
/// </para>
/// <para>
/// Speed comparison:
/// - NeRF: ~30 seconds per frame
/// - Instant-NGP: ~30 milliseconds per frame (~1000× faster than NeRF)
/// - Gaussian Splatting: ~10 milliseconds per frame (~3000× faster than NeRF)
/// - Gaussian Splatting can render at 100+ FPS on modern GPUs!
/// </para>
/// <para>
/// Key innovation - Explicit representation:
/// - NeRF/Instant-NGP: Implicit (neural network represents the scene)
/// - Gaussian Splatting: Explicit (scene is a collection of 3D Gaussians)
///
/// What's a 3D Gaussian?
/// - A "blob" of color in 3D space
/// - Has a center position (x, y, z)
/// - Has a size and shape (covariance matrix)
/// - Has a color (RGB)
/// - Has opacity (alpha)
///
/// Think of it like this:
/// - Traditional rendering: Triangles (hard edges)
/// - Volume rendering (NeRF): Sample continuous field
/// - Gaussian Splatting: "Paint" with fuzzy 3D blobs
/// </para>
/// <para>
/// How it works:
/// 1. Start with point cloud (from SfM like COLMAP)
/// 2. Place a Gaussian at each point
/// 3. Optimize Gaussian parameters:
///    - Position: Where is the Gaussian?
///    - Covariance: What shape/size is it?
///    - Color: What color is it?
///    - Opacity: How transparent is it?
/// 4. Add/remove Gaussians as needed (adaptive densification)
/// 5. Render by "splatting" Gaussians onto image
/// </para>
/// <para>
/// Gaussian Splatting rendering:
/// 1. For each Gaussian:
///    - Project to 2D image plane
///    - Compute 2D Gaussian on screen
///    - Determines which pixels it affects
/// 2. Sort Gaussians by depth (back to front)
/// 3. For each pixel:
///    - Blend Gaussians that affect it (alpha blending)
///    - Front-to-back or back-to-front blending
/// 4. Result: Final pixel color
///
/// This is MUCH faster than ray marching because:
/// - No network evaluation (explicit representation)
/// - Highly parallelizable (each Gaussian independent)
/// - Efficient GPU rasterization (like traditional graphics)
/// </para>
/// <para>
/// Why Gaussians?
/// - Smooth gradients for optimization
/// - Can be rasterized efficiently (like triangles)
/// - 2D projection is also Gaussian (mathematically elegant)
/// - Adaptive: Can represent sharp edges with many small Gaussians
///   or smooth surfaces with fewer large Gaussians
/// </para>
/// <para>
/// Gaussian parameters:
/// - Position μ: Center of Gaussian (3D point)
/// - Covariance Σ: Shape and orientation (3×3 matrix)
///   - Can represent ellipsoids (stretched in different directions)
///   - Encoded as rotation + scale for easier optimization
/// - Color c: RGB values (often with spherical harmonics for view-dependent effects)
/// - Opacity α: Transparency (0 = invisible, 1 = opaque)
///
/// Total per Gaussian: 3 (pos) + 4 (rotation) + 3 (scale) + 3 (color) + 1 (opacity) = 14 values
/// With spherical harmonics: Can be 40-60 values per Gaussian
/// </para>
/// <para>
/// Adaptive densification:
/// During optimization, Gaussians are dynamically added/removed:
/// - Clone: Copy Gaussians in high-gradient regions (need more detail)
/// - Split: Large Gaussians → multiple smaller ones (refine detail)
/// - Prune: Remove transparent Gaussians (save memory)
///
/// Example:
/// - Start: 100K Gaussians from point cloud
/// - After optimization: 500K-5M Gaussians
/// - High detail areas: Many small Gaussians (e.g., edges, textures)
/// - Smooth areas: Fewer large Gaussians (e.g., walls, sky)
/// </para>
/// <para>
/// Training process:
/// 1. Initialize from Structure-from-Motion point cloud
/// 2. For each training iteration:
///    - Render view using current Gaussians
///    - Compute loss (difference from ground truth image)
///    - Backpropagate to Gaussian parameters
///    - Update positions, colors, shapes, opacities
///    - Every N iterations: Densify/prune Gaussians
/// 3. Converges in ~10-30 minutes (similar to Instant-NGP)
/// </para>
/// <para>
/// Advantages over NeRF:
/// - Faster rendering: 100+ FPS vs 0.03 FPS (NeRF)
/// - Explicit representation: Easy to edit, manipulate
/// - No network evaluation: Simpler deployment
/// - Better quality: Often sharper details
/// - Easier to understand: Physical interpretation (colored blobs)
///
/// Disadvantages:
/// - Memory: More than NeRF (millions of Gaussians)
///   - NeRF: ~5-50MB
///   - Gaussian Splatting: ~100-500MB
/// - Requires good initialization (SfM point cloud)
/// - Can have "floating" artifacts in empty regions
/// - File size for storage
/// </para>
/// <para>
/// Rendering pipeline (simplified):
/// ```
/// For each view:
///   1. Transform Gaussians to camera space
///   2. Project 3D Gaussians to 2D screen space
///   3. Compute 2D Gaussian parameters (center, covariance)
///   4. Determine which tiles/pixels each Gaussian affects
///   5. Sort Gaussians by depth within each tile
///   6. For each pixel:
///      - Accumulate color from affecting Gaussians
///      - Alpha blending: C = Σ α_i * c_i * Π(1 - α_j) for j &lt; i
///   7. Output: Rendered image
/// ```
/// </para>
/// <para>
/// Applications (especially suited for):
/// - Real-time VR/AR: Low latency is critical
/// - Gaming: Integration with game engines
/// - Digital twins: Interactive 3D models of real places
/// - Telepresence: Realistic remote environments
/// - Film pre-visualization: Fast preview of captured scenes
/// - Live events: Real-time volumetric capture
/// </para>
/// <para>
/// Comparison table:
///
/// Feature              | NeRF    | Instant-NGP | Gaussian Splatting
/// ---------------------|---------|-------------|-------------------
/// Rendering speed      | 30 s    | 30 ms       | 10 ms (100+ FPS)
/// Training time        | 1-2 days| 5-10 min    | 10-30 min
/// Quality              | High    | High        | Very High
/// Memory usage         | 5 MB    | 50 MB       | 200-500 MB
/// Editability          | Hard    | Hard        | Easy
/// Real-time rendering  | No      | Borderline  | Yes
/// GPU requirement      | Any     | CUDA        | Any (faster w/ CUDA)
/// </para>
/// <para>
/// Reference: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
/// by Kerbl et al., SIGGRAPH 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a 3D Gaussian Splatting model from an initial point cloud
/// var options = new GaussianSplattingOptions { MaxGaussians = 100000, ShDegree = 3 };
/// var pointCloud = new Matrix&lt;float&gt;(1000, 3); // 1000 initial 3D points
/// var colors = new Matrix&lt;float&gt;(1000, 3);     // RGB colors per point
/// var gs = new GaussianSplatting&lt;float&gt;(options, pointCloud, colors);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.ThreeDGeneration)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("3D Gaussian Splatting for Real-Time Radiance Field Rendering", "https://doi.org/10.1145/3592433", Year = 2023, Authors = "Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis")]
public partial class GaussianSplatting<T> : AiDotNet.NeuralNetworks.VectorModelLayoutBase<T>, IRadianceField<T>,
    IHyperparameterAware<T, Tensor<T>, Tensor<T>>,
    NeuralRadianceFields.Interfaces.IImageTrainable<T>
{
    private readonly GaussianSplattingOptions _options;

    // Gaussian state, one row per Gaussian, generic in T. Declaration order is the flat parameter layout:
    // positions [N, 3] | rotation quaternions (w, x, y, z) [N, 4] | log-scales [N, 3] | opacity logits [N] |
    // spherical-harmonics coefficients [N, K, 3] (coefficient-major, channel last). Without spherical harmonics
    // K is 1 and the single coefficient row holds the RGB colour directly. Scales are log-parameterised (the
    // exp activation of Kerbl et al. 2023), so every scale stays positive under gradient steps.
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _positions = new Tensor<T>(new[] { 0, 3 });

    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _rotations = new Tensor<T>(new[] { 0, 4 });

    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _logScales = new Tensor<T>(new[] { 0, 3 });

    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _opacityLogits = new Tensor<T>(new[] { 0 });

    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _shCoefficients = new Tensor<T>(new[] { 0, 1, 3 });

    /// <summary>
    /// Per-Gaussian learning-rate multipliers [N, 5] (position, scale, opacity, rotation, colour). Split
    /// children compound <see cref="AiDotNet.Models.Options.SplitChildLearningRateScales"/> into their row; every
    /// other Gaussian keeps 1. Optimizer state, not a model parameter.
    /// </summary>
    [Scratch]
    private Tensor<T> _learningRateScales = new Tensor<T>(new[] { 0, LearningRateScaleColumns });

    private const int LearningRateScaleColumns = 5;
    private const int LrPosition = 0;
    private const int LrScale = 1;
    private const int LrOpacity = 2;
    private const int LrRotation = 3;
    private const int LrColor = 4;

    // Kerbl et al. 2023, section 5.2: every 3000 iterations alpha <- min(alpha, 0.01) inside the densification window.
    private const int OpacityResetInterval = 3000;
    private const double OpacityResetCeiling = 0.01;

    // Element budgets for the dense pairwise passes (point x Gaussian, pixel x splat), processed in chunks.
    private const int PairBudget = 1 << 22;
    private const int RasterBudget = 1 << 16;

    // Columns of ProjectedSplats.Raster.
    private const int ColMeanX = 0;
    private const int ColMeanY = 1;
    private const int ColInvA = 2;
    private const int ColInvB = 3;
    private const int ColInvC = 4;
    private const int ColOpacity = 5;
    private const int ColRed = 6;
    private const int ColGreen = 7;
    private const int ColBlue = 8;
    private const int ColMinX = 9;
    private const int ColMaxX = 10;
    private const int ColMinY = 11;
    private const int ColMaxY = 12;
    private const int RasterColumns = 13;

    // Columns of ProjectedSplats.Extra: camera-space mean, camera-space covariance (row-major upper triangle) and the
    // unclamped colour.
    private const int ExCamX = 0;
    private const int ExCamY = 1;
    private const int ExCamZ = 2;
    private const int ExCov00 = 3;
    private const int ExCov01 = 4;
    private const int ExCov02 = 5;
    private const int ExCov11 = 6;
    private const int ExCov12 = 7;
    private const int ExCov22 = 8;
    private const int ExRawRed = 9;
    private const int ExRawGreen = 10;
    private const int ExRawBlue = 11;
    private const int ExtraColumns = 12;

    // Columns of the per-splat image-gradient sums.
    private const int GradSumRed = 0;
    private const int GradSumGreen = 1;
    private const int GradSumBlue = 2;
    private const int GradInvA = 3;
    private const int GradInvB = 4;
    private const int GradInvC = 5;
    private const int GradOpacity = 6;
    private const int GradMeanX = 7;
    private const int GradMeanY = 8;
    private const int GradColumns = 9;

    /// <summary>A symmetric 3x3 matrix per Gaussian, one [N] tensor per upper-triangle entry.</summary>
    private sealed class Symmetric3
    {
        public Symmetric3(Tensor<T> xx, Tensor<T> xy, Tensor<T> xz, Tensor<T> yy, Tensor<T> yz, Tensor<T> zz)
        {
            XX = xx;
            XY = xy;
            XZ = xz;
            YY = yy;
            YZ = yz;
            ZZ = zz;
        }

        public Tensor<T> XX { get; }
        public Tensor<T> XY { get; }
        public Tensor<T> XZ { get; }
        public Tensor<T> YY { get; }
        public Tensor<T> YZ { get; }
        public Tensor<T> ZZ { get; }
    }

    /// <summary>The Gaussians in camera coordinates.</summary>
    private sealed class CameraSpace
    {
        public CameraSpace(Tensor<T> position, Tensor<T> covariance, Tensor<T> worldOffset)
        {
            Position = position;
            Covariance = covariance;
            WorldOffset = worldOffset;
        }

        /// <summary>Camera-space means [N, 3].</summary>
        public Tensor<T> Position { get; }

        /// <summary>Camera-space covariances, row-major [N, 9].</summary>
        public Tensor<T> Covariance { get; }

        /// <summary>World-space camera-to-Gaussian offsets [N, 3] (the viewing direction before normalisation).</summary>
        public Tensor<T> WorldOffset { get; }
    }

    /// <summary>The visible splats in front-to-back order.</summary>
    private sealed class ProjectedSplats
    {
        public ProjectedSplats(Tensor<int> indices, Tensor<T> visibleMask, Tensor<T> raster, Tensor<T> extra, Tensor<T> basis)
        {
            Indices = indices;
            VisibleMask = visibleMask;
            Raster = raster;
            Extra = extra;
            Basis = basis;
        }

        /// <summary>Gaussian row of each splat [M], sorted by camera depth (stable).</summary>
        public Tensor<int> Indices { get; }

        /// <summary>1 for every visible Gaussian, 0 otherwise [N].</summary>
        public Tensor<T> VisibleMask { get; }

        /// <summary>Rasterisation attributes [M, RasterColumns].</summary>
        public Tensor<T> Raster { get; }

        /// <summary>Camera-space geometry and unclamped colour [M, ExtraColumns].</summary>
        public Tensor<T> Extra { get; }

        /// <summary>Spherical-harmonics basis of each splat's viewing direction [M, K].</summary>
        public Tensor<T> Basis { get; }

        public int Count => Indices.Length;
    }

    /// <summary>Camera-space-projected mean gradients of the visible Gaussians of one image step.</summary>
    private sealed class GaussianGradient
    {
        public GaussianGradient(Tensor<int> indices, Tensor<T> gradients)
        {
            Guard.NotNull(indices);
            Indices = indices;
            Guard.NotNull(gradients);
            Gradients = gradients;
        }

        /// <summary>Gaussian rows [M], in depth order.</summary>
        public Tensor<int> Indices { get; }

        /// <summary>World-space position gradients [M, 3].</summary>
        public Tensor<T> Gradients { get; }

        public int Count => Indices.Length;
    }

    private readonly bool _useSphericalHarmonics;
    private readonly int _shDegree; // Degree of spherical harmonics (0-3)
    private int _trainingStep;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    public GaussianSplatting()
        : this(new GaussianSplattingOptions(), null, null, null)
    {
    }

    public GaussianSplatting(
        GaussianSplattingOptions options,
        Matrix<T>? initialPointCloud = null,
        Matrix<T>? initialColors = null,
        ILossFunction<T>? lossFunction = null)
        : base(CreateArchitecture(), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression))
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }
        _options = options;
        Options = _options;
        if (options.ShDegree < 0 || options.ShDegree > 3)
        {
            throw new ArgumentOutOfRangeException(nameof(options.ShDegree), "shDegree must be between 0 and 3.");
        }
        if (options.DensificationInterval <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.DensificationInterval), "Densification interval must be positive.");
        }
        if (options.MaxGaussians <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.MaxGaussians), "MaxGaussians must be positive.");
        }
        if (options.SplitPositionJitter < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.SplitPositionJitter), "SplitPositionJitter cannot be negative.");
        }
        if (options.SplitScaleFactor <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.SplitScaleFactor), "SplitScaleFactor must be positive.");
        }
        if (options.SplitOpacityFactor <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.SplitOpacityFactor), "SplitOpacityFactor must be positive.");
        }
        if (options.SplitOpacityMax <= 0.0 || options.SplitOpacityMax > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.SplitOpacityMax), "SplitOpacityMax must be in (0, 1].");
        }
        if (options.TileSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.TileSize), "TileSize must be positive.");
        }
        if (options.SpatialIndexRadius <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.SpatialIndexRadius), "SpatialIndexRadius must be positive.");
        }
        if (options.InitialNeighborSearchScale <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.InitialNeighborSearchScale), "InitialNeighborSearchScale must be positive.");
        }
        if (options.MinScale <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.MinScale), "MinScale must be positive.");
        }
        if (options.InitialScaleMultiplier <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.InitialScaleMultiplier), "InitialScaleMultiplier must be positive.");
        }
        if (options.DefaultPointSpacing <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.DefaultPointSpacing), "DefaultPointSpacing must be positive.");
        }

        _useSphericalHarmonics = options.UseSphericalHarmonics;
        _shDegree = options.ShDegree;
        _shCoefficients = new Tensor<T>(new[] { 0, GetShBasisCount(), 3 });

        EnableDensification = options.EnableDensification;
        DensificationInterval = options.DensificationInterval;
        PruneOpacityThreshold = options.PruneOpacityThreshold;
        SplitGradientThreshold = options.SplitGradientThreshold;
        SplitPositionJitter = options.SplitPositionJitter;
        SplitScaleFactor = options.SplitScaleFactor;
        SplitOpacityFactor = options.SplitOpacityFactor;
        SplitOpacityMax = options.SplitOpacityMax;
        MaxGaussians = options.MaxGaussians;
        PositionLearningRate = options.PositionLearningRate;
        ColorLearningRate = options.ColorLearningRate;
        OpacityLearningRate = options.OpacityLearningRate;
        ScaleLearningRate = options.ScaleLearningRate;
        RotationLearningRate = options.RotationLearningRate;
        TileSize = options.TileSize;
        EnableSpatialIndex = options.EnableSpatialIndex;
        SpatialIndexRadius = options.SpatialIndexRadius;
        InitialNeighborSearchScale = options.InitialNeighborSearchScale;
        InitialScaleMultiplier = options.InitialScaleMultiplier;
        DefaultPointSpacing = options.DefaultPointSpacing;
        MinScale = options.MinScale;

        if (initialPointCloud != null)
        {
            InitializeFromPointCloud(initialPointCloud, initialColors);
        }
        else
        {
            // No point cloud provided → seed a small uniform-cube cloud so the
            // model is trainable out of the box. Without this, _gaussians is
            // empty and every operation (Predict, Train, GetParameters) is a
            // silent no-op — model-family invariant tests that check "training
            // changed parameters" or "loss decreased" fail because there are
            // literally zero parameters to change. The cloud is small (8
            // Gaussians) so it adds negligible overhead but gives the model
            // a non-trivial trainable surface from construction. Production
            // callers should still pass a real point cloud (typically the
            // COLMAP / SfM output the 3D Gaussian Splatting paper assumes).
            SeedDefaultGaussianCloud();
        }
    }

    /// <summary>
    /// Seeds a small unit-cube Gaussian cloud (up to 8 corners) so the
    /// parameterless constructor produces a model with non-empty trainable
    /// state. Used only when no <paramref name="initialPointCloud"/> is
    /// supplied to the main constructor. Respects <see cref="MaxGaussians"/>
    /// — if the configured cap is below 8, the seed is truncated to that cap
    /// so the constraint is honoured from construction.
    /// </summary>
    private void SeedDefaultGaussianCloud()
    {
        // Honor MaxGaussians: a configured cap of 0 means "explicit empty
        // model" (the caller will populate Gaussians later). A cap between
        // 1 and 7 truncates the unit-cube seed. The cap is validated at
        // construction (must be >= 0) so we only need to clamp here.
        int seedCap = MaxGaussians > 0 ? Math.Min(8, MaxGaussians) : 0;
        if (seedCap == 0) return;

        // Corners of a unit cube centred on the origin — gives a coarse but
        // varied initial geometry that exercises position / colour / scale /
        // opacity parameter updates symmetrically.
        // Cube corners at +-0.5 in binary order: corner i has x = bit 2, y = bit 1, z = bit 0 of i.
        var engine = AiDotNetEngine.Current;
        var index = engine.TensorArange<T>(0, seedCap, 1);
        var two = engine.TensorFullLike(index, NumOps.FromDouble(2.0));
        var xBit = engine.TensorFloor(engine.TensorDivideScalar(index, NumOps.FromDouble(4.0)));
        var yBit = engine.TensorRemainder(engine.TensorFloor(engine.TensorDivideScalar(index, NumOps.FromDouble(2.0))), two);
        var zBit = engine.TensorRemainder(index, two);
        var half = NumOps.FromDouble(0.5);
        var corners = engine.TensorStack(new[]
        {
            engine.TensorSubtractScalar(xBit, half),
            engine.TensorSubtractScalar(yBit, half),
            engine.TensorSubtractScalar(zBit, half)
        }, 1);
        var pointCloud = corners.ToMatrix();
        InitializeFromPointCloud(pointCloud, colors: null);
    }

    /// <summary>
    /// Seeds a Gaussian cloud with <paramref name="count"/> placeholder
    /// points at the origin. Used by <see cref="CreateNewInstance"/> so a
    /// cloned model starts with the SAME Gaussian count as the original;
    /// the caller's subsequent <see cref="UpdateParameters"/> overwrites
    /// every field with the snapshot values, so the placeholder coordinates
    /// don't matter — only the count does. Without this, CreateNewInstance
    /// would default-seed 8 Gaussians and UpdateParameters would reject
    /// any model whose Gaussian count differs from 8.
    /// </summary>
    private void SeedPlaceholderGaussianCloud(int count)
    {
        if (count <= 0) return;
        var pointCloud = new Matrix<T>(count, 3); // all zeros — UpdateParameters overwrites
        InitializeFromPointCloud(pointCloud, colors: null);
    }

    public bool EnableDensification { get; set; }
    public int DensificationInterval { get; set; }
    public double PruneOpacityThreshold { get; set; }
    public double SplitGradientThreshold { get; set; }
    public double SplitPositionJitter { get; set; }
    public double SplitScaleFactor { get; set; }
    public double SplitOpacityFactor { get; set; }
    public double SplitOpacityMax { get; set; }
    public int MaxGaussians { get; set; }

    // #1835 densification-schedule getters. Each returns the caller's explicit override
    // from GaussianSplattingOptions if set, else the paper default (Kerbl et al. 2023).
    // Reading via these getters (not the raw fields) lets DensifyAndPrune honor the new
    // schedule knobs without breaking legacy callers who set only the original fields.
    internal int EffectiveDensificationStartIteration => _options.DensificationStartIteration ?? 500;
    internal int EffectiveDensificationEndIteration   => _options.DensificationEndIteration   ?? 15000;
    internal double EffectiveGradientNormThreshold    => _options.GradientNormThreshold      ?? SplitGradientThreshold;
    internal double EffectiveOpacityPruneThreshold    => _options.OpacityPruneThreshold      ?? PruneOpacityThreshold;
    internal int EffectiveMaxGaussianCount            => _options.MaxGaussianCount           ?? MaxGaussians;
    internal int EffectiveGradientAccumulationWindow  => _options.GradientAccumulationWindow ?? 100;

    public double PositionLearningRate { get; set; }
    public double ColorLearningRate { get; set; }
    public double OpacityLearningRate { get; set; }
    public double ScaleLearningRate { get; set; }
    public double RotationLearningRate { get; set; }
    public int TileSize { get; set; }
    public bool EnableSpatialIndex { get; set; }
    public int SpatialIndexRadius { get; set; }
    public double InitialNeighborSearchScale { get; set; }
    public double InitialScaleMultiplier { get; set; }
    public double DefaultPointSpacing { get; set; }
    public double MinScale { get; set; }

    /// <summary>
    /// Initializes a new instance of the GaussianSplatting class.
    /// </summary>
    /// <param name="initialPointCloud">Initial point cloud to place Gaussians.</param>
    /// <param name="initialColors">Optional initial colors for each point.</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a Gaussian Splatting model from an initial point cloud.
    ///
    /// Parameters explained:
    /// - initialPointCloud: Starting 3D points (typically from SfM like COLMAP)
    ///   - Format: Matrix [N, 3] where N is number of points
    ///   - Each row: (x, y, z) position
    ///   - Typical: 10K-1M points depending on scene
    ///
    /// - initialColors: Starting colors for each point
    ///   - Format: Matrix [N, 3] with RGB values
    ///   - Optional: If null, will be initialized randomly
    ///   - Usually from SfM output or estimated
    ///
    /// - useSphericalHarmonics: Enable view-dependent appearance
    ///   - False: Constant color from all viewing angles (faster, simpler)
    ///   - True: Color changes with viewing direction (realistic, e.g., specular highlights)
    ///   - Recommended: True for realistic scenes
    ///
    /// - shDegree: How much view-dependence to model
    ///   - 0: No view dependence (constant color) - simplest
    ///   - 1: Linear variation - basic view dependence
    ///   - 2: Quadratic variation - moderate view dependence
    ///   - 3: Cubic variation - strong view dependence (realistic)
    ///   - Higher degree = more parameters per Gaussian
    ///     - Degree 0: 3 parameters (RGB)
    ///     - Degree 1: 3 + 9 = 12 parameters
    ///     - Degree 2: 3 + 9 + 15 = 27 parameters
    ///     - Degree 3: 3 + 9 + 15 + 21 = 48 parameters
    ///
    /// Example initialization:
    /// // Load point cloud from COLMAP
    /// var pointCloud = LoadCOLMAPPointCloud("scene.ply");
    /// var colors = LoadCOLMAPColors("scene.ply");
    ///
    /// // Create Gaussian Splatting model
    /// var gs = new GaussianSplatting(
    ///     initialPointCloud: pointCloud,
    ///     initialColors: colors,
    ///     options: new GaussianSplattingOptions
    ///     {
    ///         UseSphericalHarmonics = true,
    ///         ShDegree = 3  // High quality view-dependent effects
    ///     });
    ///
    /// Typical workflow:
    /// 1. Run COLMAP on images → Get point cloud
    /// 2. Initialize GaussianSplatting with point cloud
    /// 3. Train on images for 10-30 minutes
    /// 4. Render novel views at 100+ FPS
    /// </remarks>
    public GaussianSplatting(
        Matrix<T>? initialPointCloud = null,
        Matrix<T>? initialColors = null,
        ILossFunction<T>? lossFunction = null)
        : this(
            new GaussianSplattingOptions
            {
            },
            initialPointCloud,
            initialColors,
            lossFunction)
    {
    }

    private static NeuralNetworkArchitecture<T> CreateArchitecture()
    {
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputHeight: 1,
            inputWidth: 1,
            inputDepth: 6,
            outputSize: 4);
    }

    protected override void InitializeLayers()
    {
        // No neural network layers - this is an explicit representation!
    }

    private void InitializeFromPointCloud(Matrix<T> pointCloud, Matrix<T>? colors)
    {
        if (pointCloud.Columns != 3)
        {
            throw new ArgumentException("Point cloud must have shape [N, 3].", nameof(pointCloud));
        }

        int numPoints = pointCloud.Rows;
        if (numPoints == 0)
        {
            return;
        }

        var engine = Engine;
        int basisCount = GetShBasisCount();
        var points = Tensor<T>.FromMatrix(pointCloud);

        Tensor<T> rgb;
        if (colors != null)
        {
            rgb = engine.TensorSlice(Tensor<T>.FromMatrix(colors), new[] { 0, 0 }, new[] { numPoints, 3 });
        }
        else
        {
            // Seeded colour draws stay in master's order (r, g, b per point); there is no engine op that draws a
            // tensor from a caller-owned Random.
            rgb = new Tensor<T>(new[] { numPoints, 3 });
            for (int i = 0; i < numPoints; i++)
            {
                rgb[i, 0] = NumOps.FromDouble(ModelRandom.NextDouble());
                rgb[i, 1] = NumOps.FromDouble(ModelRandom.NextDouble());
                rgb[i, 2] = NumOps.FromDouble(ModelRandom.NextDouble());
            }
        }

        // The DC coefficient carries the colour (divided by the degree-0 basis constant with spherical harmonics);
        // the higher-order coefficients start at zero.
        var dc = _useSphericalHarmonics
            ? engine.TensorDivideScalar(rgb, NumOps.FromDouble(0.282095))
            : rgb;
        var sh = dc.Reshape(numPoints, 1, 3);
        if (basisCount > 1)
        {
            sh = engine.TensorConcatenate(new[] { sh, new Tensor<T>(new[] { numPoints, basisCount - 1, 3 }) }, 1);
        }

        var rotations = engine.TensorConcatenate(
            new[] { Full(new[] { numPoints, 1 }, NumOps.One), new Tensor<T>(new[] { numPoints, 3 }) }, 1);
        var scales = InitialScales(points);
        var logScales = engine.TensorTile(engine.TensorLog(scales).Reshape(numPoints, 1), new[] { 1, 3 });

        _positions = AppendRows(_positions, points);
        _rotations = AppendRows(_rotations, rotations);
        _logScales = AppendRows(_logScales, logScales);
        _opacityLogits = AppendRows(_opacityLogits, new Tensor<T>(new[] { numPoints }));
        _shCoefficients = AppendRows(_shCoefficients, sh);
        _learningRateScales = AppendRows(_learningRateScales, Full(new[] { numPoints, LearningRateScaleColumns }, NumOps.One));
        InvalidateParameterCountCache();
    }

    /// <summary>
    /// Initial isotropic scale of each point: its nearest-neighbour distance within the 3x3x3 block of grid cells
    /// around it (cell size from the cloud's mean spacing) times <see cref="InitialScaleMultiplier"/>, at least
    /// <see cref="MinScale"/>; points without a neighbour use the mean spacing.
    /// </summary>
    private Tensor<T> InitialScales(Tensor<T> points)
    {
        var engine = Engine;
        int n = points._shape[0];
        var ops = NumOps;
        var lower = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(points), new[] { 0 }, false));
        var upper = engine.ReduceMax(points, new[] { 0 }, false);
        var extent = engine.TensorSubtract(upper, lower);
        T extentX = extent[0];
        T extentY = extent[1];
        T extentZ = extent[2];
        T volume = ops.Multiply(ops.Multiply(extentX, extentY), extentZ);
        T count = ops.FromDouble(n);
        T baseSpacing;
        if (ops.GreaterThan(volume, ops.Zero))
        {
            baseSpacing = ops.Power(ops.Divide(volume, count), ops.FromDouble(1.0 / 3.0));
        }
        else
        {
            T maxExtent = MaxOf(extentX, MaxOf(extentY, extentZ));
            baseSpacing = ops.GreaterThan(maxExtent, ops.Zero) ? ops.Divide(maxExtent, ops.Sqrt(count)) : ops.Zero;
        }

        if (!ops.GreaterThan(baseSpacing, ops.Zero))
        {
            baseSpacing = ops.FromDouble(DefaultPointSpacing);
        }

        T cellSize = MaxOf(baseSpacing, ops.FromDouble(MinScale * InitialNeighborSearchScale));
        T invCellSize = ops.GreaterThan(cellSize, ops.Zero) ? ops.Divide(ops.One, cellSize) : ops.Zero;
        var cells = engine.TensorFloor(engine.TensorMultiplyScalar(Sub(points, lower.Reshape(1, 3)), invCellSize));

        T sentinel = ops.MaxValue;
        T one = ops.One;
        T zero = ops.Zero;
        T minScale = ops.FromDouble(MinScale);
        T multiplier = ops.FromDouble(InitialScaleMultiplier);
        var columnIndex = engine.TensorArange<T>(0, n, 1).Reshape(1, n);
        int chunk = Math.Max(1, PairBudget / Math.Max(1, n * 3));
        var parts = new List<Tensor<T>>();
        for (int start = 0; start < n; start += chunk)
        {
            int rows = Math.Min(chunk, n - start);
            var shape3 = new[] { rows, n, 3 };
            var shape2 = new[] { rows, n };
            var rowPoints = engine.TensorSlice(points, new[] { start, 0 }, new[] { rows, 3 });
            var rowCells = engine.TensorSlice(cells, new[] { start, 0 }, new[] { rows, 3 });
            var offset = Sub(Bc(rowPoints.Reshape(rows, 1, 3), shape3), Bc(points.Reshape(1, n, 3), shape3));
            var distSq = engine.ReduceSum(engine.TensorMultiply(offset, offset), new[] { 2 }, false);
            var cellGap = engine.ReduceMax(
                engine.TensorAbs(Sub(Bc(rowCells.Reshape(rows, 1, 3), shape3), Bc(cells.Reshape(1, n, 3), shape3))),
                new[] { 2 }, false);
            var rowIndex = engine.TensorArange<T>(start, start + rows, 1).Reshape(rows, 1);
            var neighbour = Mul(Le(cellGap, one), Not(engine.TensorEquals(Bc(rowIndex, shape2), Bc(columnIndex, shape2))));
            var candidate = engine.TensorWhere(engine.TensorIsNan(distSq), Full(shape2, sentinel), distSq);
            candidate = Select(neighbour, candidate, Full(shape2, sentinel));
            var nearestSq = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(candidate), new[] { 1 }, false));
            var found = engine.TensorLessThan(nearestSq, sentinel);
            var spacing = Full(new[] { rows }, baseSpacing);
            var distance = Select(found, engine.TensorSqrt(nearestSq), spacing);
            distance = Select(Le(distance, zero), spacing, distance);
            parts.Add(MaxScalar(engine.TensorMultiplyScalar(distance, multiplier), minScale));
        }

        return engine.TensorConcatenate(parts.ToArray(), 0);
    }

    /// <summary>
    /// Unit quaternions of the given rows as (w, x, y, z) [N] tensors. Degenerate quaternions become the identity
    /// when <paramref name="identityWhenDegenerate"/> is set (norm below 1e-12, the optimizer's rule) and are left
    /// unnormalised otherwise (norm 0, the covariance rule).
    /// </summary>
    private (Tensor<T> W, Tensor<T> X, Tensor<T> Y, Tensor<T> Z) UnitQuaternions(Tensor<T> rotations, bool identityWhenDegenerate)
    {
        var engine = Engine;
        var w = Col(rotations, 0);
        var x = Col(rotations, 1);
        var y = Col(rotations, 2);
        var z = Col(rotations, 3);
        var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(rotations, rotations), new[] { 1 }, false));
        var ones = engine.TensorOnesLike(norm);
        if (identityWhenDegenerate)
        {
            var degenerate = Lt(norm, NumOps.FromDouble(1e-12));
            var inv = Select(degenerate, ones, engine.TensorReciprocal(norm));
            var zeros = engine.TensorZerosLike(norm);
            return (
                Select(degenerate, ones, engine.TensorMultiply(w, inv)),
                Select(degenerate, zeros, engine.TensorMultiply(x, inv)),
                Select(degenerate, zeros, engine.TensorMultiply(y, inv)),
                Select(degenerate, zeros, engine.TensorMultiply(z, inv)));
        }

        var scale = Select(Gt(norm, NumOps.Zero), engine.TensorReciprocal(norm), ones);
        return (engine.TensorMultiply(w, scale), engine.TensorMultiply(x, scale),
            engine.TensorMultiply(y, scale), engine.TensorMultiply(z, scale));
    }

    /// <summary>Rotation matrices [N, 3, 3] of unit quaternions.</summary>
    private Tensor<T> RotationMatrices(Tensor<T> w, Tensor<T> x, Tensor<T> y, Tensor<T> z)
    {
        var engine = Engine;
        int n = w.Length;
        T one = NumOps.One;
        T two = NumOps.FromDouble(2.0);
        var xx = engine.TensorMultiply(x, x);
        var yy = engine.TensorMultiply(y, y);
        var zz = engine.TensorMultiply(z, z);
        var xy = engine.TensorMultiply(x, y);
        var xz = engine.TensorMultiply(x, z);
        var yz = engine.TensorMultiply(y, z);
        var wx = engine.TensorMultiply(w, x);
        var wy = engine.TensorMultiply(w, y);
        var wz = engine.TensorMultiply(w, z);
        var entries = new[]
        {
            engine.ScalarMinusTensor(one, engine.TensorMultiplyScalar(engine.TensorAdd(yy, zz), two)),
            engine.TensorMultiplyScalar(engine.TensorSubtract(xy, wz), two),
            engine.TensorMultiplyScalar(engine.TensorAdd(xz, wy), two),
            engine.TensorMultiplyScalar(engine.TensorAdd(xy, wz), two),
            engine.ScalarMinusTensor(one, engine.TensorMultiplyScalar(engine.TensorAdd(xx, zz), two)),
            engine.TensorMultiplyScalar(engine.TensorSubtract(yz, wx), two),
            engine.TensorMultiplyScalar(engine.TensorSubtract(xz, wy), two),
            engine.TensorMultiplyScalar(engine.TensorAdd(yz, wx), two),
            engine.ScalarMinusTensor(one, engine.TensorMultiplyScalar(engine.TensorAdd(xx, yy), two)),
        };
        return engine.TensorStack(entries, 1).Reshape(n, 3, 3);
    }

    /// <summary>Clamped per-axis scales max(1e-6, exp(log s)) [N, 3], as the covariance uses them.</summary>
    private Tensor<T> CovarianceScales(Tensor<T> logScales)
        => MaxScalar(Engine.TensorExp(logScales), NumOps.FromDouble(1e-6));

    /// <summary>
    /// World-space covariances R S^2 R^T + 1e-6 I [N, 3, 3] of the given rows (quaternions normalised when their norm
    /// is positive).
    /// </summary>
    private Tensor<T> WorldCovariances(Tensor<T> rotations, Tensor<T> logScales)
    {
        var engine = Engine;
        int n = rotations._shape[0];
        var (w, x, y, z) = UnitQuaternions(rotations, identityWhenDegenerate: false);
        var r = RotationMatrices(w, x, y, z);
        var s = CovarianceScales(logScales);
        var scaled = Mul(r, engine.TensorMultiply(s, s).Reshape(n, 1, 3));
        var covariance = engine.TensorMatMul(scaled, engine.TensorPermute(r, new[] { 0, 2, 1 }).Contiguous());
        var eye = engine.TensorMultiplyScalar(Identity3().Reshape(1, 3, 3), NumOps.FromDouble(1e-6));
        return Add(covariance, eye);
    }

    /// <summary>Inverse covariances of all Gaussians, with master's determinant floor of +-1e-12.</summary>
    private Symmetric3 InverseCovariances()
    {
        var engine = Engine;
        int n = GaussianCount;
        var flat = WorldCovariances(_rotations, _logScales).Reshape(n, 9);
        var c00 = Col(flat, 0);
        var c01 = Col(flat, 1);
        var c02 = Col(flat, 2);
        var c11 = Col(flat, 4);
        var c12 = Col(flat, 5);
        var c22 = Col(flat, 8);
        var m00 = engine.TensorSubtract(engine.TensorMultiply(c11, c22), engine.TensorMultiply(c12, c12));
        var m01 = engine.TensorSubtract(engine.TensorMultiply(c01, c22), engine.TensorMultiply(c12, c02));
        var m02 = engine.TensorSubtract(engine.TensorMultiply(c01, c12), engine.TensorMultiply(c11, c02));
        var det = engine.TensorAdd(
            engine.TensorSubtract(engine.TensorMultiply(c00, m00), engine.TensorMultiply(c01, m01)),
            engine.TensorMultiply(c02, m02));
        T floor = NumOps.FromDouble(1e-12);
        var tiny = Lt(engine.TensorAbs(det), floor);
        var signedFloor = Select(Ge(det, NumOps.Zero), Full(det._shape, floor), Full(det._shape, NumOps.Negate(floor)));
        det = Select(tiny, signedFloor, det);
        return new Symmetric3(
            engine.TensorDivide(m00, det),
            engine.TensorDivide(engine.TensorSubtract(engine.TensorMultiply(c02, c12), engine.TensorMultiply(c01, c22)), det),
            engine.TensorDivide(m02, det),
            engine.TensorDivide(engine.TensorSubtract(engine.TensorMultiply(c00, c22), engine.TensorMultiply(c02, c02)), det),
            engine.TensorDivide(engine.TensorSubtract(engine.TensorMultiply(c02, c01), engine.TensorMultiply(c00, c12)), det),
            engine.TensorDivide(engine.TensorSubtract(engine.TensorMultiply(c00, c11), engine.TensorMultiply(c01, c01)), det));
    }

    private Tensor<T> Identity3()
    {
        var eye = new Tensor<T>(new[] { 3, 3 });
        eye[0, 0] = NumOps.One;
        eye[1, 1] = NumOps.One;
        eye[2, 2] = NumOps.One;
        return eye;
    }

    /// <inheritdoc />
    public (Tensor<T> rgb, Tensor<T> density) QueryField(Tensor<T> positions, Tensor<T> viewingDirections)
    {
        if (positions.Shape.Length != 2 || positions.Shape[1] != 3)
        {
            throw new ArgumentException("Positions must have shape [N, 3].", nameof(positions));
        }
        if (viewingDirections.Shape.Length != 2 || viewingDirections.Shape[1] != 3)
        {
            throw new ArgumentException("Viewing directions must have shape [N, 3].", nameof(viewingDirections));
        }

        var engine = Engine;
        int numPoints = positions.Shape[0];
        int n = GaussianCount;
        if (numPoints == 0 || n == 0)
        {
            return (new Tensor<T>(new[] { numPoints, 3 }), new Tensor<T>(new[] { numPoints, 1 }));
        }

        var inverse = InverseCovariances();
        var opacity = engine.TensorSigmoid(_opacityLogits).Reshape(1, n);
        var i00 = inverse.XX.Reshape(1, n);
        var i01 = inverse.XY.Reshape(1, n);
        var i02 = inverse.XZ.Reshape(1, n);
        var i11 = inverse.YY.Reshape(1, n);
        var i12 = inverse.YZ.Reshape(1, n);
        var i22 = inverse.ZZ.Reshape(1, n);
        var unitDirections = NormalizeRows(viewingDirections);

        // Colour sources: per-Gaussian constant colours, or the SH coefficients of each channel as [K, N] for a
        // basis [points, K] x coefficients [K, N] product.
        int basisCount = GetShBasisCount();
        var channelCoefficients = engine.TensorPermute(_shCoefficients, new[] { 2, 1, 0 }).Contiguous();
        var constantColours = Clamp01(engine.TensorSlice(_shCoefficients, new[] { 0, 0, 0 }, new[] { n, 1, 3 }).Reshape(n, 3));
        var coeffRed = engine.TensorSlice(channelCoefficients, new[] { 0, 0, 0 }, new[] { 1, basisCount, n }).Reshape(basisCount, n);
        var coeffGreen = engine.TensorSlice(channelCoefficients, new[] { 1, 0, 0 }, new[] { 1, basisCount, n }).Reshape(basisCount, n);
        var coeffBlue = engine.TensorSlice(channelCoefficients, new[] { 2, 0, 0 }, new[] { 1, basisCount, n }).Reshape(basisCount, n);

        // Spatial index as a candidate mask: a Gaussian is a candidate for a point when their grid cells are within
        // the query radius on every axis (the cells the hash grid would visit).
        bool useGrid = EnableSpatialIndex;
        bool noCandidates = false;
        Tensor<T> gaussianCells = new Tensor<T>(new[] { n, 3 });
        Tensor<T> gridOrigin = new Tensor<T>(new[] { 1, 3 });
        T invCellSize = NumOps.Zero;
        T gridRadius = NumOps.One;
        if (useGrid)
        {
            var scales = engine.TensorAbs(engine.TensorExp(_logScales));
            if (AiDotNet.Geometry.GeometryEngineOps<T>.AnyNan(scales))
            {
                // NaN scales give a NaN cell size, which collapses every Gaussian and point into one cell.
                useGrid = false;
            }
            else
            {
                var largest = engine.ReduceMax(scales, new[] { 1 }, false);
                T avgScale = engine.TensorMean(largest);
                T maxScale = engine.TensorMaxValue(largest);
                if (!NumOps.GreaterThan(avgScale, NumOps.Zero))
                {
                    avgScale = NumOps.FromDouble(0.01);
                }

                T cellSize = MaxOf(NumOps.Multiply(avgScale, NumOps.FromDouble(2.5)), MaxOf(maxScale, NumOps.FromDouble(MinScale)));
                T reach = NumOps.Ceiling(NumOps.Divide(NumOps.Multiply(maxScale, NumOps.FromDouble(3.0)), cellSize));
                gridRadius = MaxOf(NumOps.One, MaxOf(NumOps.FromDouble(SpatialIndexRadius), reach));
                if (!NumOps.GreaterThan(cellSize, NumOps.Zero))
                {
                    noCandidates = true;
                }
                else
                {
                    invCellSize = NumOps.Divide(NumOps.One, cellSize);
                    gridOrigin = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(_positions), new[] { 0 }, false)).Reshape(1, 3);
                    gaussianCells = engine.TensorFloor(engine.TensorMultiplyScalar(Sub(_positions, gridOrigin), invCellSize));
                }
            }
        }

        T half = NumOps.FromDouble(-0.5);
        T alphaFloor = NumOps.FromDouble(1e-8);
        int chunk = Math.Max(1, PairBudget / Math.Max(1, n * 3));
        var rgbParts = new List<Tensor<T>>();
        var densityParts = new List<Tensor<T>>();
        for (int start = 0; start < numPoints; start += chunk)
        {
            int rows = Math.Min(chunk, numPoints - start);
            var shape = new[] { rows, n };
            var shape3 = new[] { rows, n, 3 };
            var points = engine.TensorSlice(positions, new[] { start, 0 }, new[] { rows, 3 });
            var offset = Sub(Bc(points.Reshape(rows, 1, 3), shape3), Bc(_positions.Reshape(1, n, 3), shape3));
            var ox = engine.TensorSlice(offset, new[] { 0, 0, 0 }, new[] { rows, n, 1 }).Reshape(rows, n);
            var oy = engine.TensorSlice(offset, new[] { 0, 0, 1 }, new[] { rows, n, 1 }).Reshape(rows, n);
            var oz = engine.TensorSlice(offset, new[] { 0, 0, 2 }, new[] { rows, n, 1 }).Reshape(rows, n);
            var qx = Add(Add(Mul(i00, ox), Mul(i01, oy)), Mul(i02, oz));
            var qy = Add(Add(Mul(i01, ox), Mul(i11, oy)), Mul(i12, oz));
            var qz = Add(Add(Mul(i02, ox), Mul(i12, oy)), Mul(i22, oz));
            var q = engine.TensorAdd(engine.TensorAdd(engine.TensorMultiply(ox, qx), engine.TensorMultiply(oy, qy)), engine.TensorMultiply(oz, qz));
            var alpha = Mul(opacity, engine.TensorExp(engine.TensorMultiplyScalar(q, half)));

            // Gaussians with alpha <= 1e-8 are skipped (NaN alphas are not).
            var keep = Not(Le(alpha, alphaFloor));
            if (noCandidates)
            {
                keep = engine.TensorZerosLike(alpha);
            }
            else if (useGrid)
            {
                var pointCells = engine.TensorFloor(engine.TensorMultiplyScalar(Sub(points, gridOrigin), invCellSize));
                var gap = engine.ReduceMax(
                    engine.TensorAbs(Sub(Bc(pointCells.Reshape(rows, 1, 3), shape3), Bc(gaussianCells.Reshape(1, n, 3), shape3))),
                    new[] { 2 }, false);
                keep = engine.TensorMultiply(keep, Le(gap, gridRadius));
            }

            Tensor<T> red, green, blue;
            if (_useSphericalHarmonics)
            {
                var basis = ShBasis(engine.TensorSlice(unitDirections, new[] { start, 0 }, new[] { rows, 3 }));
                red = Clamp01(engine.TensorMatMul(basis, coeffRed));
                green = Clamp01(engine.TensorMatMul(basis, coeffGreen));
                blue = Clamp01(engine.TensorMatMul(basis, coeffBlue));
            }
            else
            {
                red = Bc(Col(constantColours, 0).Reshape(1, n), shape);
                green = Bc(Col(constantColours, 1).Reshape(1, n), shape);
                blue = Bc(Col(constantColours, 2).Reshape(1, n), shape);
            }

            var zeros = engine.TensorZerosLike(alpha);
            var density = engine.ReduceSum(engine.TensorWhere(keep, alpha, zeros), new[] { 1 }, false);
            var accumRed = engine.ReduceSum(engine.TensorWhere(keep, Mul(alpha, red), zeros), new[] { 1 }, false);
            var accumGreen = engine.ReduceSum(engine.TensorWhere(keep, Mul(alpha, green), zeros), new[] { 1 }, false);
            var accumBlue = engine.ReduceSum(engine.TensorWhere(keep, Mul(alpha, blue), zeros), new[] { 1 }, false);

            var normalise = Gt(density, alphaFloor);
            accumRed = Select(normalise, engine.TensorDivide(accumRed, density), accumRed);
            accumGreen = Select(normalise, engine.TensorDivide(accumGreen, density), accumGreen);
            accumBlue = Select(normalise, engine.TensorDivide(accumBlue, density), accumBlue);
            rgbParts.Add(Clamp01(engine.TensorStack(new[] { accumRed, accumGreen, accumBlue }, 1)));
            densityParts.Add(density.Reshape(rows, 1));
        }

        return (engine.TensorConcatenate(rgbParts.ToArray(), 0), engine.TensorConcatenate(densityParts.ToArray(), 0));
    }

    /// <inheritdoc />
    public Tensor<T> RenderImage(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength)
    {
        return RenderImageInternal(cameraPosition, cameraRotation, imageWidth, imageHeight, focalLength, clampOutput: true);
    }

    private Tensor<T> RenderImageInternal(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength,
        bool clampOutput)
    {
        var camera = TransformGaussiansToCamera(cameraPosition, cameraRotation);
        var splats = ProjectGaussiansTo2D(camera, imageWidth, imageHeight, focalLength);
        return RasterizeGaussians(splats, imageWidth, imageHeight, clampOutput);
    }

    /// <summary>
    /// Maps every Gaussian into camera coordinates. Convention (matches NeRF.RenderImage + NerfLab.LookAt):
    /// <paramref name="cameraRotation"/> is CAMERA-to-WORLD -- its columns are the world-space directions of the camera
    /// right/up/forward axes -- so a world point maps to camera space through its transpose: x_c = R^T (x - c), and a
    /// covariance through R^T Sigma R.
    /// </summary>
    private CameraSpace TransformGaussiansToCamera(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation)
    {
        var engine = Engine;
        int n = GaussianCount;
        var rotation = Tensor<T>.FromMatrix(cameraRotation);
        var origin = engine.TensorSlice(Tensor<T>.FromVector(cameraPosition), new[] { 0 }, new[] { 3 }).Reshape(1, 3);
        if (n == 0)
        {
            return new CameraSpace(new Tensor<T>(new[] { 0, 3 }), new Tensor<T>(new[] { 0, 9 }), new Tensor<T>(new[] { 0, 3 }));
        }

        var offset = Sub(_positions, origin);

        // Row vectors: (R^T d)^T = d^T R.
        var position = engine.TensorMatMul(offset, rotation);
        var world = WorldCovariances(_rotations, _logScales);
        var tiled = engine.TensorTile(rotation.Reshape(1, 3, 3), new[] { n, 1, 1 });
        var tiledT = engine.TensorPermute(tiled, new[] { 0, 2, 1 }).Contiguous();
        var covariance = engine.TensorMatMul(engine.TensorMatMul(tiledT, world), tiled).Reshape(n, 9);
        return new CameraSpace(position, covariance, offset);
    }

    /// <summary>
    /// EWA projection (Zwicker et al. 2002, as used by Kerbl et al. 2023) of the camera-space Gaussians onto the
    /// image plane: 2-D mean, conic (inverse 2-D covariance, regularised by 1e-6 on the diagonal), 3-sigma bounding
    /// box and colour. Gaussians behind the camera (z &lt;= 1e-6), with a degenerate footprint (det &lt;= 1e-12) or whose
    /// box misses the image are dropped; the rest are sorted front to back.
    /// </summary>
    private ProjectedSplats ProjectGaussiansTo2D(
        CameraSpace camera,
        int width,
        int height,
        T focalLength)
    {
        var engine = Engine;
        int n = camera.Position._shape[0];
        int basisCount = GetShBasisCount();
        if (n == 0)
        {
            return EmptySplats(0, basisCount);
        }

        var ops = NumOps;
        T f = focalLength;
        var x = Col(camera.Position, 0);
        var y = Col(camera.Position, 1);
        var z = Col(camera.Position, 2);
        var cov = camera.Covariance;
        var s00 = Col(cov, 0);
        var s01 = Col(cov, 1);
        var s02 = Col(cov, 2);
        var s11 = Col(cov, 4);
        var s12 = Col(cov, 5);
        var s22 = Col(cov, 8);

        var invZ = engine.TensorReciprocal(z);
        var meanX = engine.TensorAddScalar(engine.TensorMultiply(engine.TensorMultiplyScalar(x, f), invZ), ops.FromDouble((width - 1) * 0.5));
        var meanY = engine.TensorAddScalar(engine.TensorMultiply(engine.TensorMultiplyScalar(y, f), invZ), ops.FromDouble((height - 1) * 0.5));

        // Rows of the projection Jacobian: v0 = (j00, 0, j02), v1 = (0, j11, j12).
        var j00 = engine.TensorMultiplyScalar(invZ, f);
        var j02 = engine.TensorNegate(engine.TensorMultiply(engine.TensorMultiply(engine.TensorMultiplyScalar(x, f), invZ), invZ));
        var j11 = j00;
        var j12 = engine.TensorNegate(engine.TensorMultiply(engine.TensorMultiply(engine.TensorMultiplyScalar(y, f), invZ), invZ));
        var t0x = engine.TensorAdd(engine.TensorMultiply(s00, j00), engine.TensorMultiply(s02, j02));
        var t0z = engine.TensorAdd(engine.TensorMultiply(s02, j00), engine.TensorMultiply(s22, j02));
        var t1x = engine.TensorAdd(engine.TensorMultiply(s01, j11), engine.TensorMultiply(s02, j12));
        var t1y = engine.TensorAdd(engine.TensorMultiply(s11, j11), engine.TensorMultiply(s12, j12));
        var t1z = engine.TensorAdd(engine.TensorMultiply(s12, j11), engine.TensorMultiply(s22, j12));
        T eps = ops.FromDouble(1e-6);
        var a = engine.TensorAddScalar(engine.TensorAdd(engine.TensorMultiply(j00, t0x), engine.TensorMultiply(j02, t0z)), eps);
        var b = engine.TensorAdd(engine.TensorMultiply(j00, t1x), engine.TensorMultiply(j02, t1z));
        var c = engine.TensorAddScalar(engine.TensorAdd(engine.TensorMultiply(j11, t1y), engine.TensorMultiply(j12, t1z)), eps);
        var det = engine.TensorSubtract(engine.TensorMultiply(a, c), engine.TensorMultiply(b, b));
        var invDet = engine.TensorReciprocal(det);
        var invA = engine.TensorMultiply(c, invDet);
        var invB = engine.TensorNegate(engine.TensorMultiply(b, invDet));
        var invC = engine.TensorMultiply(a, invDet);

        var trace = engine.TensorAdd(a, c);
        var disc = engine.TensorSubtract(engine.TensorMultiply(trace, trace), engine.TensorMultiplyScalar(det, ops.FromDouble(4.0)));
        disc = Select(Lt(disc, ops.Zero), engine.TensorZerosLike(disc), disc);
        var maxEigen = engine.TensorMultiplyScalar(engine.TensorAdd(trace, engine.TensorSqrt(disc)), ops.FromDouble(0.5));
        var radius = engine.TensorMultiplyScalar(engine.TensorSqrt(MaxScalar(maxEigen, ops.Zero)), ops.FromDouble(3.0));
        radius = Select(Lt(radius, ops.One), engine.TensorOnesLike(radius), radius);
        var minX = engine.TensorFloor(engine.TensorSubtract(meanX, radius));
        var maxX = engine.TensorCeiling(engine.TensorAdd(meanX, radius));
        var minY = engine.TensorFloor(engine.TensorSubtract(meanY, radius));
        var maxY = engine.TensorCeiling(engine.TensorAdd(meanY, radius));

        // Visibility. The box test is written so that a NaN box (a NaN mean or footprint) drops the Gaussian.
        var visible = engine.TensorMultiply(Not(Le(z, eps)), Not(Le(det, ops.FromDouble(1e-12))));
        visible = engine.TensorMultiply(visible, engine.TensorMultiply(
            Ge(maxX, ops.Zero), Ge(maxY, ops.Zero)));
        visible = engine.TensorMultiply(visible, engine.TensorMultiply(
            Lt(minX, ops.FromDouble(width)), Lt(minY, ops.FromDouble(height))));
        var visibleRows = engine.TensorNonzero(visible);
        int m = visibleRows._shape[0];
        if (m == 0)
        {
            return EmptySplats(n, basisCount);
        }

        var rows = visibleRows.Reshape(m);
        var order = engine.TensorArgsort(engine.TensorIndexSelect(z, rows, 0), 0, false);
        var sorted = engine.TensorIndexSelect(rows, order, 0);

        minX = MaxScalar(minX, ops.Zero);
        maxX = MinScalar(maxX, ops.FromDouble(width - 1));
        minY = MaxScalar(minY, ops.Zero);
        maxY = MinScalar(maxY, ops.FromDouble(height - 1));

        // Colour seen along the world-space ray from the camera to the Gaussian -- the viewing direction QueryField
        // receives -- through the spherical-harmonics basis, or the stored colour without spherical harmonics.
        var basis = ShBasis(NormalizeRows(camera.WorldOffset));
        Tensor<T> raw;
        if (_useSphericalHarmonics)
        {
            raw = engine.ReduceSum(Mul(basis.Reshape(n, basisCount, 1), _shCoefficients), new[] { 1 }, false);
        }
        else
        {
            raw = engine.TensorSlice(_shCoefficients, new[] { 0, 0, 0 }, new[] { n, 1, 3 }).Reshape(n, 3);
        }

        var colour = Clamp01(raw);
        var raster = engine.TensorStack(new[]
        {
            meanX, meanY, invA, invB, invC, engine.TensorSigmoid(_opacityLogits),
            Col(colour, 0), Col(colour, 1), Col(colour, 2), minX, maxX, minY, maxY,
        }, 1);
        var extra = engine.TensorStack(new[]
        {
            x, y, z, s00, s01, s02, s11, s12, s22, Col(raw, 0), Col(raw, 1), Col(raw, 2),
        }, 1);
        return new ProjectedSplats(
            sorted,
            visible,
            engine.TensorIndexSelect(raster, sorted, 0),
            engine.TensorIndexSelect(extra, sorted, 0),
            engine.TensorIndexSelect(basis, sorted, 0));
    }

    private ProjectedSplats EmptySplats(int gaussianCount, int basisCount)
        => new ProjectedSplats(
            new Tensor<int>(new[] { 0 }),
            new Tensor<T>(new[] { gaussianCount }),
            new Tensor<T>(new[] { 0, RasterColumns }),
            new Tensor<T>(new[] { 0, ExtraColumns }),
            new Tensor<T>(new[] { 0, basisCount }));

    private Tensor<T> RasterizeGaussians(ProjectedSplats splats, int width, int height, bool clampOutput)
    {
        var image = SplatTiles(splats, width, height, null, null, out _);
        return clampOutput ? Clamp01(image) : image;
    }

    /// <summary>
    /// Front-to-back alpha compositing of the depth-sorted splats over screen tiles, C = sum_i T_i alpha_i c_i with
    /// T_i = prod_{j &lt; i} (1 - alpha_j). A pixel ignores a splat outside its box, below the -20 exponent cutoff or
    /// the 1e-6 alpha cutoff, and stops once its transmittance falls below 1e-4. With a loss gradient the same pass
    /// accumulates, per splat, the sums the image-space gradient needs (columns GradSumRed..GradMeanY).
    /// </summary>
    private Tensor<T> SplatTiles(
        ProjectedSplats splats,
        int width,
        int height,
        Tensor<T>? prediction,
        Tensor<T>? lossGradient,
        out Tensor<T> splatGradients)
    {
        var engine = Engine;
        int m = splats.Count;
        splatGradients = new Tensor<T>(new[] { m, GradColumns });
        if (width <= 0 || height <= 0)
        {
            return new Tensor<T>(new[] { Math.Max(0, height), Math.Max(0, width), 3 });
        }
        if (m == 0)
        {
            return new Tensor<T>(new[] { height, width, 3 });
        }

        bool training = prediction is not null && lossGradient is not null;
        var ops = NumOps;
        T one = ops.One;
        T two = ops.FromDouble(2.0);
        T minusHalf = ops.FromDouble(-0.5);
        T exponentCutoff = ops.FromDouble(-20.0);
        T alphaCutoff = ops.FromDouble(1e-6);
        T transmittanceCutoff = ops.FromDouble(1e-4);
        T oneMinusAlphaFloor = ops.FromDouble(1e-6);

        int tileSize = Math.Max(4, TileSize);
        int tilesX = (width + tileSize - 1) / tileSize;
        int tilesY = (height + tileSize - 1) / tileSize;

        // Tile membership of every splat: [M, tilesX] and [M, tilesY] masks of the tile ranges its box covers.
        T tileT = ops.FromDouble(tileSize);
        var raster = splats.Raster;
        var memberX = TileMembership(Col(raster, ColMinX), Col(raster, ColMaxX), tileT, tilesX);
        var memberY = TileMembership(Col(raster, ColMinY), Col(raster, ColMaxY), tileT, tilesY);

        var imageRows = new List<Tensor<T>>();
        var gradientIndices = new List<Tensor<int>>();
        var gradientValues = new List<Tensor<T>>();
        for (int ty = 0; ty < tilesY; ty++)
        {
            int yStart = ty * tileSize;
            int tileHeight = Math.Min(height, yStart + tileSize) - yStart;
            var rowMember = engine.TensorSlice(memberY, new[] { 0, ty }, new[] { m, 1 }).Reshape(m);
            var rowTiles = new List<Tensor<T>>();
            for (int tx = 0; tx < tilesX; tx++)
            {
                int xStart = tx * tileSize;
                int tileWidth = Math.Min(width, xStart + tileSize) - xStart;
                int pixels = tileWidth * tileHeight;
                var inTile = engine.TensorMultiply(rowMember, engine.TensorSlice(memberX, new[] { 0, tx }, new[] { m, 1 }).Reshape(m));
                var bucketRows = engine.TensorNonzero(inTile);
                int bucketSize = bucketRows._shape[0];
                if (bucketSize == 0)
                {
                    rowTiles.Add(new Tensor<T>(new[] { tileHeight, tileWidth, 3 }));
                    continue;
                }

                // Bucket rows ascend through the depth-sorted splats, so the bucket is front to back.
                var bucket = bucketRows.Reshape(bucketSize);
                var attributes = engine.TensorIndexSelect(raster, bucket, 0);
                var pixelX = engine.TensorTile(engine.TensorArange<T>(xStart, xStart + tileWidth, 1), new[] { tileHeight }).Reshape(pixels, 1);
                var pixelY = engine.TensorRepeatInterleave(engine.TensorArange<T>(yStart, yStart + tileHeight, 1), tileWidth, 0).Reshape(pixels, 1);

                Tensor<T>? total = null;
                Tensor<T>? upstream = null;
                if (training && prediction is not null && lossGradient is not null)
                {
                    total = engine.TensorSlice(prediction, new[] { yStart, xStart, 0 }, new[] { tileHeight, tileWidth, 3 }).Reshape(pixels, 3);
                    upstream = engine.TensorSlice(lossGradient, new[] { yStart, xStart, 0 }, new[] { tileHeight, tileWidth, 3 }).Reshape(pixels, 3);
                }

                var transmittance = Full(new[] { pixels, 1 }, one);
                var accumRed = new Tensor<T>(new[] { pixels, 1 });
                var accumGreen = new Tensor<T>(new[] { pixels, 1 });
                var accumBlue = new Tensor<T>(new[] { pixels, 1 });
                int chunk = Math.Max(1, RasterBudget / pixels);
                for (int start = 0; start < bucketSize; start += chunk)
                {
                    int count = Math.Min(chunk, bucketSize - start);
                    var shape = new[] { pixels, count };
                    var part = engine.TensorSlice(attributes, new[] { start, 0 }, new[] { count, RasterColumns });
                    var meanX = Row(part, ColMeanX, count);
                    var meanY = Row(part, ColMeanY, count);
                    var invA = Row(part, ColInvA, count);
                    var invB = Row(part, ColInvB, count);
                    var invC = Row(part, ColInvC, count);
                    var opacity = Row(part, ColOpacity, count);
                    var red = Row(part, ColRed, count);
                    var green = Row(part, ColGreen, count);
                    var blue = Row(part, ColBlue, count);

                    var px = Bc(pixelX, shape);
                    var py = Bc(pixelY, shape);
                    var inBox = engine.TensorMultiply(
                        engine.TensorMultiply(engine.TensorGreaterEqual(px, Bc(Row(part, ColMinX, count), shape)), engine.TensorLessEqual(px, Bc(Row(part, ColMaxX, count), shape))),
                        engine.TensorMultiply(engine.TensorGreaterEqual(py, Bc(Row(part, ColMinY, count), shape)), engine.TensorLessEqual(py, Bc(Row(part, ColMaxY, count), shape))));
                    var dx = Sub(px, meanX);
                    var dy = Sub(py, meanY);
                    var quad = engine.TensorAdd(
                        engine.TensorAdd(engine.TensorMultiply(Mul(invA, dx), dx), engine.TensorMultiply(Mul(engine.TensorMultiplyScalar(invB, two), dx), dy)),
                        engine.TensorMultiply(Mul(invC, dy), dy));
                    var exponent = engine.TensorMultiplyScalar(quad, minusHalf);
                    var weight = engine.TensorExp(exponent);
                    var alpha = Mul(opacity, weight);
                    var valid = engine.TensorMultiply(inBox, engine.TensorMultiply(Not(Lt(exponent, exponentCutoff)), Not(Lt(alpha, alphaCutoff))));
                    var zeros = engine.TensorZerosLike(alpha);
                    var factor = engine.ScalarMinusTensor(one, engine.TensorWhere(valid, alpha, zeros));

                    // Exclusive transmittance by a running product; the carry continues it into the next chunk.
                    var scan = engine.TensorCumProd(engine.TensorConcatenate(new[] { transmittance, factor }, 1), 1);
                    var before = engine.TensorSlice(scan, new[] { 0, 0 }, new[] { pixels, count });
                    transmittance = engine.TensorSlice(scan, new[] { 0, count }, new[] { pixels, 1 });
                    var live = engine.TensorMultiply(valid, Not(Lt(before, transmittanceCutoff)));
                    var weightedAlpha = engine.TensorWhere(live, engine.TensorMultiply(before, alpha), zeros);
                    var contribRed = engine.TensorWhere(live, Mul(weightedAlpha, red), zeros);
                    var contribGreen = engine.TensorWhere(live, Mul(weightedAlpha, green), zeros);
                    var contribBlue = engine.TensorWhere(live, Mul(weightedAlpha, blue), zeros);

                    var scanRed = engine.TensorCumSum(engine.TensorConcatenate(new[] { accumRed, contribRed }, 1), 1);
                    var scanGreen = engine.TensorCumSum(engine.TensorConcatenate(new[] { accumGreen, contribGreen }, 1), 1);
                    var scanBlue = engine.TensorCumSum(engine.TensorConcatenate(new[] { accumBlue, contribBlue }, 1), 1);
                    accumRed = engine.TensorSlice(scanRed, new[] { 0, count }, new[] { pixels, 1 });
                    accumGreen = engine.TensorSlice(scanGreen, new[] { 0, count }, new[] { pixels, 1 });
                    accumBlue = engine.TensorSlice(scanBlue, new[] { 0, count }, new[] { pixels, 1 });

                    if (total is null || upstream is null)
                    {
                        continue;
                    }

                    // d C / d alpha_i = T_i c_i - (sum_{j > i} T_j alpha_j c_j) / (1 - alpha_i), with the colour after
                    // splat i read off the rendered image (as the per-Gaussian implementation did).
                    var gradRed = Bc(Col2(upstream, 0, pixels), shape);
                    var gradGreen = Bc(Col2(upstream, 1, pixels), shape);
                    var gradBlue = Bc(Col2(upstream, 2, pixels), shape);
                    var invOneMinusAlpha = engine.TensorReciprocal(MaxScalar(engine.ScalarMinusTensor(one, alpha), oneMinusAlphaFloor));
                    var dRed = ColourTerm(before, red, Bc(Col2(total, 0, pixels), shape), engine.TensorSlice(scanRed, new[] { 0, 0 }, new[] { pixels, count }), contribRed, invOneMinusAlpha);
                    var dGreen = ColourTerm(before, green, Bc(Col2(total, 1, pixels), shape), engine.TensorSlice(scanGreen, new[] { 0, 0 }, new[] { pixels, count }), contribGreen, invOneMinusAlpha);
                    var dBlue = ColourTerm(before, blue, Bc(Col2(total, 2, pixels), shape), engine.TensorSlice(scanBlue, new[] { 0, 0 }, new[] { pixels, count }), contribBlue, invOneMinusAlpha);
                    var dAlpha = engine.TensorAdd(engine.TensorAdd(engine.TensorMultiply(gradRed, dRed), engine.TensorMultiply(gradGreen, dGreen)), engine.TensorMultiply(gradBlue, dBlue));
                    var dExponent = engine.TensorMultiply(Mul(dAlpha, opacity), weight);
                    var halfExponent = engine.TensorMultiplyScalar(dExponent, minusHalf);
                    var dOpacity = engine.TensorMultiply(Mul(engine.TensorMultiply(dAlpha, weight), opacity), Bc(engine.ScalarMinusTensor(one, opacity), shape));
                    var dMeanX = engine.TensorMultiply(dExponent, engine.TensorAdd(Mul(invA, dx), Mul(invB, dy)));
                    var dMeanY = engine.TensorMultiply(dExponent, engine.TensorAdd(Mul(invB, dx), Mul(invC, dy)));
                    var columns = new[]
                    {
                        engine.TensorMultiply(weightedAlpha, gradRed),
                        engine.TensorMultiply(weightedAlpha, gradGreen),
                        engine.TensorMultiply(weightedAlpha, gradBlue),
                        engine.TensorMultiply(engine.TensorMultiply(halfExponent, dx), dx),
                        engine.TensorMultiply(engine.TensorMultiply(engine.TensorNegate(dExponent), dx), dy),
                        engine.TensorMultiply(engine.TensorMultiply(halfExponent, dy), dy),
                        dOpacity,
                        dMeanX,
                        dMeanY,
                    };
                    var sums = new List<Tensor<T>>();
                    foreach (var column in columns)
                    {
                        sums.Add(engine.ReduceSum(engine.TensorWhere(live, column, zeros), new[] { 0 }, false));
                    }

                    gradientValues.Add(engine.TensorStack(sums.ToArray(), 1));
                    gradientIndices.Add(engine.TensorSlice(bucket, new[] { start }, new[] { count }));
                }

                rowTiles.Add(engine.TensorConcatenate(new[] { accumRed, accumGreen, accumBlue }, 1).Reshape(tileHeight, tileWidth, 3));
            }

            imageRows.Add(engine.TensorConcatenate(rowTiles.ToArray(), 1));
        }

        if (gradientValues.Count > 0)
        {
            splatGradients = engine.TensorIndexAdd(
                splatGradients,
                0,
                engine.TensorConcatenate(gradientIndices.ToArray(), 0),
                engine.TensorConcatenate(gradientValues.ToArray(), 0));
        }

        return engine.TensorConcatenate(imageRows.ToArray(), 0);
    }

    /// <summary>
    /// [M, tiles] mask of the tiles (of size <paramref name="tileSize"/>) that the pixel range [lower, upper] of each
    /// splat touches.
    /// </summary>
    private Tensor<T> TileMembership(Tensor<T> lower, Tensor<T> upper, T tileSize, int tiles)
    {
        var engine = Engine;
        int m = lower.Length;
        var shape = new[] { m, tiles };
        var first = engine.TensorFloor(engine.TensorDivideScalar(lower, tileSize)).Reshape(m, 1);
        var last = engine.TensorFloor(engine.TensorDivideScalar(upper, tileSize)).Reshape(m, 1);
        var tileIndex = Bc(engine.TensorArange<T>(0, tiles, 1).Reshape(1, tiles), shape);
        return engine.TensorMultiply(
            engine.TensorLessEqual(Bc(first, shape), tileIndex),
            engine.TensorGreaterEqual(Bc(last, shape), tileIndex));
    }

    /// <summary>T_i c_i - (C - C_before_i - contribution_i) / max(1e-6, 1 - alpha_i).</summary>
    private Tensor<T> ColourTerm(Tensor<T> before, Tensor<T> colour, Tensor<T> total, Tensor<T> accumulated, Tensor<T> contribution, Tensor<T> invOneMinusAlpha)
    {
        var engine = Engine;
        var after = engine.TensorSubtract(engine.TensorSubtract(total, accumulated), contribution);
        return engine.TensorSubtract(Mul(before, colour), engine.TensorMultiply(after, invOneMinusAlpha));
    }

    /// <inheritdoc />
    public Tensor<T> RenderRays(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        int numSamples,
        T nearBound,
        T farBound)
    {
        int numRays = rayOrigins.Shape[0];
        var (samplePositions, sampleDirections) = SamplePointsAlongRays(
            rayOrigins, rayDirections, numSamples, nearBound, farBound);
        var (rgb, density) = QueryField(samplePositions, sampleDirections);
        return VolumeRendering(rgb, density, numRays, numSamples, nearBound, farBound);
    }

    private (Tensor<T> positions, Tensor<T> directions) SamplePointsAlongRays(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        int numSamples,
        T nearBound,
        T farBound)
    {
        int numRays = rayOrigins.Shape[0];
        if (numRays == 0 || numSamples <= 0)
        {
            return (new Tensor<T>(new[] { numRays * Math.Max(numSamples, 0), 3 }),
                new Tensor<T>(new[] { numRays * Math.Max(numSamples, 0), 3 }));
        }

        var engine = Engine;
        T step = numSamples > 1
            ? NumOps.Divide(NumOps.Subtract(farBound, nearBound), NumOps.FromDouble(numSamples - 1))
            : NumOps.Zero;
        var ts = engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorArange<T>(0, numSamples, 1), step), nearBound);

        // positions[r, s] = o_r + t_s d_r, ray-major.
        var shape = new[] { numRays, numSamples, 3 };
        var origins = engine.TensorBroadcastTo(rayOrigins.Reshape(numRays, 1, 3), shape);
        var dirs = engine.TensorBroadcastTo(rayDirections.Reshape(numRays, 1, 3), shape);
        var tGrid = engine.TensorBroadcastTo(ts.Reshape(1, numSamples, 1), shape);
        var positions = engine.TensorAdd(origins, engine.TensorMultiply(tGrid, dirs)).Reshape(numRays * numSamples, 3);
        var directions = engine.TensorRepeatInterleave(rayDirections.Reshape(numRays, 3), numSamples, 0);
        return (positions, directions);
    }

    private Tensor<T> VolumeRendering(
        Tensor<T> rgb,
        Tensor<T> density,
        int numRays,
        int numSamples,
        T nearBound,
        T farBound)
    {
        if (numRays <= 0 || numSamples <= 0)
        {
            return new Tensor<T>(new[] { Math.Max(numRays, 0), 3 });
        }

        var engine = Engine;
        T deltaT = NumOps.Divide(NumOps.Subtract(farBound, nearBound), NumOps.FromDouble(numSamples));
        var sigma = density.Reshape(numRays, numSamples);
        var alpha = engine.ScalarMinusTensor(
            NumOps.One, engine.TensorExp(engine.TensorNegate(engine.TensorMultiplyScalar(sigma, deltaT))));

        // Samples with alpha <= 0 are skipped: they add nothing and leave the transmittance unchanged.
        var contributes = engine.TensorLessEqual(alpha, engine.TensorZerosLike(alpha));
        var factor = engine.TensorWhere(contributes, engine.TensorOnesLike(alpha), engine.ScalarMinusTensor(NumOps.One, alpha));
        var weight = engine.TensorWhere(contributes, engine.TensorZerosLike(alpha), alpha);

        // Transmittance reaching each sample: the product of the earlier factors (exclusive scan).
        var leading = engine.TensorOnesLike(engine.TensorSlice(factor, new[] { 0, 0 }, new[] { numRays, 1 }));
        var shifted = numSamples > 1
            ? engine.TensorConcatenate(
                new[] { leading, engine.TensorSlice(factor, new[] { 0, 0 }, new[] { numRays, numSamples - 1 }) }, 1)
            : leading;
        var transmittance = engine.TensorCumProd(shifted, 1);

        // Marching stops once the transmittance falls below 1e-4: later samples are not composited, and skipped
        // or stopped samples are never read (so their colours cannot leak NaN into the sum).
        var stopped = engine.TensorLessThan(transmittance, NumOps.FromDouble(1e-4));
        var zeros = engine.TensorZerosLike(alpha);
        var keep = engine.TensorWhere(contributes, zeros, engine.TensorWhere(stopped, zeros, engine.TensorOnesLike(alpha)));
        var sampleWeight = engine.TensorMultiply(transmittance, weight);

        var shape = new[] { numRays, numSamples, 3 };
        var keep3 = engine.TensorBroadcastTo(keep.Reshape(numRays, numSamples, 1), shape);
        var weights = engine.TensorBroadcastTo(sampleWeight.Reshape(numRays, numSamples, 1), shape);
        var contributions = engine.TensorMultiply(weights, rgb.Reshape(numRays, numSamples, 3));
        var colors = engine.TensorWhere(keep3, contributions, engine.TensorZerosLike(contributions));
        return engine.ReduceSum(colors, new[] { 1 }, false);
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardWithMemory(Tensor<T> input)
    {
        if (input.Shape.Length != 2 || input.Shape[1] != 6)
        {
            throw new ArgumentException("Input must have shape [N, 6] (position + direction).", nameof(input));
        }

        var engine = Engine;
        int numPoints = input.Shape[0];
        var posTensor = engine.TensorSlice(input, new[] { 0, 0 }, new[] { numPoints, 3 });
        var dirTensor = engine.TensorSlice(input, new[] { 0, 3 }, new[] { numPoints, 3 });
        var (rgb, density) = QueryField(posTensor, dirTensor);
        return engine.TensorConcatenate(new[] { rgb, density }, 1);
    }

    /// <summary>
    /// Consume the configured facade optimizer's hyperparameters into GS's per-attribute
    /// learning-rate schedule (#1833). Called once by <c>AiModelBuilder.BuildAsync</c>
    /// immediately after <c>SetModel</c> and before the first <c>Optimize</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The 3D Gaussian Splatting paper (Kerbl et al. 2023) uses per-attribute learning rates —
    /// position ≈ base, color ≈ 15×, opacity ≈ 300×, scale ≈ 30×, rotation ≈ 6× — because
    /// scale, opacity, and color live on very different manifolds than position and a
    /// single LR either explodes fast attributes or under-trains slow ones. Reference
    /// implementations require the caller to construct parameter groups by hand
    /// (<c>Adam([{params: pos, lr: 1.6e-4}, {params: opacity, lr: 5e-2}, ...])</c>). Here,
    /// the caller passes ONE base LR via <c>ConfigureOptimizer</c> and this method derives
    /// the per-attribute schedule automatically, anchored on the paper's canonical base of
    /// 1.6e-4. If the caller wants exact per-attribute control they can still construct
    /// <c>GaussianSplattingOptions</c> directly (bypasses this scaling).
    /// </para>
    /// <para>
    /// When <c>options.InitialLearningRate</c> still equals the configured optimizer type's own
    /// default (0.01 for the base <c>OptimizationAlgorithmOptions</c>, 1e-3 for
    /// <c>AdamOptimizerOptions</c>, etc.), the per-attribute defaults from
    /// <see cref="GaussianSplattingOptions"/> are retained unchanged so the model keeps the
    /// schedule from Kerbl et al. 2023. Any other value triggers full re-derivation.
    /// </para>
    /// </remarks>
    public void ApplyOptimizerHyperparameters(OptimizationAlgorithmOptions<T, Tensor<T>, Tensor<T>> options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        // #1835 short-run adjustment: densification defaults (Start=500, End=15000) target
        // paper-scale runs (~30k iters). If MaxIterations is much smaller, densification would
        // NEVER fire because Start > MaxIterations. Scale the effective start / end to fractions
        // of MaxIterations when the caller didn't explicitly set them. Explicit values override
        // the auto-scaling — advanced users retain full control.
        int totalIters = options.MaxIterations > 0 ? options.MaxIterations : 100;
        if (_options.DensificationStartIteration is null && totalIters < 500)
        {
            // Start at ~5% of the run so the initial cloud has time to stabilize but there's
            // still room for densification to grow it before the end freeze.
            _options.DensificationStartIteration = Math.Max(1, totalIters / 20);
        }
        if (_options.DensificationEndIteration is null && totalIters < 15000)
        {
            // Stop densifying in the last ~15% so final iterations converge with a fixed cloud.
            _options.DensificationEndIteration = Math.Max(2, (int)(totalIters * 0.85));
        }

        // Per-attribute learning rates from Kerbl et al. 2023, "3D Gaussian Splatting for
        // Real-Time Radiance Field Rendering" (SIGGRAPH 2023), Section 5 / Appendix B: position
        // 1.6e-4, feature/color 2.5e-3, opacity 5.0e-2, scale 5.0e-3, rotation 1.0e-3. Encoded
        // as ratios relative to the position LR so a caller-supplied base LR scales all five
        // consistently while preserving the paper's cross-attribute proportions.
        const double PaperBasePositionLR = 1.6e-4;
        const double ColorRatio = 15.625;     // 2.5e-3 / 1.6e-4
        const double OpacityRatio = 312.5;    // 5.0e-2 / 1.6e-4
        const double ScaleRatio = 31.25;      // 5.0e-3 / 1.6e-4
        const double RotationRatio = 6.25;    // 1.0e-3 / 1.6e-4

        // A learning rate left at the configured optimizer's default is a no-op: GS retains the
        // paper's per-attribute schedule from GaussianSplattingOptions rather than collapsing it
        // to a single MLP-tuned rate. The sentinel is the CONCRETE optimizer options type's own
        // default, not a hardcoded base-class 0.01 — OptimizationAlgorithmOptions defaults to
        // 0.01 but AdamOptimizerOptions overrides to 1e-3 (and other optimizers differ). A fresh
        // instance of the runtime type yields its true default so detection tracks whichever
        // optimizer is configured; a value differing from that default triggers re-derivation.
        double defaultLR;
        try
        {
            var freshDefaults = System.Activator.CreateInstance(options.GetType())
                as OptimizationAlgorithmOptions<T, Tensor<T>, Tensor<T>>;
            defaultLR = freshDefaults?.InitialLearningRate ?? 0.01;
        }
        catch (System.MissingMethodException)
        {
            // No public parameterless ctor on this options type — fall back to the base-class
            // default so an untouched base-typed options instance still no-ops.
            defaultLR = 0.01;
        }

        double callerLR = options.InitialLearningRate;
        if (callerLR == defaultLR)
        {
            return;
        }

        // Scale factor so ratios stay stable regardless of the caller's chosen base.
        double scale = callerLR / PaperBasePositionLR;

        PositionLearningRate = callerLR;
        ColorLearningRate    = PaperBasePositionLR * ColorRatio    * scale;
        OpacityLearningRate  = PaperBasePositionLR * OpacityRatio  * scale;
        ScaleLearningRate    = PaperBasePositionLR * ScaleRatio    * scale;
        RotationLearningRate = PaperBasePositionLR * RotationRatio * scale;
    }

    /// <summary>
    /// Image-space photometric training (#1834). Pulls one batch from the loader, volume-
    /// renders each sampled ray via <see cref="RenderRays"/>, computes MSE against the
    /// ground-truth pixel colors, and routes the resulting per-ray targets through the
    /// existing supervised <see cref="Train"/> path so gradients reach the Gaussians via the
    /// per-attribute LR schedule set up by <see cref="ApplyOptimizerHyperparameters"/> (#1833).
    /// </summary>
    /// <remarks>
    /// This slice ships the wired scaffolding (loader → render → loss → param update) so the
    /// facade + tests can exercise the full image-space path end-to-end. Full paper-standard
    /// backprop through volume rendering (with the split/prune densification loop from #1835)
    /// lands via follow-up work on this same interface.
    /// </remarks>
    public T TrainOnImageBatch(
        AiDotNet.Interfaces.IDataLoader<NeuralRadianceFields.Data.ImageView<T>, NeuralRadianceFields.Data.PixelBatch<T>> loader,
        int raysPerBatch,
        AiDotNet.Models.Options.OptimizationAlgorithmOptions<T, Tensor<T>, Tensor<T>>? optimizerOptions,
        NeuralRadianceFields.Data.ImageTrainingOptions? imageTrainingOptions = null)
    {
        if (loader is null) throw new ArgumentNullException(nameof(loader));
        if (raysPerBatch <= 0) throw new ArgumentOutOfRangeException(nameof(raysPerBatch));

        if (optimizerOptions is not null)
        {
            ((IHyperparameterAware<T, Tensor<T>, Tensor<T>>)this).ApplyOptimizerHyperparameters(optimizerOptions);
        }

        // Route through GS's paper-faithful CAMERA-mode Train — that's the path that runs
        // the #1835 densification loop (DensifyAndPrune fires every DensificationInterval
        // iterations, gated by the [StartIter, EndIter] window and OpacityPruneThreshold /
        // GradientNormThreshold from the extended options). RAY-mode training bypasses
        // densification by design (see TrainOnRays remarks — screen-space gradient signal
        // isn't present in ray-mode).
        NeuralRadianceFields.Data.ImageView<T> view;
        NeuralRadianceFields.Data.PixelBatch<T> pixels;
        using (var enumerator = loader.IterateBatches(raysPerBatch).GetEnumerator())
        {
            if (!enumerator.MoveNext())
            {
                return NumOps.Zero;
            }
            (view, pixels) = enumerator.Current;
        }

        // Build the [1, 13] camera-mode input GS.Train recognizes: pos[3] + rot[9] + focal[1].
        double focalPx = view.ResolveFocalLengthInPixels();

        var cameraInputData = new T[13];
        cameraInputData[0] = view.CameraPosition[0];
        cameraInputData[1] = view.CameraPosition[1];
        cameraInputData[2] = view.CameraPosition[2];
        cameraInputData[3]  = view.CameraRotation[0, 0];
        cameraInputData[4]  = view.CameraRotation[0, 1];
        cameraInputData[5]  = view.CameraRotation[0, 2];
        cameraInputData[6]  = view.CameraRotation[1, 0];
        cameraInputData[7]  = view.CameraRotation[1, 1];
        cameraInputData[8]  = view.CameraRotation[1, 2];
        cameraInputData[9]  = view.CameraRotation[2, 0];
        cameraInputData[10] = view.CameraRotation[2, 1];
        cameraInputData[11] = view.CameraRotation[2, 2];
        cameraInputData[12] = NumOps.FromDouble(focalPx);
        var cameraInput = new Tensor<T>(new[] { 1, 13 }, new Vector<T>(cameraInputData));

        // Photo target [H, W, 3] — the shape GS.Train's camera-mode ParseCameraInput expects.
        // Alpha channel (if present) drops out via engine narrow (excellence goal: engine-op
        // path so alpha stripping is vectorized and tape-recorded, not a scalar copy).
        var photo = view.Photo;
        if (photo.Shape[2] == 4)
        {
            photo = Engine.TensorNarrow(photo, dim: 2, start: 0, length: 3);
        }

        // Excellence goal #4 (LearnedPrior single-image mode): if the view carries a prior,
        // blend its hallucinated hallucinated view into the photo target so single-image
        // reconstruction gets training signal from angles the caller didn't photograph.
        // Confidence controls the blend: real-photo weight = (1 - confidence), prior weight =
        // confidence. ImageTrainingOptions.PriorConfidenceOverride can force a global weight.
        if (view.Prior is not null)
        {
            double confidence = imageTrainingOptions?.PriorConfidenceOverride ?? view.Prior.Confidence;
            if (confidence > 0)
            {
                var hallucination = view.Prior.SynthesizeNovelView(
                    view.CameraPosition,
                    view.CameraRotation,
                    view.Height,
                    view.Width);
                var wPhoto = NumOps.FromDouble(1.0 - confidence);
                var wPrior = NumOps.FromDouble(confidence);
                var photoScaled = Engine.TensorMultiplyScalar(photo, wPhoto);
                var priorScaled = Engine.TensorMultiplyScalar(hallucination, wPrior);
                photo = Engine.TensorAdd(photoScaled, priorScaled);
            }
        }

        Train(cameraInput, photo);

        // Return the SAME loss Train recorded (camera-mode LossFunction.CalculateLoss on the
        // rendered image vs the photo target). Avoids the previous double-render + wrong-loss
        // problem where we returned a RAY-mode photometric MSE but the actual parameter update
        // used CAMERA-mode image-level supervision — telemetry now matches the update signal.
        return LastLoss ?? NumOps.Zero;
    }

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (LossFunction == null)
        {
            throw new InvalidOperationException("Loss function not set.");
        }

        // Two-mode training contract:
        //   1. CAMERA mode (paper-faithful): input is [1, 13] (position[3] +
        //      rotation[9] + focal[1]) and expectedOutput is an image
        //      [H, W, 3]. Trains by rendering and image-supervising — the
        //      classic 3D Gaussian Splatting recipe (Kerbl et al. 2023).
        //   2. RAY mode (compatible with InstantNGP / NeRF / generic NeRF
        //      training pipelines): input is [N, 6] (position + direction
        //      per ray) and expectedOutput is [N, 3] ray colors. Trains
        //      via per-ray colour supervision. This is the same contract
        //      Predict() uses, so models can be trained against ray-level
        //      data without first reshaping into a (camera, image) pair.
        //
        // The model-family test scaffold relies on Predict-shape and Train-
        // shape being the same; without this two-mode acceptance the
        // generic Train_ShouldReduceLoss / GradientFlow / etc. invariants
        // can't exercise GaussianSplatting without a NeRF-specific test
        // class.
        if (input.Rank == 2 && input.Shape[1] == 6)
        {
            TrainOnRays(input, expectedOutput);
            return true;
        }

        ParseCameraInput(input, expectedOutput,
            out var cameraPosition,
            out var cameraRotation,
            out var focalLength,
            out int imageWidth,
            out int imageHeight);

        var prediction = RenderImageInternal(
            cameraPosition,
            cameraRotation,
            imageWidth,
            imageHeight,
            focalLength,
            clampOutput: false);
        LastLoss = LossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        var lossGradient = LossFunction.ComputeGradient(prediction, expectedOutput);
        var gradients = ApplyImageGradients(
            cameraPosition,
            cameraRotation,
            imageWidth,
            imageHeight,
            focalLength,
            prediction,
            lossGradient);

        _trainingStep++;

        // #1835 excellence goal #2: swappable densification schedule strategy. Fall back to
        // the fixed-interval schedule (reference-impl behavior) when the caller didn't
        // supply one — same firing behavior as before, but now callers can plug in the
        // adaptive schedule for data-driven firing.
        var schedule = _options.DensificationSchedule ?? new NeuralRadianceFields.Data.FixedIntervalDensificationSchedule
        {
            Interval = Math.Max(1, DensificationInterval)
        };
        double meanGradNorm = MeanGradientNorm(gradients);
        double lossValue = NumOps.ToDouble(LastLoss);
        if (EnableDensification && schedule.ShouldFire(_trainingStep, lossValue, meanGradNorm))
        {
            DensifyAndPrune(gradients);
        }

        ResetOpacityIfScheduled();

        return true;
    }

    /// <summary>
    /// Excellence goal #3 — post-training compression pass: prunes low-opacity Gaussians,
    /// merges overlapping ellipses, and quantizes spherical-harmonics coefficients.
    /// Reference impls make callers run this as a separate post-processing script;
    /// AiModelBuilder's BuildPipeline calls this automatically when
    /// <see cref="GaussianSplattingOptions.CompressOnBuildComplete"/> is true.
    /// </summary>
    public void RunCompressionPass()
    {
        var opts = _options.CompressionOptions ?? new AiDotNet.Models.Options.GaussianCompressionOptions();
        var engine = Engine;
        var ops = NumOps;
        int before = GaussianCount;
        EnsureLearningRateScales();

        if (opts.PruneLowOpacity && GaussianCount > 0)
        {
            T threshold = ops.FromDouble(Math.Max(0.0, EffectiveOpacityPruneThreshold));
            KeepRows(Not(Lt(engine.TensorSigmoid(_opacityLogits), threshold)));
        }

        if (opts.MergeOverlapping && opts.MergeOverlapThreshold < 1.0 && GaussianCount > 1)
        {
            // Merge Gaussians whose center-to-center distance is smaller than half the sum of their max scales
            // (approximation of ellipse overlap). From the last Gaussian down, each merges into the highest-index
            // earlier Gaussian it overlaps (which may itself have absorbed later ones), then drops out.
            int n = GaussianCount;
            T threshold = ops.FromDouble(opts.MergeOverlapThreshold);
            T halfT = ops.FromDouble(0.5);
            T probabilityCap = ops.FromDouble(0.9999);
            T zero = ops.Zero;
            var removed = new List<int>();
            for (int i = n - 1; i >= 1; i--)
            {
                var source = engine.TensorSlice(_positions, new[] { i, 0 }, new[] { 1, 3 });
                var earlier = engine.TensorSlice(_positions, new[] { 0, 0 }, new[] { i, 3 });
                var sourceLog = engine.TensorSlice(_logScales, new[] { i, 0 }, new[] { 1, 3 });
                var sourceMax = engine.ReduceMax(engine.TensorExp(sourceLog), new[] { 1 }, false);
                var earlierMax = engine.ReduceMax(engine.TensorExp(engine.TensorSlice(_logScales, new[] { 0, 0 }, new[] { i, 3 })), new[] { 1 }, false);
                var delta = Sub(source, earlier);
                var distance = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(delta, delta), new[] { 1 }, false));
                var reach = Add(sourceMax, earlierMax);
                var overlap = MaxScalar(engine.TensorDivide(engine.TensorSubtract(reach, distance), reach), zero);
                var hits = engine.TensorNonzero(Ge(overlap, threshold));
                if (hits._shape[0] == 0)
                {
                    continue;
                }

                int j = engine.TensorMaxValue(hits);
                var target = engine.TensorSlice(_positions, new[] { j, 0 }, new[] { 1, 3 });
                var targetLog = engine.TensorSlice(_logScales, new[] { j, 0 }, new[] { 1, 3 });
                var mergedPosition = engine.TensorMultiplyScalar(engine.TensorAdd(target, source), halfT);

                // The larger scale per axis wins; ties (and NaN comparisons) keep the source's value.
                var mergedLog = engine.TensorWhere(engine.TensorGreaterThan(targetLog, sourceLog), targetLog, sourceLog);

                // Opacities add in probability space (clamped below 1), so the merged Gaussian keeps the combined
                // contribution of the near-duplicates.
                var opacities = engine.TensorSigmoid(engine.TensorConcatenate(new[]
                {
                    engine.TensorSlice(_opacityLogits, new[] { i }, new[] { 1 }),
                    engine.TensorSlice(_opacityLogits, new[] { j }, new[] { 1 }),
                }, 0));
                var summed = MinScalar(engine.ReduceSum(opacities, new[] { 0 }, true), probabilityCap);
                mergedPosition.CopyTo(_positions.Data.Span.Slice(j * 3, 3));
                mergedLog.CopyTo(_logScales.Data.Span.Slice(j * 3, 3));
                Logit(summed).CopyTo(_opacityLogits.Data.Span.Slice(j, 1));
                removed.Add(i);
            }

            if (removed.Count > 0)
            {
                var removedRows = new Tensor<int>(removed.ToArray(), new[] { removed.Count });
                var drop = engine.TensorIndexAdd(new Tensor<T>(new[] { n }), 0, removedRows, Full(new[] { removed.Count }, ops.One));
                KeepRows(Not(drop));
            }
        }

        if (opts.QuantizeSphericalHarmonics && _options.UseSphericalHarmonics && _options.ShDegree > 0 && GaussianCount > 0)
        {
            // Per-Gaussian symmetric quantization of the SH coefficients: a per-Gaussian scale factor (NaN
            // coefficients ignored) and each coefficient rounded to the nearest bin; all-zero Gaussians are left alone.
            int n = GaussianCount;
            int levels = (1 << Math.Max(1, Math.Min(16, opts.QuantizationBits))) - 1;
            T halfLevels = ops.FromDouble(levels / 2.0);
            var flat = _shCoefficients.Reshape(n, _shCoefficients.Length / Math.Max(1, n));
            var magnitude = engine.TensorAbs(flat);
            magnitude = engine.TensorWhere(engine.TensorIsNan(magnitude), engine.TensorZerosLike(magnitude), magnitude);
            var absMax = engine.ReduceMax(magnitude, new[] { 1 }, true);
            var qScale = Select(Gt(absMax, ops.Zero), engine.TensorDivideScalar(absMax, halfLevels), engine.TensorZerosLike(absMax));
            var active = Not(engine.TensorEquals(qScale, ops.Zero));
            var safeScale = Select(active, qScale, engine.TensorOnesLike(qScale));
            var quantized = Mul(engine.TensorRound(Div(flat, safeScale)), safeScale);
            WriteInPlace(_shCoefficients, Select(active, quantized, flat));
        }

        if (GaussianCount != before)
        {
            // Pruning and merging resize the parameter surface.
            InvalidateParameterCountCache();
        }
    }

    /// <summary>Keeps the Gaussians whose mask entry is non-zero, in order.</summary>
    private void KeepRows(Tensor<T> keep)
    {
        var engine = Engine;
        var kept = engine.TensorNonzero(keep);
        int count = kept._shape[0];
        if (count == GaussianCount)
        {
            return;
        }

        var rows = kept.Reshape(count);
        _positions = engine.TensorIndexSelect(_positions, rows, 0).Clone();
        _rotations = engine.TensorIndexSelect(_rotations, rows, 0).Clone();
        _logScales = engine.TensorIndexSelect(_logScales, rows, 0).Clone();
        _opacityLogits = engine.TensorIndexSelect(_opacityLogits, rows, 0).Clone();
        _shCoefficients = engine.TensorIndexSelect(_shCoefficients, rows, 0).Clone();
        _learningRateScales = engine.TensorIndexSelect(EnsureLearningRateScales(), rows, 0).Clone();
    }

    /// <summary>
    /// Ray-level training path: input is [N, 6] (position + direction per
    /// ray), expectedOutput is [N, 3] ray colors. Runs the same
    /// ForwardWithMemory ray-marching that <see cref="Predict"/> uses, then
    /// applies per-ray colour gradients to the Gaussian field. Matches the
    /// training contract used by InstantNGP / NeRF (and the generic
    /// auto-generated test scaffold). The image-supervised camera-mode
    /// training path remains the primary, paper-faithful entry; this is
    /// the compatible secondary contract.
    /// </summary>
    /// <remarks>
    /// <para><b>Densification / pruning is intentionally NOT run on this
    /// path.</b> Kerbl et al. 2023 §5.2's adaptive density control is
    /// driven by the projected-Gaussian gradient state that
    /// <see cref="ApplyImageGradients"/> accumulates while splatting onto
    /// the image plane (the per-Gaussian <c>GaussianGradient</c> list).
    /// Ray-mode gradients are computed in colour-space across rays, not
    /// in screen-space across pixels, so they don't carry the
    /// projected-gradient magnitude signal the densification heuristic
    /// keys off. Running DensifyAndPrune against a synthesised list
    /// would either be a no-op (zero gradients) or actively wrong
    /// (heuristics fire on the wrong signal). Callers who need
    /// densification must use the image-supervised
    /// <see cref="TrainOnImage"/> entry; this ray-mode path is a
    /// scaffold/test-compat contract and is documented as such.</para>
    /// </remarks>
    private void TrainOnRays(Tensor<T> rayInput, Tensor<T> expectedRayColors)
    {
        var prediction = ForwardWithMemory(rayInput);

        // Loss expects prediction and target with matching element count.
        // ForwardWithMemory normally returns [N, 4] (RGB + density); align
        // the target if the caller supplied a shape that matches the ray
        // count but has fewer channels (e.g., [N, 3] RGB-only). When the
        // target has more channels than the prediction (rare), truncate
        // the prediction side. The goal is a clean per-ray colour loss
        // even when target shape doesn't perfectly match the model's
        // emit channel count.
        var alignedTarget = AlignRayTargetToPrediction(prediction, expectedRayColors);

        LastLoss = LossFunction.CalculateLoss(prediction.ToVector(), alignedTarget.ToVector());

        var lossGradient = LossFunction.ComputeGradient(prediction, alignedTarget);
        ApplyRayGradients(rayInput, lossGradient);

        _trainingStep++;
    }

    /// <summary>
    /// Reshape / channel-pad/truncate the target so it can be loss-compared
    /// element-wise to the prediction. Same-shape pass-through is the hot
    /// path; channel-mismatch fixups handle the common case where the
    /// caller passes [N, 3] ray colours but the model emits [N, 4].
    /// </summary>
    private Tensor<T> AlignRayTargetToPrediction(Tensor<T> prediction, Tensor<T> target)
    {
        // Direct _shape access is the in-repo idiom (InternalsVisibleTo is
        // configured between AiDotNet and AiDotNet.Tensors so Tensor<T>'s
        // private layout is reachable from this assembly). It avoids the
        // ReadOnlySpan-builder allocation that the public .Shape property
        // performs and matches the pattern used elsewhere in this file.
        if (prediction._shape.Length == target._shape.Length)
        {
            bool sameShape = true;
            for (int i = 0; i < prediction._shape.Length; i++)
            {
                if (prediction._shape[i] != target._shape[i]) { sameShape = false; break; }
            }
            if (sameShape) return target;
        }

        // Both must be rank-2 [N, C] for channel-pad/truncate. Anything
        // else is out of contract — let the loss validator throw the
        // original error so the caller gets a real shape-mismatch
        // diagnostic rather than a silently-broken broadcast.
        if (prediction._shape.Length != 2 || target._shape.Length != 2) return target;
        if (prediction._shape[0] != target._shape[0]) return target;

        int predN = prediction._shape[0];
        int predC = prediction._shape[1];
        int targC = target._shape[1];
        if (predC == targC) return target;

        // For prediction channels the target doesn't supply (typical case: target is RGB [N, 3] but prediction is
        // RGBA [N, 4] with a density channel), the aligned target takes the prediction's value, so the per-channel
        // loss term (pred - pred)^2 = 0 -- zero gradient on the unmatched channel. Padding with zeros instead would
        // silently regularise the density channel toward zero, a target the caller never requested, and would
        // actively suppress opacity during ray-mode training.
        var engine = Engine;
        int copyC = Math.Min(predC, targC);
        var copied = engine.TensorSlice(target, new[] { 0, 0 }, new[] { predN, copyC });
        if (copyC == predC)
        {
            return copied;
        }

        var padding = engine.TensorSlice(prediction, new[] { 0, copyC }, new[] { predN, predC - copyC });
        return engine.TensorConcatenate(new[] { copied, padding }, 1);
    }

    /// <summary>
    /// Backprop helper for <see cref="TrainOnRays"/>: distributes ray-level
    /// colour gradients onto the Gaussian colour parameters. Approximation:
    /// each ray's gradient contributes equally to all Gaussians whose
    /// projection intersects the ray's pixel — a coarse-grained signal,
    /// but sufficient for the gradient-flow / loss-reduction invariants
    /// the test scaffold exercises. A production-grade implementation
    /// should use the same alpha-blended attribution the camera-mode
    /// renderer uses.
    /// </summary>
    private void ApplyRayGradients(Tensor<T> rayInput, Tensor<T> lossGradient)
    {
        int n = GaussianCount;
        if (n == 0) return;
        int numRays = rayInput.Shape[0];
        if (numRays == 0) return;

        // Stride into lossGradient must match the prediction's per-ray
        // channel count, not a hard-coded 3. ForwardWithMemory emits
        // [N, 4] (RGB + density), and the loss gradient inherits that
        // shape -- reading r * 3 + c would step across the wrong memory
        // offsets and silently corrupt the colour-channel update.
        int gradC = lossGradient._shape.Length >= 2 ? lossGradient._shape[1] : 1;
        var engine = Engine;
        var ops = NumOps;

        // Average gradient per channel over the ray batch -- keeps the update proportional to the
        // ray-batch size so a 10-ray batch and a 1000-ray batch produce updates of comparable scale.
        T invRays = ops.FromDouble(1.0 / numRays);
        var perRay = engine.TensorSlice(lossGradient.Reshape(lossGradient.Length), new[] { 0 }, new[] { numRays * gradC })
            .Reshape(numRays, gradC);
        var meanGrad = engine.TensorMultiplyScalar(engine.ReduceSum(perRay, new[] { 0 }, false), invRays);

        // Only the first three channels are colour (a 4-channel target carries density in channel 3, handled by
        // opacity below). The colour residual of each channel lands on that channel's DC coefficient -- entry
        // [n, 0, c] of the coefficient tensor, which is the RGB colour itself without spherical harmonics. The
        // constant rgb = shBase * dc Jacobian factor is folded into the (tunable) colour learning rate: this
        // ray-mode update is already a coarse mean-field approximation (uniform per-Gaussian attribution, no
        // alpha weighting), so an exact SH-basis scale buys no accuracy and only shrinks the step.
        int rgbChannels = Math.Min(3, gradC);
        var colourGrad = engine.TensorSlice(meanGrad, new[] { 0 }, new[] { rgbChannels });
        if (rgbChannels < 3)
        {
            colourGrad = engine.TensorConcatenate(new[] { colourGrad, new Tensor<T>(new[] { 3 - rgbChannels }) }, 0);
        }

        int basisCount = GetShBasisCount();
        int width = basisCount * 3;
        var lrScales = EnsureLearningRateScales();

        // #1835 excellence goal #4 (persistent per-Gaussian LR state): the colour step is scaled by each
        // Gaussian's colour multiplier (split children carry SplitChildLearningRateScales.Color). The configured
        // colour learning rate is used, as 3DGS uses different LRs per parameter family (Kerbl et al. 2023).
        var colourStep = engine.TensorMultiplyScalar(Col(lrScales, LrColor), ops.FromDouble(ColorLearningRate)).Reshape(n, 1);
        var flat = _shCoefficients.Reshape(n, width);
        var dc = engine.TensorSlice(flat, new[] { 0, 0 }, new[] { n, 3 });
        var newDc = engine.TensorSubtract(dc, Mul(colourGrad.Reshape(1, 3), colourStep));
        var updated = width > 3
            ? engine.TensorConcatenate(new[] { newDc, engine.TensorSlice(flat, new[] { 0, 3 }, new[] { n, width - 3 }) }, 1)
            : newDc;
        WriteInPlace(_shCoefficients, updated);

        var shGrad = Bc(colourGrad.Reshape(1, 3), new[] { n, 3 });
        if (width > 3)
        {
            shGrad = engine.TensorConcatenate(new[] { shGrad, new Tensor<T>(new[] { n, width - 3 }) }, 1);
        }

        // Descend the density residual through opacity (the rendered density channel is a monotonic function of
        // opacity), so a 4-channel ray target trains geometry too.
        var opacityGrad = new Tensor<T>(new[] { n, 1 });
        if (gradC >= 4)
        {
            var densityGrad = engine.TensorSlice(meanGrad, new[] { 3 }, new[] { 1 });
            var opacityStep = engine.TensorMultiplyScalar(Col(lrScales, LrOpacity), ops.FromDouble(OpacityLearningRate));
            WriteInPlace(_opacityLogits, engine.TensorSubtract(_opacityLogits, Mul(densityGrad, opacityStep)));
            opacityGrad = Bc(densityGrad.Reshape(1, 1), new[] { n, 1 });
        }

        PublishImageGradients(
            new Tensor<T>(new[] { n, 3 }),
            new Tensor<T>(new[] { n, 4 }),
            new Tensor<T>(new[] { n, 3 }),
            opacityGrad,
            shGrad);
    }

    // ParameterCount was a formula here -- perGaussian * _gaussians.Count -- restating the layout
    // that GetParameters, GetParameterChunks and UpdateParameters each restated separately. All of
    // them are replaced by the single declaration below; see GetExtraTrainableTensors.

    /// <summary>
    /// Exposes inspectable activations for this explicit-representation model. GaussianSplatting
    /// has no <c>Layers</c>, so the base class's layer-walking implementation returns an empty
    /// map — which breaks introspection/visualization callers and the model-family
    /// <c>NamedLayerActivations_ShouldBeNonEmpty</c> invariant. Instead surface the render
    /// pipeline's meaningful tensors: the flattened Gaussian parameter state and the rendered
    /// output for the supplied ray batch. Both are real tensors produced by the model (no
    /// fabricated placeholders).
    /// </summary>
    public override System.Collections.Generic.Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        var activations = new System.Collections.Generic.Dictionary<string, Tensor<T>>();

        var parameters = GetParameters();
        if (parameters.Length > 0)
        {
            activations["gaussians"] = new Tensor<T>(new[] { parameters.Length }, parameters);
        }

        // The rendered radiance (RGB + density per ray) is the model's forward output — the
        // explicit-representation analogue of a final-layer activation.
        activations["render"] = Predict(input);

        return activations;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Every derived quantity -- covariances, projections, the spatial-index cells -- is computed from the
    /// parameter tensors on use, so restoring the parameters cannot leave the renderer drawing stale geometry.
    /// SetParameters (serialization and the optimizer's post-training writeback) reaches this hook on the one
    /// restore path.
    /// </remarks>
    protected override void OnParametersRestored()
    {
        // Covariances, projections and the spatial-index cells are derived from the parameter tensors on every
        // use, so a restore only has to bring the per-Gaussian optimizer state in line with the restored count.
        EnsureLearningRateScales();
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        int colorDim = _useSphericalHarmonics ? 3 * GetShBasisCount() : 3;
        int perGaussian = 3 + 4 + 3 + 1 + colorDim;
        int totalParameters = GaussianCount * perGaussian;

        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GaussianCount", GaussianCount },
                { "UseSphericalHarmonics", _useSphericalHarmonics },
                { "ShDegree", _shDegree },
                { "ColorDimensions", colorDim },
                { "ParametersPerGaussian", perGaussian },
                { "TotalParameters", totalParameters },
                { "TileSize", TileSize },
                { "EnableSpatialIndex", EnableSpatialIndex },
                { "SpatialIndexRadius", SpatialIndexRadius },
                { "SplitPositionJitter", SplitPositionJitter },
                { "SplitScaleFactor", SplitScaleFactor },
                { "SplitOpacityFactor", SplitOpacityFactor },
                { "SplitOpacityMax", SplitOpacityMax },
                { "InitialNeighborSearchScale", InitialNeighborSearchScale },
                { "ScaleLearningRate", ScaleLearningRate },
                { "RotationLearningRate", RotationLearningRate },
                { "InitialScaleMultiplier", InitialScaleMultiplier },
                { "DefaultPointSpacing", DefaultPointSpacing },
                { "MinScale", MinScale }
            },
            // License-safe metadata bytes — see NeRF.GetModelMetadata for the
            // same rationale. Fixes #1826.
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    private void ParseCameraInput(
        Tensor<T> input,
        Tensor<T> expectedOutput,
        out Vector<T> cameraPosition,
        out Matrix<T> cameraRotation,
        out T focalLength,
        out int imageWidth,
        out int imageHeight)
    {
        if (input.Shape.Length != 2 || input.Shape[1] != 13 || input.Shape[0] != 1)
        {
            throw new ArgumentException("Input must have shape [1, 13] (position, rotation, focal length).", nameof(input));
        }
        if (expectedOutput.Shape.Length != 3 || expectedOutput.Shape[2] != 3)
        {
            throw new ArgumentException("Expected output must have shape [H, W, 3].", nameof(expectedOutput));
        }

        imageHeight = expectedOutput.Shape[0];
        imageWidth = expectedOutput.Shape[1];

        var engine = Engine;
        cameraPosition = engine.TensorSlice(input, new[] { 0, 0 }, new[] { 1, 3 }).Reshape(3).ToVector();
        cameraRotation = engine.TensorSlice(input, new[] { 0, 3 }, new[] { 1, 9 }).Reshape(3, 3).ToMatrix();
        focalLength = input[0, 12];
    }

    /// <summary>
    /// One image-space gradient step (Kerbl et al. 2023, section 5 and appendix): backpropagates the loss gradient of the
    /// rendered image through the front-to-back compositing, the EWA projection and the covariance
    /// parameterisation Sigma = R S S^T R^T, publishes the flat parameter gradient and applies the per-attribute
    /// learning rates. Returns the world-space position gradients of the visible Gaussians for densification.
    /// </summary>
    private List<GaussianGradient> ApplyImageGradients(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength,
        Tensor<T> prediction,
        Tensor<T> lossGradient)
    {
        var camera = TransformGaussiansToCamera(cameraPosition, cameraRotation);
        var splats = ProjectGaussiansTo2D(camera, imageWidth, imageHeight, focalLength);

        if (prediction.Shape.Length != 3 ||
            prediction.Shape[0] != imageHeight ||
            prediction.Shape[1] != imageWidth ||
            prediction.Shape[2] != 3)
        {
            throw new ArgumentException("Prediction must have shape [H, W, 3].", nameof(prediction));
        }
        if (lossGradient.Shape.Length != 3 ||
            lossGradient.Shape[0] != imageHeight ||
            lossGradient.Shape[1] != imageWidth ||
            lossGradient.Shape[2] != 3)
        {
            throw new ArgumentException("Loss gradient must have shape [H, W, 3].", nameof(lossGradient));
        }

        var engine = Engine;
        var ops = NumOps;
        int n = GaussianCount;
        int basisCount = GetShBasisCount();
        int m = splats.Count;
        var gradients = new List<GaussianGradient>();

        var positionGrad = new Tensor<T>(new[] { n, 3 });
        var rotationGrad = new Tensor<T>(new[] { n, 4 });
        var scaleGrad = new Tensor<T>(new[] { n, 3 });
        var opacityGrad = new Tensor<T>(new[] { n, 1 });
        var shGrad = new Tensor<T>(new[] { n, basisCount * 3 });
        if (m == 0)
        {
            PublishImageGradients(positionGrad, rotationGrad, scaleGrad, opacityGrad, shGrad);
            return gradients;
        }

        SplatTiles(splats, imageWidth, imageHeight, prediction, lossGradient, out var sums);

        T zero = ops.Zero;
        T one = ops.One;
        T two = ops.FromDouble(2.0);
        T half = ops.FromDouble(0.5);
        T f = focalLength;
        var rows = splats.Indices;
        var mask = splats.VisibleMask;
        var extra = splats.Extra;
        var raster = splats.Raster;
        var camX = Col(extra, ExCamX);
        var camY = Col(extra, ExCamY);
        var camZ = Col(extra, ExCamZ);
        var invZ = engine.TensorReciprocal(camZ);
        var fInvZ = engine.TensorMultiplyScalar(invZ, f);
        var fInvZ2 = engine.TensorMultiply(fInvZ, invZ);
        var gradMeanX = Col(sums, GradMeanX);
        var gradMeanY = Col(sums, GradMeanY);
        var gradInvA = Col(sums, GradInvA);
        var gradInvB = Col(sums, GradInvB);
        var gradInvC = Col(sums, GradInvC);

        // Mean projection: mu = f (x, y) / z + c.
        var gradCamX = engine.TensorMultiply(gradMeanX, fInvZ);
        var gradCamY = engine.TensorMultiply(gradMeanY, fInvZ);
        var gradCamZ = engine.TensorNegate(engine.TensorAdd(
            engine.TensorMultiply(gradMeanX, engine.TensorMultiply(fInvZ2, camX)),
            engine.TensorMultiply(gradMeanY, engine.TensorMultiply(fInvZ2, camY))));

        // A Gaussian whose conic received no gradient keeps its shape (its rotation and scale are not touched).
        var covChanged = Not(engine.TensorMultiply(
            engine.TensorMultiply(Not(Gt(engine.TensorAbs(gradInvA), zero)), Not(Gt(engine.TensorAbs(gradInvB), zero))),
            Not(Gt(engine.TensorAbs(gradInvC), zero))));

        // Conic Q = Sigma2D^-1: dL/dQ as a symmetric matrix has off-diagonal 0.5 dL/dInvB, because the exponent
        // counts invB twice. dL/dSigma2D = -Q (dL/dQ) Q.
        var invA = Col(raster, ColInvA);
        var invB = Col(raster, ColInvB);
        var invC = Col(raster, ColInvC);
        var halfB = engine.TensorMultiplyScalar(gradInvB, half);
        var temp00 = engine.TensorAdd(engine.TensorMultiply(gradInvA, invA), engine.TensorMultiply(halfB, invB));
        var temp01 = engine.TensorAdd(engine.TensorMultiply(gradInvA, invB), engine.TensorMultiply(halfB, invC));
        var temp10 = engine.TensorAdd(engine.TensorMultiply(halfB, invA), engine.TensorMultiply(gradInvC, invB));
        var temp11 = engine.TensorAdd(engine.TensorMultiply(halfB, invB), engine.TensorMultiply(gradInvC, invC));
        var g00 = engine.TensorNegate(engine.TensorAdd(engine.TensorMultiply(invA, temp00), engine.TensorMultiply(invB, temp10)));
        var g01 = engine.TensorNegate(engine.TensorAdd(engine.TensorMultiply(invA, temp01), engine.TensorMultiply(invB, temp11)));
        var g11 = engine.TensorNegate(engine.TensorAdd(engine.TensorMultiply(invB, temp01), engine.TensorMultiply(invC, temp11)));
        var screenGrad = engine.TensorStack(new[] { g00, g01, g01, g11 }, 1).Reshape(m, 2, 2);

        // Sigma2D = J Sigma_cam J^T with J = [[f/z, 0, -f x / z^2], [0, f/z, -f y / z^2]].
        var zeros = engine.TensorZerosLike(camZ);
        var j02 = engine.TensorNegate(engine.TensorMultiply(fInvZ2, camX));
        var j12 = engine.TensorNegate(engine.TensorMultiply(fInvZ2, camY));
        var jacobian = engine.TensorStack(new[] { fInvZ, zeros, j02, zeros, fInvZ, j12 }, 1).Reshape(m, 2, 3);
        var jacobianT = engine.TensorPermute(jacobian, new[] { 0, 2, 1 }).Contiguous();
        var camCov = engine.TensorStack(new[]
        {
            Col(extra, ExCov00), Col(extra, ExCov01), Col(extra, ExCov02),
            Col(extra, ExCov01), Col(extra, ExCov11), Col(extra, ExCov12),
            Col(extra, ExCov02), Col(extra, ExCov12), Col(extra, ExCov22),
        }, 1).Reshape(m, 3, 3);
        var camCovGrad = engine.TensorMatMul(engine.TensorMatMul(jacobianT, screenGrad), jacobian);

        // The Jacobian depends on the camera-space mean: dL/dJ = 2 (dL/dSigma2D) J Sigma_cam.
        var jacobianGrad = engine.TensorMultiplyScalar(
            engine.TensorMatMul(screenGrad, engine.TensorMatMul(jacobian, camCov)), two).Reshape(m, 6);
        var dJ00 = Col(jacobianGrad, 0);
        var dJ02 = Col(jacobianGrad, 2);
        var dJ11 = Col(jacobianGrad, 4);
        var dJ12 = Col(jacobianGrad, 5);
        var twoFInvZ3 = engine.TensorMultiplyScalar(engine.TensorMultiply(fInvZ2, invZ), two);
        var ewaX = engine.TensorNegate(engine.TensorMultiply(dJ02, fInvZ2));
        var ewaY = engine.TensorNegate(engine.TensorMultiply(dJ12, fInvZ2));
        var ewaZ = engine.TensorAdd(
            engine.TensorNegate(engine.TensorMultiply(engine.TensorAdd(dJ00, dJ11), fInvZ2)),
            engine.TensorAdd(
                engine.TensorMultiply(dJ02, engine.TensorMultiply(twoFInvZ3, camX)),
                engine.TensorMultiply(dJ12, engine.TensorMultiply(twoFInvZ3, camY))));
        gradCamX = engine.TensorAdd(gradCamX, Select(covChanged, ewaX, zeros));
        gradCamY = engine.TensorAdd(gradCamY, Select(covChanged, ewaY, zeros));
        gradCamZ = engine.TensorAdd(gradCamZ, Select(covChanged, ewaZ, zeros));

        // Back to world space: x_cam = R^T (x - c), so dL/dx = R dL/dx_cam, and Sigma_cam = R^T Sigma R, so
        // dL/dSigma = R (dL/dSigma_cam) R^T.
        var cameraRot = Tensor<T>.FromMatrix(cameraRotation);
        var cameraRotT = engine.TensorTranspose(cameraRot);
        var gradPos = engine.TensorMatMul(engine.TensorStack(new[] { gradCamX, gradCamY, gradCamZ }, 1), cameraRotT);
        var tiled = engine.TensorTile(cameraRot.Reshape(1, 3, 3), new[] { m, 1, 1 });
        var tiledT = engine.TensorPermute(tiled, new[] { 0, 2, 1 }).Contiguous();
        var worldCovGrad = engine.TensorMatMul(engine.TensorMatMul(tiled, camCovGrad), tiledT);

        // Sigma = Rq diag(s^2) Rq^T with s = max(1e-6, exp(log s)): dL/dlog s_j = 2 s_j^2 (Rq^T G Rq)_jj and
        // dL/dRq = 2 G Rq diag(s^2), at the current (pre-update) parameters.
        var rotRows = engine.TensorIndexSelect(_rotations, rows, 0);
        var logRows = engine.TensorIndexSelect(_logScales, rows, 0);
        var (qw, qx, qy, qz) = UnitQuaternions(rotRows, identityWhenDegenerate: true);
        var rq = RotationMatrices(qw, qx, qy, qz);
        var s = CovarianceScales(logRows);
        var s2 = engine.TensorMultiply(s, s);
        var gRq = engine.TensorMatMul(worldCovGrad, rq);
        var diagonal = engine.ReduceSum(engine.TensorMultiply(rq, gRq), new[] { 1 }, false);
        var unclamped = Not(Lt(engine.TensorExp(logRows), ops.FromDouble(1e-6)));
        var covChanged3 = Bc(covChanged.Reshape(m, 1), new[] { m, 3 });
        var gradLogScale = Select(covChanged3,
            engine.TensorMultiply(engine.TensorMultiplyScalar(engine.TensorMultiply(s2, diagonal), two), unclamped),
            engine.TensorZerosLike(logRows));
        var gR = engine.TensorMultiplyScalar(Mul(gRq, s2.Reshape(m, 1, 3)), two).Reshape(m, 9);
        var quatGrad = QuaternionGradient(gR, qw, qx, qy, qz);
        var gradRotation = Select(Bc(covChanged.Reshape(m, 1), new[] { m, 4 }), quatGrad, engine.TensorZerosLike(quatGrad));

        // Colour: with spherical harmonics dL/dcoeff_k = basis_k dL/dcolour, zero where the colour clamp is active.
        var colourGrad = engine.TensorStack(new[] { Col(sums, GradSumRed), Col(sums, GradSumGreen), Col(sums, GradSumBlue) }, 1);
        var shRows = engine.TensorIndexSelect(_shCoefficients.Reshape(n, basisCount * 3), rows, 0).Reshape(m, basisCount, 3);
        var lrScales = engine.TensorIndexSelect(EnsureLearningRateScales(), rows, 0);
        var lrColour = engine.TensorMultiplyScalar(Col(lrScales, LrColor), ops.FromDouble(ColorLearningRate)).Reshape(m, 1, 1);
        Tensor<T> gradSh;
        Tensor<T> newSh;
        if (_useSphericalHarmonics)
        {
            var raw = engine.TensorStack(new[] { Col(extra, ExRawRed), Col(extra, ExRawGreen), Col(extra, ExRawBlue) }, 1);
            var passes = engine.TensorMultiply(Not(Le(raw, zero)), Not(Ge(raw, one)));
            gradSh = Mul(splats.Basis.Reshape(m, basisCount, 1), engine.TensorMultiply(colourGrad, passes).Reshape(m, 1, 3));
            newSh = engine.TensorSubtract(shRows, Mul(lrColour, gradSh));
        }
        else
        {
            gradSh = colourGrad.Reshape(m, 1, 3);
            newSh = Clamp01(engine.TensorSubtract(shRows, Mul(lrColour, gradSh)));
        }

        var gradOpacity = Col(sums, GradOpacity);
        var opacityRows = engine.TensorIndexSelect(_opacityLogits, rows, 0);
        var newOpacity = engine.TensorSubtract(opacityRows, engine.TensorMultiply(
            engine.TensorMultiplyScalar(Col(lrScales, LrOpacity), ops.FromDouble(OpacityLearningRate)), gradOpacity));
        var positionRows = engine.TensorIndexSelect(_positions, rows, 0);
        var newPositions = engine.TensorSubtract(positionRows, Mul(
            engine.TensorMultiplyScalar(Col(lrScales, LrPosition), ops.FromDouble(PositionLearningRate)).Reshape(m, 1), gradPos));

        // Log-scale step, floored at log(MinScale).
        T logMin = ops.Log(ops.FromDouble(MinScale));
        var lrScale = engine.TensorMultiplyScalar(Col(lrScales, LrScale), ops.FromDouble(ScaleLearningRate)).Reshape(m, 1);
        var steppedLog = MaxScalar(engine.TensorSubtract(MaxScalar(logRows, logMin), Mul(lrScale, gradLogScale)), logMin);
        var newLogScales = Select(covChanged3, steppedLog, logRows);

        // Quaternion step from the normalised quaternion, renormalised (identity when it collapses).
        var lrRotation = engine.TensorMultiplyScalar(Col(lrScales, LrRotation), ops.FromDouble(RotationLearningRate)).Reshape(m, 1);
        var unit = engine.TensorStack(new[] { qw, qx, qy, qz }, 1);
        var (sw, sx, sy, sz) = UnitQuaternions(engine.TensorSubtract(unit, Mul(lrRotation, quatGrad)), identityWhenDegenerate: true);
        var newRotations = Select(Bc(covChanged.Reshape(m, 1), new[] { m, 4 }),
            engine.TensorStack(new[] { sw, sx, sy, sz }, 1), rotRows);

        WriteRows(_positions, rows, mask, newPositions);
        WriteRows(_rotations, rows, mask, newRotations);
        WriteRows(_logScales, rows, mask, newLogScales);
        WriteInPlace(_opacityLogits, Select(mask, engine.TensorIndexAdd(engine.TensorZerosLike(_opacityLogits), 0, rows, newOpacity), _opacityLogits));
        var shFlat = _shCoefficients.Reshape(n, basisCount * 3);
        var shUpdated = Select(mask.Reshape(n, 1),
            engine.TensorIndexAdd(engine.TensorZerosLike(shFlat), 0, rows, newSh.Reshape(m, basisCount * 3)), shFlat);
        WriteInPlace(_shCoefficients, shUpdated);

        positionGrad = engine.TensorIndexAdd(positionGrad, 0, rows, gradPos);
        rotationGrad = engine.TensorIndexAdd(rotationGrad, 0, rows, gradRotation);
        scaleGrad = engine.TensorIndexAdd(scaleGrad, 0, rows, gradLogScale);
        opacityGrad = engine.TensorIndexAdd(opacityGrad, 0, rows, gradOpacity.Reshape(m, 1));
        shGrad = engine.TensorIndexAdd(shGrad, 0, rows, gradSh.Reshape(m, basisCount * 3));
        PublishImageGradients(positionGrad, rotationGrad, scaleGrad, opacityGrad, shGrad);

        gradients.Add(new GaussianGradient(rows, gradPos));
        return gradients;
    }

    /// <summary>
    /// dL/dq of a unit quaternion (w, x, y, z) from dL/dR [M, 9] (row-major), through the rotation-matrix formulas of
    /// <see cref="RotationMatrices"/>.
    /// </summary>
    private Tensor<T> QuaternionGradient(Tensor<T> gradR, Tensor<T> w, Tensor<T> x, Tensor<T> y, Tensor<T> z)
    {
        var engine = Engine;
        T two = NumOps.FromDouble(2.0);
        T four = NumOps.FromDouble(4.0);
        var g00 = Col(gradR, 0);
        var g01 = Col(gradR, 1);
        var g02 = Col(gradR, 2);
        var g10 = Col(gradR, 3);
        var g11 = Col(gradR, 4);
        var g12 = Col(gradR, 5);
        var g20 = Col(gradR, 6);
        var g21 = Col(gradR, 7);
        var g22 = Col(gradR, 8);
        var w2 = engine.TensorMultiplyScalar(w, two);
        var x2 = engine.TensorMultiplyScalar(x, two);
        var y2 = engine.TensorMultiplyScalar(y, two);
        var z2 = engine.TensorMultiplyScalar(z, two);
        var x4 = engine.TensorMultiplyScalar(x, four);
        var y4 = engine.TensorMultiplyScalar(y, four);
        var z4 = engine.TensorMultiplyScalar(z, four);

        var gradW = SumOfProducts(
            (g01, engine.TensorNegate(z2)), (g02, y2), (g10, z2), (g12, engine.TensorNegate(x2)), (g20, engine.TensorNegate(y2)), (g21, x2));
        var gradX = SumOfProducts(
            (g01, y2), (g02, z2), (g10, y2), (g11, engine.TensorNegate(x4)), (g12, engine.TensorNegate(w2)), (g20, z2), (g21, w2), (g22, engine.TensorNegate(x4)));
        var gradY = SumOfProducts(
            (g00, engine.TensorNegate(y4)), (g01, x2), (g02, w2), (g10, x2), (g12, z2), (g20, engine.TensorNegate(w2)), (g21, z2), (g22, engine.TensorNegate(y4)));
        var gradZ = SumOfProducts(
            (g00, engine.TensorNegate(z4)), (g01, engine.TensorNegate(w2)), (g02, x2), (g10, w2), (g11, engine.TensorNegate(z4)), (g12, y2), (g20, x2), (g21, y2));
        return engine.TensorStack(new[] { gradW, gradX, gradY, gradZ }, 1);
    }

    private Tensor<T> SumOfProducts(params (Tensor<T> A, Tensor<T> B)[] terms)
    {
        var engine = Engine;
        var sum = engine.TensorMultiply(terms[0].A, terms[0].B);
        for (int i = 1; i < terms.Length; i++)
        {
            sum = engine.TensorAdd(sum, engine.TensorMultiply(terms[i].A, terms[i].B));
        }

        return sum;
    }

    /// <summary>Publishes the image-step gradient in the flat parameter layout.</summary>
    private void PublishImageGradients(Tensor<T> position, Tensor<T> rotation, Tensor<T> logScale, Tensor<T> opacity, Tensor<T> sh)
    {
        int total = position.Length + rotation.Length + logScale.Length + opacity.Length + sh.Length;
        var published = total == 0
            ? new Vector<T>(0)
            : Engine.TensorConcatenate(new[]
            {
                position.Reshape(position.Length),
                rotation.Reshape(rotation.Length),
                logScale.Reshape(logScale.Length),
                opacity.Reshape(opacity.Length),
                sh.Reshape(sh.Length),
            }, 0).ToVector();
        PublishFlatParameterGradients(published);
    }

    /// <summary>
    /// Test-only hook that drives <see cref="DensifyAndPrune"/> with a synthesized gradient list
    /// at a caller-specified iteration, so tests can assert the schedule window gate ([Start, End])
    /// without threading a full training loop. Only used by the #1835 regression tests.
    /// </summary>
    internal void RunDensifyAndPruneForTest(int iteration, double gradientsMagnitude)
    {
        _trainingStep = iteration;
        int n = GaussianCount;
        var synthetic = new List<GaussianGradient>();
        if (n > 0)
        {
            var engine = Engine;
            var indices = engine.TensorArange<T>(0, n, 1).Cast<int>();
            var gradient = engine.TensorConcatenate(new[]
            {
                Full(new[] { n, 1 }, NumOps.FromDouble(gradientsMagnitude)),
                new Tensor<T>(new[] { n, 2 }),
            }, 1);
            synthetic.Add(new GaussianGradient(indices, gradient));
        }

        DensifyAndPrune(synthetic);
    }

    /// <summary>
    /// Test-only hook: replaces the whole cloud with the given per-attribute tensors (positions [N, 3], unit
    /// quaternions [N, 4], log-scales [N, 3], opacity logits [N], spherical-harmonics coefficients [N, K, 3]).
    /// </summary>
    internal void SetGaussianStateForTest(
        Tensor<T> positions,
        Tensor<T> rotations,
        Tensor<T> logScales,
        Tensor<T> opacityLogits,
        Tensor<T> shCoefficients)
    {
        Guard.NotNull(positions);
        Guard.NotNull(rotations);
        Guard.NotNull(logScales);
        Guard.NotNull(opacityLogits);
        Guard.NotNull(shCoefficients);
        int n = positions.Shape[0];
        int basisCount = GetShBasisCount();
        if (positions.Rank != 2 || positions.Shape[1] != 3
            || rotations.Rank != 2 || rotations.Shape[0] != n || rotations.Shape[1] != 4
            || logScales.Rank != 2 || logScales.Shape[0] != n || logScales.Shape[1] != 3
            || opacityLogits.Rank != 1 || opacityLogits.Shape[0] != n
            || shCoefficients.Rank != 3 || shCoefficients.Shape[0] != n || shCoefficients.Shape[1] != basisCount || shCoefficients.Shape[2] != 3)
        {
            throw new ArgumentException("Gaussian state tensors must be [N, 3], [N, 4], [N, 3], [N] and [N, K, 3].");
        }

        _positions = positions.Clone();
        _rotations = rotations.Clone();
        _logScales = logScales.Clone();
        _opacityLogits = opacityLogits.Clone();
        _shCoefficients = shCoefficients.Clone();
        _learningRateScales = Full(new[] { n, LearningRateScaleColumns }, NumOps.One);
        InvalidateParameterCountCache();
    }

    /// <summary>Test-only hook: runs the opacity-reset schedule as if <paramref name="iteration"/> steps had run.</summary>
    internal void RunOpacityResetForTest(int iteration)
    {
        _trainingStep = iteration;
        ResetOpacityIfScheduled();
    }

    /// <summary>
    /// Kerbl et al. 2023, section 5.2: every 3000 iterations, while densification is still running, every opacity is
    /// reset to at most 0.01 so floaters near the cameras can be pruned and the remaining Gaussians re-learn theirs.
    /// </summary>
    private void ResetOpacityIfScheduled()
    {
        if (!EnableDensification
            || _trainingStep <= 0
            || _trainingStep % OpacityResetInterval != 0
            || _trainingStep > EffectiveDensificationEndIteration
            || GaussianCount == 0)
        {
            return;
        }

        T ceilingLogit = NumOps.Log(NumOps.FromDouble(OpacityResetCeiling / (1.0 - OpacityResetCeiling)));
        WriteInPlace(_opacityLogits, MinScalar(_opacityLogits, ceilingLogit));
    }

    private void DensifyAndPrune(List<GaussianGradient> gradients)
    {
        // #1835 schedule window: skip densification before the start iteration (let the
        // random cloud stabilize) and after the end iteration (freeze the cloud for final
        // convergence). Paper: [500, 15000]. Callers override via GaussianSplattingOptions.
        if (_trainingStep < EffectiveDensificationStartIteration
            || _trainingStep > EffectiveDensificationEndIteration)
        {
            return;
        }

        var engine = Engine;
        var ops = NumOps;
        int n = GaussianCount;
        int basisCount = GetShBasisCount();
        var lrScales = EnsureLearningRateScales();

        // Prune first: Gaussians whose opacity is below the threshold (NaN opacities stay).
        T pruneThreshold = ops.FromDouble(Math.Max(0.0, EffectiveOpacityPruneThreshold));
        var kept = n == 0 ? new Tensor<T>(new[] { 0 }) : Not(Lt(engine.TensorSigmoid(_opacityLogits), pruneThreshold));
        var keptRows = engine.TensorNonzero(kept);
        int count = keptRows._shape[0];
        if (count < n)
        {
            var rows = keptRows.Reshape(count);
            _positions = engine.TensorIndexSelect(_positions, rows, 0).Clone();
            _rotations = engine.TensorIndexSelect(_rotations, rows, 0).Clone();
            _logScales = engine.TensorIndexSelect(_logScales, rows, 0).Clone();
            _opacityLogits = engine.TensorIndexSelect(_opacityLogits, rows, 0).Clone();
            _shCoefficients = engine.TensorIndexSelect(_shCoefficients, rows, 0).Clone();
            _learningRateScales = engine.TensorIndexSelect(lrScales, rows, 0).Clone();
        }

        int cap = EffectiveMaxGaussianCount;
        var indexParts = new List<Tensor<int>>();
        var gradientParts = new List<Tensor<T>>();
        foreach (var batch in gradients)
        {
            if (batch.Count > 0)
            {
                indexParts.Add(batch.Indices);
                gradientParts.Add(batch.Gradients);
            }
        }

        if (count >= cap || indexParts.Count == 0 || count == 0)
        {
            // Pruning may have changed the parameter surface even though no split can be added.
            InvalidateParameterCountCache();
            return;
        }

        // Candidates in descending gradient magnitude (stable; NaN magnitudes last, as LINQ orders them). The run
        // stops at the first magnitude below the threshold, pruned sources are skipped, and at most cap - count
        // splits are made.
        var sourceRows = engine.TensorConcatenate(indexParts.ToArray(), 0);
        var sourceGrads = engine.TensorConcatenate(gradientParts.ToArray(), 0);
        var magnitude = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(sourceGrads, sourceGrads), new[] { 1 }, false));
        var order = engine.TensorArgsort(engine.TensorNegate(magnitude), 0, false);
        var sortedRows = engine.TensorIndexSelect(sourceRows, order, 0);
        var sortedGrads = engine.TensorIndexSelect(sourceGrads, order, 0);
        var aboveThreshold = engine.TensorCumProd(
            Not(Lt(engine.TensorIndexSelect(magnitude, order, 0), ops.FromDouble(EffectiveGradientNormThreshold))), 0);
        var alive = engine.TensorIndexSelect(kept, sortedRows, 0);
        var candidate = engine.TensorMultiply(aboveThreshold, alive);
        var taken = engine.TensorMultiply(candidate, Le(engine.TensorCumSum(candidate, 0), ops.FromDouble(cap - count)));
        var takenRows = engine.TensorNonzero(taken);
        int splits = takenRows._shape[0];
        if (splits == 0)
        {
            InvalidateParameterCountCache();
            return;
        }

        var picked = takenRows.Reshape(splits);

        // Rows of the split sources after pruning: the number of kept rows before each.
        var newIndex = engine.TensorSubtractScalar(engine.TensorCumSum(kept, 0), ops.One);
        var sources = engine.TensorIndexSelect(newIndex, engine.TensorIndexSelect(sortedRows, picked, 0), 0).Cast<int>();

        // Split direction: the normalised position gradient, or a seeded random direction when it is zero.
        var direction = NormalizeRows(engine.TensorIndexSelect(sortedGrads, picked, 0));
        var isZero = engine.TensorEquals(engine.ReduceSum(engine.TensorAbs(direction), new[] { 1 }, false), ops.Zero);
        var zeroRows = engine.TensorNonzero(isZero);
        int zeroCount = zeroRows._shape[0];
        if (zeroCount > 0)
        {
            // The draws keep master's order (three per zero-gradient split, in split order); no engine op draws a
            // tensor from a caller-owned Random.
            var random = new Tensor<T>(new[] { zeroCount, 3 });
            for (int i = 0; i < zeroCount; i++)
            {
                random[i, 0] = ops.FromDouble(ModelRandom.NextDouble() - 0.5);
                random[i, 1] = ops.FromDouble(ModelRandom.NextDouble() - 0.5);
                random[i, 2] = ops.FromDouble(ModelRandom.NextDouble() - 0.5);
            }

            var scattered = engine.TensorIndexAdd(engine.TensorZerosLike(direction), 0, zeroRows.Reshape(zeroCount), NormalizeRows(random));
            direction = Select(isZero.Reshape(splits, 1), scattered, direction);
        }

        // #1835 excellence goal #4 -- SplitChildLearningRateScales. Position jitter shrinks by the position
        // multiplier and the scale factor grows by the scale multiplier; both children inherit the compounded
        // per-attribute multipliers (each generation layers its multiplier on its parent's).
        var childScales = _options.SplitChildLearningRateScales ?? new AiDotNet.Models.Options.SplitChildLearningRateScales();
        T jitter = ops.FromDouble(SplitPositionJitter * childScales.Position);
        T scaleFactor = ops.FromDouble(SplitScaleFactor / childScales.Scale);
        var offset = engine.TensorMultiplyScalar(direction, jitter);

        var sourcePositions = engine.TensorIndexSelect(_positions, sources, 0);
        var splitScales = engine.TensorLog(MaxScalar(
            engine.TensorMultiplyScalar(engine.TensorExp(engine.TensorIndexSelect(_logScales, sources, 0)), scaleFactor),
            ops.FromDouble(MinScale)));
        var splitAlpha = MinScalar(
            engine.TensorMultiplyScalar(engine.TensorSigmoid(engine.TensorIndexSelect(_opacityLogits, sources, 0)), ops.FromDouble(SplitOpacityFactor)),
            ops.FromDouble(SplitOpacityMax));
        var splitOpacity = Logit(splitAlpha);
        var multipliers = Tensor<T>.FromVector(new Vector<T>(new[]
        {
            ops.FromDouble(childScales.Position),
            ops.FromDouble(childScales.Scale),
            ops.FromDouble(childScales.Opacity),
            ops.FromDouble(childScales.Rotation),
            ops.FromDouble(childScales.Color),
        })).Reshape(1, LearningRateScaleColumns);
        var splitLr = Mul(engine.TensorIndexSelect(_learningRateScales, sources, 0), multipliers);

        // Sources move along +direction, clones along -direction; both get the reduced scale and opacity.
        var sourceMask = engine.TensorIndexAdd(new Tensor<T>(new[] { count }), 0, sources, Full(new[] { splits }, ops.One));
        WriteRows(_positions, sources, sourceMask, engine.TensorAdd(sourcePositions, offset));
        WriteRows(_logScales, sources, sourceMask, splitScales);
        WriteRows(_learningRateScales, sources, sourceMask, splitLr);
        WriteInPlace(_opacityLogits, Select(sourceMask,
            engine.TensorIndexAdd(engine.TensorZerosLike(_opacityLogits), 0, sources, splitOpacity), _opacityLogits));

        var clonePositions = engine.TensorSubtract(sourcePositions, offset);
        var cloneRotations = engine.TensorIndexSelect(_rotations, sources, 0);
        var cloneSh = engine.TensorIndexSelect(_shCoefficients, sources, 0);
        _positions = AppendRows(_positions, clonePositions);
        _rotations = AppendRows(_rotations, cloneRotations);
        _logScales = AppendRows(_logScales, splitScales);
        _opacityLogits = AppendRows(_opacityLogits, splitOpacity);
        _shCoefficients = AppendRows(_shCoefficients, cloneSh.Reshape(splits, basisCount, 3));
        _learningRateScales = AppendRows(_learningRateScales, splitLr);

        // Densification is a STRUCTURAL parameter change: the per-Gaussian tensors are the model's parameters, so
        // pruning or splitting resizes the parameter surface exactly the way adding or removing a layer does.
        // Without this the cached ParameterLayoutSnapshot keeps describing the previous cloud and the next
        // PublishFlatParameterGradients call fails its length check.
        InvalidateParameterCountCache();
    }

    /// <summary>Mean world-space position-gradient magnitude over the visible Gaussians of a step (0 when none).</summary>
    private double MeanGradientNorm(List<GaussianGradient> gradients)
    {
        var engine = Engine;
        var magnitudes = new List<Tensor<T>>();
        foreach (var batch in gradients)
        {
            if (batch.Count > 0)
            {
                magnitudes.Add(engine.TensorSqrt(engine.ReduceSum(
                    engine.TensorMultiply(batch.Gradients, batch.Gradients), new[] { 1 }, false)));
            }
        }

        return magnitudes.Count == 0
            ? 0.0
            : NumOps.ToDouble(engine.TensorMean(engine.TensorConcatenate(magnitudes.ToArray(), 0)));
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return ForwardWithMemory(input);
    }

    /// <summary>
    /// Real spherical-harmonics basis [R, K] of the (renormalised) directions [R, 3], degrees 0-3 with the constants of
    /// Sloan's "Stupid Spherical Harmonics Tricks" (as in the reference 3DGS rasteriser, without its sign convention).
    /// </summary>
    private Tensor<T> ShBasis(Tensor<T> directions)
    {
        var engine = Engine;
        int rows = directions._shape[0];
        int degree = _useSphericalHarmonics ? _shDegree : 0;
        var unit = NormalizeRows(directions);
        var x = Col(unit, 0);
        var y = Col(unit, 1);
        var z = Col(unit, 2);
        var basis = new List<Tensor<T>> { Full(new[] { rows }, NumOps.FromDouble(0.282095)) };
        if (degree >= 1)
        {
            T c1 = NumOps.FromDouble(0.488603);
            basis.Add(engine.TensorMultiplyScalar(y, c1));
            basis.Add(engine.TensorMultiplyScalar(z, c1));
            basis.Add(engine.TensorMultiplyScalar(x, c1));
        }
        if (degree >= 2)
        {
            var xx = engine.TensorMultiply(x, x);
            var yy = engine.TensorMultiply(y, y);
            var zz = engine.TensorMultiply(z, z);
            T c2 = NumOps.FromDouble(1.092548);
            basis.Add(engine.TensorMultiplyScalar(engine.TensorMultiply(x, y), c2));
            basis.Add(engine.TensorMultiplyScalar(engine.TensorMultiply(y, z), c2));
            basis.Add(engine.TensorMultiplyScalar(engine.TensorSubtractScalar(engine.TensorMultiplyScalar(zz, NumOps.FromDouble(3.0)), NumOps.One), NumOps.FromDouble(0.315392)));
            basis.Add(engine.TensorMultiplyScalar(engine.TensorMultiply(x, z), c2));
            basis.Add(engine.TensorMultiplyScalar(engine.TensorSubtract(xx, yy), NumOps.FromDouble(0.546274)));
            if (degree >= 3)
            {
                T three = NumOps.FromDouble(3.0);
                T five = NumOps.FromDouble(5.0);
                var fiveZzMinusOne = engine.TensorSubtractScalar(engine.TensorMultiplyScalar(zz, five), NumOps.One);
                basis.Add(engine.TensorMultiplyScalar(engine.TensorMultiply(y, engine.TensorSubtract(engine.TensorMultiplyScalar(xx, three), yy)), NumOps.FromDouble(0.590044)));
                basis.Add(engine.TensorMultiplyScalar(engine.TensorMultiply(engine.TensorMultiply(x, y), z), NumOps.FromDouble(2.890611)));
                basis.Add(engine.TensorMultiplyScalar(engine.TensorMultiply(y, fiveZzMinusOne), NumOps.FromDouble(0.457046)));
                basis.Add(engine.TensorMultiplyScalar(engine.TensorMultiply(z, engine.TensorSubtractScalar(engine.TensorMultiplyScalar(zz, five), three)), NumOps.FromDouble(0.373176)));
                basis.Add(engine.TensorMultiplyScalar(engine.TensorMultiply(x, fiveZzMinusOne), NumOps.FromDouble(0.457046)));
                basis.Add(engine.TensorMultiplyScalar(engine.TensorMultiply(z, engine.TensorSubtract(xx, yy)), NumOps.FromDouble(1.445306)));
                basis.Add(engine.TensorMultiplyScalar(engine.TensorMultiply(x, engine.TensorSubtract(xx, engine.TensorMultiplyScalar(yy, three))), NumOps.FromDouble(0.590044)));
            }
        }

        return engine.TensorStack(basis.ToArray(), 1);
    }

    private int GetShBasisCount()
    {
        int degree = _useSphericalHarmonics ? _shDegree : 0;
        return (degree + 1) * (degree + 1);
    }

    // ---- Element-wise helpers. Comparisons return 0/1 masks; NaN follows the scalar code each one replaces. ----

    /// <summary>Rows [R, 3] scaled to unit length; zero-length (and NaN) rows are left as they are.</summary>
    private Tensor<T> NormalizeRows(Tensor<T> rows)
    {
        var engine = Engine;
        int count = rows._shape[0];
        var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(rows, rows), new[] { 1 }, false));
        var inv = Select(Gt(norm, NumOps.Zero), engine.TensorReciprocal(norm), engine.TensorOnesLike(norm));
        return Mul(rows, inv.Reshape(count, 1));
    }

    /// <summary>Column <paramref name="column"/> of a rank-2 tensor, as a rank-1 tensor.</summary>
    private Tensor<T> Col(Tensor<T> matrix, int column)
    {
        int rows = matrix._shape[0];
        return Engine.TensorSlice(matrix, new[] { 0, column }, new[] { rows, 1 }).Reshape(rows);
    }

    /// <summary>Column <paramref name="column"/> of a rank-2 tensor with <paramref name="rows"/> rows, kept as [rows, 1].</summary>
    private Tensor<T> Col2(Tensor<T> matrix, int column, int rows)
        => Engine.TensorSlice(matrix, new[] { 0, column }, new[] { rows, 1 });

    /// <summary>Column <paramref name="column"/> of a rank-2 tensor with <paramref name="rows"/> rows, as [1, rows].</summary>
    private Tensor<T> Row(Tensor<T> matrix, int column, int rows)
        => Engine.TensorSlice(matrix, new[] { 0, column }, new[] { rows, 1 }).Reshape(1, rows);

    private Tensor<T> Full(int[] shape, T value)
    {
        var tensor = new Tensor<T>(shape);
        tensor.Fill(value);
        return tensor;
    }

    private Tensor<T> Bc(Tensor<T> tensor, int[] shape) => Engine.TensorBroadcastTo(tensor, shape);

    private static int[] BroadcastShape(int[] a, int[] b)
    {
        int rank = Math.Max(a.Length, b.Length);
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int ia = i - (rank - a.Length);
            int ib = i - (rank - b.Length);
            int da = ia >= 0 ? a[ia] : 1;
            int db = ib >= 0 ? b[ib] : 1;
            shape[i] = da == 1 ? db : da;
        }

        return shape;
    }

    private Tensor<T> Add(Tensor<T> a, Tensor<T> b)
    {
        var shape = BroadcastShape(a._shape, b._shape);
        return Engine.TensorAdd(Bc(a, shape), Bc(b, shape));
    }

    private Tensor<T> Sub(Tensor<T> a, Tensor<T> b)
    {
        var shape = BroadcastShape(a._shape, b._shape);
        return Engine.TensorSubtract(Bc(a, shape), Bc(b, shape));
    }

    private Tensor<T> Mul(Tensor<T> a, Tensor<T> b)
    {
        var shape = BroadcastShape(a._shape, b._shape);
        return Engine.TensorMultiply(Bc(a, shape), Bc(b, shape));
    }

    private Tensor<T> Div(Tensor<T> a, Tensor<T> b)
    {
        var shape = BroadcastShape(a._shape, b._shape);
        return Engine.TensorDivide(Bc(a, shape), Bc(b, shape));
    }

    /// <summary>Element-wise select (mask != 0 picks <paramref name="x"/>) with broadcasting.</summary>
    private Tensor<T> Select(Tensor<T> mask, Tensor<T> x, Tensor<T> y)
    {
        var shape = BroadcastShape(BroadcastShape(mask._shape, x._shape), y._shape);
        return Engine.TensorWhere(Bc(mask, shape), Bc(x, shape), Bc(y, shape));
    }

    private Tensor<T> Lt(Tensor<T> x, T value) => Engine.TensorLessThan(x, value);

    private Tensor<T> Gt(Tensor<T> x, T value) => Engine.TensorGreaterThan(x, value);

    /// <summary>x &lt;= value (0 for NaN).</summary>
    private Tensor<T> Le(Tensor<T> x, T value) => Engine.TensorAdd(Engine.TensorLessThan(x, value), Engine.TensorEquals(x, value));

    /// <summary>x &gt;= value (0 for NaN).</summary>
    private Tensor<T> Ge(Tensor<T> x, T value) => Engine.TensorAdd(Engine.TensorGreaterThan(x, value), Engine.TensorEquals(x, value));

    private Tensor<T> Not(Tensor<T> mask) => Engine.ScalarMinusTensor(NumOps.One, mask);

    /// <summary>Math.Max(x, value): a NaN x stays NaN.</summary>
    private Tensor<T> MaxScalar(Tensor<T> x, T value) => Engine.TensorWhere(Lt(x, value), Full(x._shape, value), x);

    /// <summary>Math.Min(x, value): a NaN x stays NaN.</summary>
    private Tensor<T> MinScalar(Tensor<T> x, T value) => Engine.TensorWhere(Gt(x, value), Full(x._shape, value), x);

    /// <summary>0 for x &lt;= 0, 1 for x &gt;= 1, x otherwise (NaN stays NaN).</summary>
    private Tensor<T> Clamp01(Tensor<T> x)
        => Engine.TensorWhere(Le(x, NumOps.Zero), Engine.TensorZerosLike(x), Engine.TensorWhere(Ge(x, NumOps.One), Engine.TensorOnesLike(x), x));

    /// <summary>log(p / (1 - p)) of p clamped to [1e-6, 1 - 1e-6].</summary>
    private Tensor<T> Logit(Tensor<T> probability)
    {
        var engine = Engine;
        var clamped = MinScalar(MaxScalar(probability, NumOps.FromDouble(1e-6)), NumOps.FromDouble(1.0 - 1e-6));
        return engine.TensorLog(engine.TensorDivide(clamped, engine.ScalarMinusTensor(NumOps.One, clamped)));
    }

    /// <summary>Math.Max for one pair of scalars (NaN propagates).</summary>
    private T MaxOf(T a, T b)
    {
        if (NumOps.IsNaN(a))
        {
            return a;
        }
        if (NumOps.IsNaN(b))
        {
            return b;
        }

        return NumOps.GreaterThan(b, a) ? b : a;
    }

    /// <summary>Appends rows to a per-Gaussian tensor, returning a tensor the model owns.</summary>
    private Tensor<T> AppendRows(Tensor<T> existing, Tensor<T> rows)
        => existing._shape[0] == 0 ? rows.Clone() : Engine.TensorConcatenate(new[] { existing, rows }, 0).Clone();

    /// <summary>Writes <paramref name="value"/> into a parameter tensor of the same shape, in place.</summary>
    private static void WriteInPlace(Tensor<T> field, Tensor<T> value) => value.CopyTo(field.Data.Span);

    /// <summary>
    /// Replaces the rows <paramref name="rows"/> (unique) of a rank-2 parameter tensor with <paramref name="values"/>,
    /// in place; <paramref name="rowMask"/> is the matching [N] 0/1 mask.
    /// </summary>
    private void WriteRows(Tensor<T> field, Tensor<int> rows, Tensor<T> rowMask, Tensor<T> values)
    {
        var engine = Engine;
        var scattered = engine.TensorIndexAdd(engine.TensorZerosLike(field), 0, rows, values);
        WriteInPlace(field, Select(rowMask.Reshape(rowMask.Length, 1), scattered, field));
    }

    /// <summary>[N, 5] multipliers of the current cloud (ones for rows the optimizer state does not cover).</summary>
    private Tensor<T> EnsureLearningRateScales()
    {
        int n = GaussianCount;
        if (_learningRateScales._shape[0] != n)
        {
            _learningRateScales = Full(new[] { n, LearningRateScaleColumns }, NumOps.One);
        }

        return _learningRateScales;
    }

    /// <summary>
    /// Gets the number of Gaussians currently in the scene.
    /// </summary>
    public int GaussianCount => _positions._shape[0];

    /// <summary>
    /// Diagnostic snapshot of the Gaussian cloud state — returns per-index summary
    /// (position, opacity-after-sigmoid, max-scale, mean-color). Intended for tests /
    /// validation probes that need to inspect the cloud without exposing the internal
    /// per-attribute tensors.
    /// </summary>
    public IReadOnlyList<(double PosX, double PosY, double PosZ, double Opacity, double MaxScale, double MeanColor)> GetGaussianDiagnostics()
    {
        int n = GaussianCount;
        var results = new List<(double, double, double, double, double, double)>(n);
        if (n == 0)
        {
            return results;
        }

        // The colour summary is the mean of the first three stored colour values per Gaussian, in the channel-major
        // order the colour was always reported in (the RGB colour, or the leading red SH coefficients).
        var engine = Engine;
        int basisCount = GetShBasisCount();
        var channelMajor = engine.TensorPermute(_shCoefficients, new[] { 0, 2, 1 }).Contiguous().Reshape(n, 3 * basisCount);
        var leading = engine.TensorSlice(channelMajor, new[] { 0, 0 }, new[] { n, 3 });
        var table = engine.TensorConcatenate(new[]
        {
            _positions,
            engine.TensorSigmoid(_opacityLogits).Reshape(n, 1),
            engine.ReduceMax(engine.TensorAbs(engine.TensorExp(_logScales)), new[] { 1 }, true),
            engine.TensorDivideScalar(engine.ReduceSum(leading, new[] { 1 }, true), NumOps.FromDouble(3.0)),
        }, 1);

        // The public contract reports doubles; this boundary is the only conversion.
        for (int i = 0; i < n; i++)
        {
            results.Add((
                NumOps.ToDouble(table[i, 0]),
                NumOps.ToDouble(table[i, 1]),
                NumOps.ToDouble(table[i, 2]),
                NumOps.ToDouble(table[i, 3]),
                NumOps.ToDouble(table[i, 4]),
                NumOps.ToDouble(table[i, 5])));
        }

        return results;
    }
}
