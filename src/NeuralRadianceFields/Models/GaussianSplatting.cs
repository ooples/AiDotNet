using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralRadianceFields.Interfaces;

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
///      - Alpha blending: C = Σ α_i * c_i * Π(1 - α_j) for j<i
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
public class GaussianSplatting<T> : NeuralNetworkBase<T>, IRadianceField<T>
{
    /// <summary>
    /// Represents a single 3D Gaussian in the scene.
    /// </summary>
    private sealed class Gaussian
    {
        public Gaussian(int colorDim, INumericOperations<T> numOps)
        {
            if (colorDim <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(colorDim), "Color dimension must be positive.");
            }
            if (numOps == null)
            {
                throw new ArgumentNullException(nameof(numOps));
            }

            Position = new Vector<T>(3);
            Rotation = new Vector<T>(4);
            Scale = new Vector<T>(3);
            Color = new Vector<T>(colorDim);
            Opacity = numOps.FromDouble(0.0);
        }

        public Vector<T> Position { get; }
        public Vector<T> Rotation { get; }
        public Vector<T> Scale { get; }
        public Vector<T> Color { get; }
        public T Opacity { get; set; }

        // Derived: Covariance matrix Σ = R * S * S^T * R^T
        public Matrix<T>? Covariance { get; set; }
        public Matrix<T>? CovarianceInverse { get; set; }
    }

    private sealed class CameraGaussian
    {
        public CameraGaussian(
            Gaussian source,
            Vector<T> position,
            Matrix<T> covariance,
            Vector<T> color,
            T opacity)
        {
            Source = source ?? throw new ArgumentNullException(nameof(source));
            Position = position ?? throw new ArgumentNullException(nameof(position));
            Covariance = covariance ?? throw new ArgumentNullException(nameof(covariance));
            Color = color ?? throw new ArgumentNullException(nameof(color));
            Opacity = opacity;
        }

        public Gaussian Source { get; }
        public Vector<T> Position { get; }
        public Matrix<T> Covariance { get; }
        public Vector<T> Color { get; }
        public T Opacity { get; }
    }

    private sealed class ProjectedGaussian
    {
        public ProjectedGaussian(
            Gaussian source,
            double meanX,
            double meanY,
            double depth,
            double invA,
            double invB,
            double invC,
            double opacity,
            double colorR,
            double colorG,
            double colorB,
            int minX,
            int maxX,
            int minY,
            int maxY,
            double camX,
            double camY,
            double camZ,
            double focalLength)
        {
            Source = source ?? throw new ArgumentNullException(nameof(source));
            MeanX = meanX;
            MeanY = meanY;
            Depth = depth;
            InvA = invA;
            InvB = invB;
            InvC = invC;
            Opacity = opacity;
            ColorR = colorR;
            ColorG = colorG;
            ColorB = colorB;
            MinX = minX;
            MaxX = maxX;
            MinY = minY;
            MaxY = maxY;
            CamX = camX;
            CamY = camY;
            CamZ = camZ;
            FocalLength = focalLength;
        }

        public Gaussian Source { get; }
        public double MeanX { get; }
        public double MeanY { get; }
        public double Depth { get; }
        public double InvA { get; }
        public double InvB { get; }
        public double InvC { get; }
        public double Opacity { get; }
        public double ColorR { get; }
        public double ColorG { get; }
        public double ColorB { get; }
        public int MinX { get; }
        public int MaxX { get; }
        public int MinY { get; }
        public int MaxY { get; }
        public double CamX { get; }
        public double CamY { get; }
        public double CamZ { get; }
        public double FocalLength { get; }
    }

    private sealed class SpatialHashGrid
    {
        private readonly Dictionary<(int X, int Y, int Z), List<int>> _cells;
        private readonly double _cellSize;
        private readonly double _invCellSize;
        private readonly double _minX;
        private readonly double _minY;
        private readonly double _minZ;
        private readonly int _radius;

        public SpatialHashGrid(
            List<Gaussian> gaussians,
            double cellSize,
            int radius,
            AiDotNet.Tensors.Interfaces.INumericOperations<T> numOps)
        {
            _cells = new Dictionary<(int, int, int), List<int>>(gaussians.Count);
            _cellSize = cellSize;
            _invCellSize = cellSize > 0.0 ? 1.0 / cellSize : 0.0;
            _radius = Math.Max(1, radius);

            double minX = double.PositiveInfinity;
            double minY = double.PositiveInfinity;
            double minZ = double.PositiveInfinity;

            for (int i = 0; i < gaussians.Count; i++)
            {
                var pos = gaussians[i].Position;
                minX = Math.Min(minX, numOps.ToDouble(pos[0]));
                minY = Math.Min(minY, numOps.ToDouble(pos[1]));
                minZ = Math.Min(minZ, numOps.ToDouble(pos[2]));
            }

            if (double.IsInfinity(minX) || double.IsInfinity(minY) || double.IsInfinity(minZ))
            {
                minX = 0.0;
                minY = 0.0;
                minZ = 0.0;
            }

            _minX = minX;
            _minY = minY;
            _minZ = minZ;

            for (int i = 0; i < gaussians.Count; i++)
            {
                var pos = gaussians[i].Position;
                int ix = (int)Math.Floor((numOps.ToDouble(pos[0]) - _minX) * _invCellSize);
                int iy = (int)Math.Floor((numOps.ToDouble(pos[1]) - _minY) * _invCellSize);
                int iz = (int)Math.Floor((numOps.ToDouble(pos[2]) - _minZ) * _invCellSize);
                var key = (ix, iy, iz);
                if (!_cells.TryGetValue(key, out var list))
                {
                    list = [];
                    _cells[key] = list;
                }
                list.Add(i);
            }
        }

        public IEnumerable<int> Query(double x, double y, double z)
        {
            if (_cellSize <= 0.0)
            {
                yield break;
            }

            int ix = (int)Math.Floor((x - _minX) * _invCellSize);
            int iy = (int)Math.Floor((y - _minY) * _invCellSize);
            int iz = (int)Math.Floor((z - _minZ) * _invCellSize);

            for (int dx = -_radius; dx <= _radius; dx++)
            {
                for (int dy = -_radius; dy <= _radius; dy++)
                {
                    for (int dz = -_radius; dz <= _radius; dz++)
                    {
                        if (_cells.TryGetValue((ix + dx, iy + dy, iz + dz), out var list))
                        {
                            for (int i = 0; i < list.Count; i++)
                            {
                                yield return list[i];
                            }
                        }
                    }
                }
            }
        }
    }

    private readonly struct GaussianGradient
    {
        public GaussianGradient(Gaussian gaussian, double gradX, double gradY, double gradZ)
        {
            Gaussian = gaussian;
            GradX = gradX;
            GradY = gradY;
            GradZ = gradZ;
        }

        public Gaussian Gaussian { get; }
        public double GradX { get; }
        public double GradY { get; }
        public double GradZ { get; }

        public double Magnitude => Math.Sqrt(GradX * GradX + GradY * GradY + GradZ * GradZ);
    }

    private readonly List<Gaussian> _gaussians;
    private readonly bool _useSphericalHarmonics;
    private readonly int _shDegree; // Degree of spherical harmonics (0-3)
    private int _trainingStep;
    private SpatialHashGrid? _spatialIndex;
    private bool _spatialIndexDirty = true;
    private Tensor<T>? _lastQueryPositions;
    private Tensor<T>? _lastQueryDirections;

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

        _gaussians = [];
        _useSphericalHarmonics = options.UseSphericalHarmonics;
        _shDegree = options.ShDegree;

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
    /// <param name="useSphericalHarmonics">Whether to use spherical harmonics for view-dependent appearance.</param>
    /// <param name="shDegree">Degree of spherical harmonics (0-3, higher = more view dependence).</param>
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
    ///     useSphericalHarmonics: true,
    ///     shDegree: 3  // High quality view-dependent effects
    /// );
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
        bool useSphericalHarmonics = true,
        int shDegree = 3,
        ILossFunction<T>? lossFunction = null)
        : this(
            new GaussianSplattingOptions
            {
                UseSphericalHarmonics = useSphericalHarmonics,
                ShDegree = shDegree
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
        int basisCount = GetShBasisCount();
        int colorDim = _useSphericalHarmonics ? 3 * basisCount : 3;
        var numOps = NumOps;
        const double shBase = 0.282095;
        var positionsX = new double[numPoints];
        var positionsY = new double[numPoints];
        var positionsZ = new double[numPoints];
        double minX = double.PositiveInfinity;
        double minY = double.PositiveInfinity;
        double minZ = double.PositiveInfinity;
        double maxX = double.NegativeInfinity;
        double maxY = double.NegativeInfinity;
        double maxZ = double.NegativeInfinity;

        for (int i = 0; i < numPoints; i++)
        {
            var gaussian = new Gaussian(colorDim, numOps);
            double x = numOps.ToDouble(pointCloud[i, 0]);
            double y = numOps.ToDouble(pointCloud[i, 1]);
            double z = numOps.ToDouble(pointCloud[i, 2]);
            positionsX[i] = x;
            positionsY[i] = y;
            positionsZ[i] = z;
            minX = Math.Min(minX, x);
            minY = Math.Min(minY, y);
            minZ = Math.Min(minZ, z);
            maxX = Math.Max(maxX, x);
            maxY = Math.Max(maxY, y);
            maxZ = Math.Max(maxZ, z);

            gaussian.Position[0] = numOps.FromDouble(x);
            gaussian.Position[1] = numOps.FromDouble(y);
            gaussian.Position[2] = numOps.FromDouble(z);
            gaussian.Rotation[0] = numOps.FromDouble(1.0);
            gaussian.Rotation[1] = numOps.FromDouble(0.0);
            gaussian.Rotation[2] = numOps.FromDouble(0.0);
            gaussian.Rotation[3] = numOps.FromDouble(0.0);
            gaussian.Scale[0] = numOps.FromDouble(MinScale);
            gaussian.Scale[1] = numOps.FromDouble(MinScale);
            gaussian.Scale[2] = numOps.FromDouble(MinScale);

            double r = colors != null ? numOps.ToDouble(colors[i, 0]) : Random.NextDouble();
            double g = colors != null ? numOps.ToDouble(colors[i, 1]) : Random.NextDouble();
            double b = colors != null ? numOps.ToDouble(colors[i, 2]) : Random.NextDouble();

            if (_useSphericalHarmonics)
            {
                gaussian.Color[0] = numOps.FromDouble(r / shBase);
                gaussian.Color[basisCount] = numOps.FromDouble(g / shBase);
                gaussian.Color[2 * basisCount] = numOps.FromDouble(b / shBase);
            }
            else
            {
                gaussian.Color[0] = numOps.FromDouble(r);
                gaussian.Color[1] = numOps.FromDouble(g);
                gaussian.Color[2] = numOps.FromDouble(b);
            }

            _gaussians.Add(gaussian);
        }

        double extentX = maxX - minX;
        double extentY = maxY - minY;
        double extentZ = maxZ - minZ;
        double volume = extentX * extentY * extentZ;
        double baseSpacing;
        if (volume > 0.0)
        {
            baseSpacing = Math.Pow(volume / numPoints, 1.0 / 3.0);
        }
        else
        {
            double maxExtent = Math.Max(extentX, Math.Max(extentY, extentZ));
            baseSpacing = maxExtent > 0.0 ? maxExtent / Math.Sqrt(numPoints) : 0.0;
        }

        if (baseSpacing <= 0.0)
        {
            baseSpacing = DefaultPointSpacing;
        }

        double cellSize = Math.Max(baseSpacing, MinScale * InitialNeighborSearchScale);
        double invCellSize = cellSize > 0.0 ? 1.0 / cellSize : 0.0;
        var grid = new Dictionary<(int X, int Y, int Z), List<int>>(numPoints);
        for (int i = 0; i < numPoints; i++)
        {
            int ix = (int)Math.Floor((positionsX[i] - minX) * invCellSize);
            int iy = (int)Math.Floor((positionsY[i] - minY) * invCellSize);
            int iz = (int)Math.Floor((positionsZ[i] - minZ) * invCellSize);
            var key = (ix, iy, iz);
            if (!grid.TryGetValue(key, out var list))
            {
                list = [];
                grid[key] = list;
            }
            list.Add(i);
        }

        for (int i = 0; i < numPoints; i++)
        {
            double nearestSq = double.PositiveInfinity;
            int ix = (int)Math.Floor((positionsX[i] - minX) * invCellSize);
            int iy = (int)Math.Floor((positionsY[i] - minY) * invCellSize);
            int iz = (int)Math.Floor((positionsZ[i] - minZ) * invCellSize);

            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        if (!grid.TryGetValue((ix + dx, iy + dy, iz + dz), out var list))
                        {
                            continue;
                        }

                        for (int idx = 0; idx < list.Count; idx++)
                        {
                            int neighbor = list[idx];
                            if (neighbor == i)
                            {
                                continue;
                            }

                            double ox = positionsX[i] - positionsX[neighbor];
                            double oy = positionsY[i] - positionsY[neighbor];
                            double oz = positionsZ[i] - positionsZ[neighbor];
                            double distSq = ox * ox + oy * oy + oz * oz;
                            if (distSq < nearestSq)
                            {
                                nearestSq = distSq;
                            }
                        }
                    }
                }
            }

            double neighborDistance = nearestSq < double.PositiveInfinity
                ? Math.Sqrt(nearestSq)
                : baseSpacing;
            if (neighborDistance <= 0.0)
            {
                neighborDistance = baseSpacing;
            }

            double scale = Math.Max(MinScale, neighborDistance * InitialScaleMultiplier);
            var gaussian = _gaussians[i];
            gaussian.Scale[0] = numOps.FromDouble(scale);
            gaussian.Scale[1] = numOps.FromDouble(scale);
            gaussian.Scale[2] = numOps.FromDouble(scale);
            ComputeCovariance(gaussian);
        }

        MarkSpatialIndexDirty();
    }

    private void ComputeCovariance(Gaussian gaussian)
    {
        double qw = NumOps.ToDouble(gaussian.Rotation[0]);
        double qx = NumOps.ToDouble(gaussian.Rotation[1]);
        double qy = NumOps.ToDouble(gaussian.Rotation[2]);
        double qz = NumOps.ToDouble(gaussian.Rotation[3]);

        double qNorm = Math.Sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        if (qNorm > 0.0)
        {
            double inv = 1.0 / qNorm;
            qw *= inv;
            qx *= inv;
            qy *= inv;
            qz *= inv;
        }

        double xx = qx * qx;
        double yy = qy * qy;
        double zz = qz * qz;
        double xy = qx * qy;
        double xz = qx * qz;
        double yz = qy * qz;
        double wx = qw * qx;
        double wy = qw * qy;
        double wz = qw * qz;

        double r00 = 1.0 - 2.0 * (yy + zz);
        double r01 = 2.0 * (xy - wz);
        double r02 = 2.0 * (xz + wy);
        double r10 = 2.0 * (xy + wz);
        double r11 = 1.0 - 2.0 * (xx + zz);
        double r12 = 2.0 * (yz - wx);
        double r20 = 2.0 * (xz - wy);
        double r21 = 2.0 * (yz + wx);
        double r22 = 1.0 - 2.0 * (xx + yy);

        double sx = Math.Max(1e-6, Math.Abs(NumOps.ToDouble(gaussian.Scale[0])));
        double sy = Math.Max(1e-6, Math.Abs(NumOps.ToDouble(gaussian.Scale[1])));
        double sz = Math.Max(1e-6, Math.Abs(NumOps.ToDouble(gaussian.Scale[2])));
        double sx2 = sx * sx;
        double sy2 = sy * sy;
        double sz2 = sz * sz;

        double m00 = r00 * sx2;
        double m01 = r01 * sy2;
        double m02 = r02 * sz2;
        double m10 = r10 * sx2;
        double m11 = r11 * sy2;
        double m12 = r12 * sz2;
        double m20 = r20 * sx2;
        double m21 = r21 * sy2;
        double m22 = r22 * sz2;

        double cov00 = m00 * r00 + m01 * r01 + m02 * r02;
        double cov01 = m00 * r10 + m01 * r11 + m02 * r12;
        double cov02 = m00 * r20 + m01 * r21 + m02 * r22;
        double cov11 = m10 * r10 + m11 * r11 + m12 * r12;
        double cov12 = m10 * r20 + m11 * r21 + m12 * r22;
        double cov22 = m20 * r20 + m21 * r21 + m22 * r22;

        const double eps = 1e-6;
        cov00 += eps;
        cov11 += eps;
        cov22 += eps;

        var covariance = new Matrix<T>(3, 3);
        covariance[0, 0] = NumOps.FromDouble(cov00);
        covariance[0, 1] = NumOps.FromDouble(cov01);
        covariance[0, 2] = NumOps.FromDouble(cov02);
        covariance[1, 0] = NumOps.FromDouble(cov01);
        covariance[1, 1] = NumOps.FromDouble(cov11);
        covariance[1, 2] = NumOps.FromDouble(cov12);
        covariance[2, 0] = NumOps.FromDouble(cov02);
        covariance[2, 1] = NumOps.FromDouble(cov12);
        covariance[2, 2] = NumOps.FromDouble(cov22);
        gaussian.Covariance = covariance;

        double det = cov00 * (cov11 * cov22 - cov12 * cov12)
                     - cov01 * (cov01 * cov22 - cov12 * cov02)
                     + cov02 * (cov01 * cov12 - cov11 * cov02);
        if (Math.Abs(det) < 1e-12)
        {
            det = det >= 0.0 ? 1e-12 : -1e-12;
        }

        double inv00 = (cov11 * cov22 - cov12 * cov12) / det;
        double inv01 = (cov02 * cov12 - cov01 * cov22) / det;
        double inv02 = (cov01 * cov12 - cov02 * cov11) / det;
        double inv11 = (cov00 * cov22 - cov02 * cov02) / det;
        double inv12 = (cov02 * cov01 - cov00 * cov12) / det;
        double inv22 = (cov00 * cov11 - cov01 * cov01) / det;

        var inverse = new Matrix<T>(3, 3);
        inverse[0, 0] = NumOps.FromDouble(inv00);
        inverse[0, 1] = NumOps.FromDouble(inv01);
        inverse[0, 2] = NumOps.FromDouble(inv02);
        inverse[1, 0] = NumOps.FromDouble(inv01);
        inverse[1, 1] = NumOps.FromDouble(inv11);
        inverse[1, 2] = NumOps.FromDouble(inv12);
        inverse[2, 0] = NumOps.FromDouble(inv02);
        inverse[2, 1] = NumOps.FromDouble(inv12);
        inverse[2, 2] = NumOps.FromDouble(inv22);
        gaussian.CovarianceInverse = inverse;
    }

    private void EnsureSpatialIndex()
    {
        if (!EnableSpatialIndex || _gaussians.Count == 0)
        {
            _spatialIndex = null;
            _spatialIndexDirty = false;
            return;
        }

        if (!_spatialIndexDirty && _spatialIndex != null)
        {
            return;
        }

        _spatialIndex = BuildSpatialIndex();
        _spatialIndexDirty = false;
    }

    private SpatialHashGrid BuildSpatialIndex()
    {
        double avgScale = 0.0;
        double maxScale = 0.0;
        for (int i = 0; i < _gaussians.Count; i++)
        {
            var scale = _gaussians[i].Scale;
            double sx = Math.Abs(NumOps.ToDouble(scale[0]));
            double sy = Math.Abs(NumOps.ToDouble(scale[1]));
            double sz = Math.Abs(NumOps.ToDouble(scale[2]));
            double max = Math.Max(sx, Math.Max(sy, sz));
            avgScale += max;
            maxScale = Math.Max(maxScale, max);
        }

        avgScale = _gaussians.Count > 0 ? avgScale / _gaussians.Count : 0.0;
        if (avgScale <= 0.0)
        {
            avgScale = 0.01;
        }

        double cellSize = Math.Max(avgScale * 2.5, Math.Max(maxScale, MinScale));
        int radius = Math.Max(SpatialIndexRadius, (int)Math.Ceiling(maxScale * 3.0 / cellSize));
        return new SpatialHashGrid(_gaussians, cellSize, radius, NumOps);
    }

    private IEnumerable<Gaussian> GetCandidateGaussians(double x, double y, double z)
    {
        EnsureSpatialIndex();
        if (_spatialIndex == null)
        {
            for (int i = 0; i < _gaussians.Count; i++)
            {
                yield return _gaussians[i];
            }
            yield break;
        }

        foreach (int idx in _spatialIndex.Query(x, y, z))
        {
            if ((uint)idx < (uint)_gaussians.Count)
            {
                yield return _gaussians[idx];
            }
        }
    }

    private void MarkSpatialIndexDirty()
    {
        _spatialIndexDirty = true;
    }

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

        int numPoints = positions.Shape[0];
        var rgb = new T[numPoints * 3];
        var density = new T[numPoints];
        var posData = positions.Data.Span;
        var dirData = viewingDirections.Data.Span;
        EnsureSpatialIndex();
        var spatialIndex = _spatialIndex;

        for (int i = 0; i < numPoints; i++)
        {
            double px = NumOps.ToDouble(posData[i * 3]);
            double py = NumOps.ToDouble(posData[i * 3 + 1]);
            double pz = NumOps.ToDouble(posData[i * 3 + 2]);

            double dx = NumOps.ToDouble(dirData[i * 3]);
            double dy = NumOps.ToDouble(dirData[i * 3 + 1]);
            double dz = NumOps.ToDouble(dirData[i * 3 + 2]);
            NormalizeDirection(ref dx, ref dy, ref dz);

            double accumR = 0.0;
            double accumG = 0.0;
            double accumB = 0.0;
            double accumDensity = 0.0;

            if (spatialIndex == null)
            {
                for (int gIdx = 0; gIdx < _gaussians.Count; gIdx++)
                {
                    var gaussian = _gaussians[gIdx];
                    if (gaussian.CovarianceInverse == null)
                    {
                        ComputeCovariance(gaussian);
                    }

                    double gx = NumOps.ToDouble(gaussian.Position[0]);
                    double gy = NumOps.ToDouble(gaussian.Position[1]);
                    double gz = NumOps.ToDouble(gaussian.Position[2]);

                    double ox = px - gx;
                    double oy = py - gy;
                    double oz = pz - gz;

                    var inv = gaussian.CovarianceInverse;
                    if (inv == null)
                    {
                        throw new InvalidOperationException("Gaussian covariance inverse is not initialized.");
                    }
                    double inv00 = NumOps.ToDouble(inv[0, 0]);
                    double inv01 = NumOps.ToDouble(inv[0, 1]);
                    double inv02 = NumOps.ToDouble(inv[0, 2]);
                    double inv11 = NumOps.ToDouble(inv[1, 1]);
                    double inv12 = NumOps.ToDouble(inv[1, 2]);
                    double inv22 = NumOps.ToDouble(inv[2, 2]);

                    double q =
                        ox * (inv00 * ox + inv01 * oy + inv02 * oz) +
                        oy * (inv01 * ox + inv11 * oy + inv12 * oz) +
                        oz * (inv02 * ox + inv12 * oy + inv22 * oz);

                    double weight = Math.Exp(-0.5 * q);
                    double alpha = Sigmoid(NumOps.ToDouble(gaussian.Opacity)) * weight;
                    if (alpha <= 1e-8)
                    {
                        continue;
                    }

                    var (r, g, b) = EvaluateGaussianColor(gaussian, dx, dy, dz);
                    accumDensity += alpha;
                    accumR += alpha * r;
                    accumG += alpha * g;
                    accumB += alpha * b;
                }
            }
            else
            {
                foreach (int gIdx in spatialIndex.Query(px, py, pz))
                {
                    if ((uint)gIdx >= (uint)_gaussians.Count)
                    {
                        continue;
                    }

                    var gaussian = _gaussians[gIdx];
                    if (gaussian.CovarianceInverse == null)
                    {
                        ComputeCovariance(gaussian);
                    }

                    double gx = NumOps.ToDouble(gaussian.Position[0]);
                    double gy = NumOps.ToDouble(gaussian.Position[1]);
                    double gz = NumOps.ToDouble(gaussian.Position[2]);

                    double ox = px - gx;
                    double oy = py - gy;
                    double oz = pz - gz;

                    var inv = gaussian.CovarianceInverse;
                    if (inv == null)
                    {
                        throw new InvalidOperationException("Gaussian covariance inverse is not initialized.");
                    }
                    double inv00 = NumOps.ToDouble(inv[0, 0]);
                    double inv01 = NumOps.ToDouble(inv[0, 1]);
                    double inv02 = NumOps.ToDouble(inv[0, 2]);
                    double inv11 = NumOps.ToDouble(inv[1, 1]);
                    double inv12 = NumOps.ToDouble(inv[1, 2]);
                    double inv22 = NumOps.ToDouble(inv[2, 2]);

                    double q =
                        ox * (inv00 * ox + inv01 * oy + inv02 * oz) +
                        oy * (inv01 * ox + inv11 * oy + inv12 * oz) +
                        oz * (inv02 * ox + inv12 * oy + inv22 * oz);

                    double weight = Math.Exp(-0.5 * q);
                    double alpha = Sigmoid(NumOps.ToDouble(gaussian.Opacity)) * weight;
                    if (alpha <= 1e-8)
                    {
                        continue;
                    }

                    var (r, g, b) = EvaluateGaussianColor(gaussian, dx, dy, dz);
                    accumDensity += alpha;
                    accumR += alpha * r;
                    accumG += alpha * g;
                    accumB += alpha * b;
                }
            }

            if (accumDensity > 1e-8)
            {
                accumR /= accumDensity;
                accumG /= accumDensity;
                accumB /= accumDensity;
            }

            int baseIdx = i * 3;
            rgb[baseIdx] = NumOps.FromDouble(Clamp01(accumR));
            rgb[baseIdx + 1] = NumOps.FromDouble(Clamp01(accumG));
            rgb[baseIdx + 2] = NumOps.FromDouble(Clamp01(accumB));
            density[i] = NumOps.FromDouble(accumDensity);
        }

        return (new Tensor<T>(rgb, [numPoints, 3]), new Tensor<T>(density, [numPoints, 1]));
    }

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
        var image = new T[imageHeight * imageWidth * 3];
        var transformedGaussians = TransformGaussiansToCamera(cameraPosition, cameraRotation);
        var projected2DGaussians = ProjectGaussiansTo2D(transformedGaussians, imageWidth, imageHeight, focalLength);
        RasterizeGaussians(projected2DGaussians, image, imageWidth, imageHeight, clampOutput);

        return new Tensor<T>(image, [imageHeight, imageWidth, 3]);
    }

    private List<CameraGaussian> TransformGaussiansToCamera(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation)
    {
        var transformed = new List<CameraGaussian>(_gaussians.Count);

        double r00 = NumOps.ToDouble(cameraRotation[0, 0]);
        double r01 = NumOps.ToDouble(cameraRotation[0, 1]);
        double r02 = NumOps.ToDouble(cameraRotation[0, 2]);
        double r10 = NumOps.ToDouble(cameraRotation[1, 0]);
        double r11 = NumOps.ToDouble(cameraRotation[1, 1]);
        double r12 = NumOps.ToDouble(cameraRotation[1, 2]);
        double r20 = NumOps.ToDouble(cameraRotation[2, 0]);
        double r21 = NumOps.ToDouble(cameraRotation[2, 1]);
        double r22 = NumOps.ToDouble(cameraRotation[2, 2]);

        double camX = NumOps.ToDouble(cameraPosition[0]);
        double camY = NumOps.ToDouble(cameraPosition[1]);
        double camZ = NumOps.ToDouble(cameraPosition[2]);

        foreach (var gaussian in _gaussians)
        {
            if (gaussian.Covariance == null)
            {
                ComputeCovariance(gaussian);
            }

            double px = NumOps.ToDouble(gaussian.Position[0]) - camX;
            double py = NumOps.ToDouble(gaussian.Position[1]) - camY;
            double pz = NumOps.ToDouble(gaussian.Position[2]) - camZ;

            double tx = r00 * px + r01 * py + r02 * pz;
            double ty = r10 * px + r11 * py + r12 * pz;
            double tz = r20 * px + r21 * py + r22 * pz;

            var cov = gaussian.Covariance;
            if (cov == null)
            {
                throw new InvalidOperationException("Gaussian covariance is not initialized.");
            }
            double c00 = NumOps.ToDouble(cov[0, 0]);
            double c01 = NumOps.ToDouble(cov[0, 1]);
            double c02 = NumOps.ToDouble(cov[0, 2]);
            double c11 = NumOps.ToDouble(cov[1, 1]);
            double c12 = NumOps.ToDouble(cov[1, 2]);
            double c22 = NumOps.ToDouble(cov[2, 2]);

            double t00 = r00 * c00 + r01 * c01 + r02 * c02;
            double t01 = r00 * c01 + r01 * c11 + r02 * c12;
            double t02 = r00 * c02 + r01 * c12 + r02 * c22;
            double t10 = r10 * c00 + r11 * c01 + r12 * c02;
            double t11 = r10 * c01 + r11 * c11 + r12 * c12;
            double t12 = r10 * c02 + r11 * c12 + r12 * c22;
            double t20 = r20 * c00 + r21 * c01 + r22 * c02;
            double t21 = r20 * c01 + r21 * c11 + r22 * c12;
            double t22 = r20 * c02 + r21 * c12 + r22 * c22;

            double cam00 = t00 * r00 + t01 * r01 + t02 * r02;
            double cam01 = t00 * r10 + t01 * r11 + t02 * r12;
            double cam02 = t00 * r20 + t01 * r21 + t02 * r22;
            double cam11 = t10 * r10 + t11 * r11 + t12 * r12;
            double cam12 = t10 * r20 + t11 * r21 + t12 * r22;
            double cam22 = t20 * r20 + t21 * r21 + t22 * r22;

            var covCam = new Matrix<T>(3, 3);
            covCam[0, 0] = NumOps.FromDouble(cam00);
            covCam[0, 1] = NumOps.FromDouble(cam01);
            covCam[0, 2] = NumOps.FromDouble(cam02);
            covCam[1, 0] = NumOps.FromDouble(cam01);
            covCam[1, 1] = NumOps.FromDouble(cam11);
            covCam[1, 2] = NumOps.FromDouble(cam12);
            covCam[2, 0] = NumOps.FromDouble(cam02);
            covCam[2, 1] = NumOps.FromDouble(cam12);
            covCam[2, 2] = NumOps.FromDouble(cam22);

            var camPosition = new Vector<T>(3)
            {
                [0] = NumOps.FromDouble(tx),
                [1] = NumOps.FromDouble(ty),
                [2] = NumOps.FromDouble(tz)
            };

            transformed.Add(new CameraGaussian(
                source: gaussian,
                position: camPosition,
                covariance: covCam,
                color: gaussian.Color,
                opacity: gaussian.Opacity));
        }

        return transformed;
    }

    private List<ProjectedGaussian> ProjectGaussiansTo2D(
        List<CameraGaussian> gaussians3D,
        int width,
        int height,
        T focalLength)
    {
        var projected = new List<ProjectedGaussian>(gaussians3D.Count);
        double f = NumOps.ToDouble(focalLength);
        double cx = (width - 1) * 0.5;
        double cy = (height - 1) * 0.5;

        foreach (var gaussian in gaussians3D)
        {
            double x = NumOps.ToDouble(gaussian.Position[0]);
            double y = NumOps.ToDouble(gaussian.Position[1]);
            double z = NumOps.ToDouble(gaussian.Position[2]);

            if (z <= 1e-6)
            {
                continue;
            }

            double invZ = 1.0 / z;
            double meanX = f * x * invZ + cx;
            double meanY = f * y * invZ + cy;

            var cov = gaussian.Covariance;
            double s00 = NumOps.ToDouble(cov[0, 0]);
            double s01 = NumOps.ToDouble(cov[0, 1]);
            double s02 = NumOps.ToDouble(cov[0, 2]);
            double s11 = NumOps.ToDouble(cov[1, 1]);
            double s12 = NumOps.ToDouble(cov[1, 2]);
            double s22 = NumOps.ToDouble(cov[2, 2]);

            double j00 = f * invZ;
            double j02 = -f * x * invZ * invZ;
            double j11 = f * invZ;
            double j12 = -f * y * invZ * invZ;

            double v0x = j00;
            double v0y = 0.0;
            double v0z = j02;
            double v1x = 0.0;
            double v1y = j11;
            double v1z = j12;

            double t0x = s00 * v0x + s01 * v0y + s02 * v0z;
            double t0y = s01 * v0x + s11 * v0y + s12 * v0z;
            double t0z = s02 * v0x + s12 * v0y + s22 * v0z;
            double t1x = s00 * v1x + s01 * v1y + s02 * v1z;
            double t1y = s01 * v1x + s11 * v1y + s12 * v1z;
            double t1z = s02 * v1x + s12 * v1y + s22 * v1z;

            double a = v0x * t0x + v0y * t0y + v0z * t0z;
            double b = v0x * t1x + v0y * t1y + v0z * t1z;
            double c = v1x * t1x + v1y * t1y + v1z * t1z;

            const double eps = 1e-6;
            a += eps;
            c += eps;

            double det = a * c - b * b;
            if (det <= 1e-12)
            {
                continue;
            }

            double invDet = 1.0 / det;
            double invA = c * invDet;
            double invB = -b * invDet;
            double invC = a * invDet;

            double trace = a + c;
            double disc = trace * trace - 4.0 * det;
            if (disc < 0.0)
            {
                disc = 0.0;
            }
            double maxEigen = 0.5 * (trace + Math.Sqrt(disc));
            double radius = 3.0 * Math.Sqrt(Math.Max(maxEigen, 0.0));
            if (radius < 1.0)
            {
                radius = 1.0;
            }

            int minX = (int)Math.Floor(meanX - radius);
            int maxX = (int)Math.Ceiling(meanX + radius);
            int minY = (int)Math.Floor(meanY - radius);
            int maxY = (int)Math.Ceiling(meanY + radius);

            if (maxX < 0 || maxY < 0 || minX >= width || minY >= height)
            {
                continue;
            }

            minX = Math.Max(0, minX);
            maxX = Math.Min(width - 1, maxX);
            minY = Math.Max(0, minY);
            maxY = Math.Min(height - 1, maxY);

            double dirX = -x;
            double dirY = -y;
            double dirZ = -z;
            NormalizeDirection(ref dirX, ref dirY, ref dirZ);

            var (r, g, bColor) = EvaluateGaussianColor(gaussian.Source, dirX, dirY, dirZ);
            double opacity = Sigmoid(NumOps.ToDouble(gaussian.Opacity));

            projected.Add(new ProjectedGaussian(
                source: gaussian.Source,
                meanX: meanX,
                meanY: meanY,
                depth: z,
                invA: invA,
                invB: invB,
                invC: invC,
                opacity: opacity,
                colorR: r,
                colorG: g,
                colorB: bColor,
                minX: minX,
                maxX: maxX,
                minY: minY,
                maxY: maxY,
                camX: x,
                camY: y,
                camZ: z,
                focalLength: f));
        }

        return projected;
    }

    private static List<int>[] BuildTileBuckets(
        List<ProjectedGaussian> gaussians2D,
        int tilesX,
        int tilesY,
        int tileSize)
    {
        var buckets = new List<int>[tilesX * tilesY];
        for (int i = 0; i < gaussians2D.Count; i++)
        {
            var gaussian = gaussians2D[i];
            int tileMinX = gaussian.MinX / tileSize;
            int tileMaxX = gaussian.MaxX / tileSize;
            int tileMinY = gaussian.MinY / tileSize;
            int tileMaxY = gaussian.MaxY / tileSize;

            tileMinX = Math.Max(0, Math.Min(tilesX - 1, tileMinX));
            tileMaxX = Math.Max(0, Math.Min(tilesX - 1, tileMaxX));
            tileMinY = Math.Max(0, Math.Min(tilesY - 1, tileMinY));
            tileMaxY = Math.Max(0, Math.Min(tilesY - 1, tileMaxY));

            for (int ty = tileMinY; ty <= tileMaxY; ty++)
            {
                int row = ty * tilesX;
                for (int tx = tileMinX; tx <= tileMaxX; tx++)
                {
                    int idx = row + tx;
                    if (buckets[idx] == null)
                    {
                        buckets[idx] = [];
                    }
                    buckets[idx].Add(i);
                }
            }
        }

        return buckets;
    }

    private void RasterizeGaussians(List<ProjectedGaussian> gaussians2D, T[] image, int width, int height, bool clampOutput)
    {
        if (gaussians2D.Count == 0 || width <= 0 || height <= 0)
        {
            return;
        }

        int tileSize = Math.Max(4, TileSize);
        int tilesX = (width + tileSize - 1) / tileSize;
        int tilesY = (height + tileSize - 1) / tileSize;
        var buckets = BuildTileBuckets(gaussians2D, tilesX, tilesY, tileSize);

        System.Threading.Tasks.Parallel.For(0, tilesX * tilesY, tileIdx =>
        {
            var bucket = buckets[tileIdx];
            if (bucket == null || bucket.Count == 0)
            {
                return;
            }

            bucket.Sort((a, b) => gaussians2D[a].Depth.CompareTo(gaussians2D[b].Depth));

            int tileX = tileIdx % tilesX;
            int tileY = tileIdx / tilesX;
            int xStart = tileX * tileSize;
            int yStart = tileY * tileSize;
            int xEnd = Math.Min(width, xStart + tileSize) - 1;
            int yEnd = Math.Min(height, yStart + tileSize) - 1;
            int tileWidth = xEnd - xStart + 1;
            int tileHeight = yEnd - yStart + 1;
            int tilePixelCount = tileWidth * tileHeight;

            var accumR = new double[tilePixelCount];
            var accumG = new double[tilePixelCount];
            var accumB = new double[tilePixelCount];
            var transmittance = new double[tilePixelCount];

            for (int i = 0; i < tilePixelCount; i++)
            {
                transmittance[i] = 1.0;
            }

            for (int i = 0; i < bucket.Count; i++)
            {
                var gaussian = gaussians2D[bucket[i]];
                int minX = Math.Max(gaussian.MinX, xStart);
                int maxX = Math.Min(gaussian.MaxX, xEnd);
                int minY = Math.Max(gaussian.MinY, yStart);
                int maxY = Math.Min(gaussian.MaxY, yEnd);

                for (int y = minY; y <= maxY; y++)
                {
                    int localRow = (y - yStart) * tileWidth;
                    for (int x = minX; x <= maxX; x++)
                    {
                        int localIdx = localRow + (x - xStart);
                        double t = transmittance[localIdx];
                        if (t < 1e-4)
                        {
                            continue;
                        }

                        double dx = x - gaussian.MeanX;
                        double dy = y - gaussian.MeanY;
                        double exponent =
                            -0.5 * (gaussian.InvA * dx * dx + 2.0 * gaussian.InvB * dx * dy + gaussian.InvC * dy * dy);
                        if (exponent < -20.0)
                        {
                            continue;
                        }

                        double weight = Math.Exp(exponent);
                        double alpha = gaussian.Opacity * weight;
                        if (alpha < 1e-6)
                        {
                            continue;
                        }

                        accumR[localIdx] += t * alpha * gaussian.ColorR;
                        accumG[localIdx] += t * alpha * gaussian.ColorG;
                        accumB[localIdx] += t * alpha * gaussian.ColorB;
                        transmittance[localIdx] = t * (1.0 - alpha);
                    }
                }
            }

            for (int y = 0; y < tileHeight; y++)
            {
                int globalRow = (yStart + y) * width;
                int localRow = y * tileWidth;
                for (int x = 0; x < tileWidth; x++)
                {
                    int globalIdx = globalRow + (xStart + x);
                    int baseIdx = globalIdx * 3;
                    int localIdx = localRow + x;
                    double r = accumR[localIdx];
                    double g = accumG[localIdx];
                    double b = accumB[localIdx];
                    if (clampOutput)
                    {
                        r = Clamp01(r);
                        g = Clamp01(g);
                        b = Clamp01(b);
                    }

                    image[baseIdx] = NumOps.FromDouble(r);
                    image[baseIdx + 1] = NumOps.FromDouble(g);
                    image[baseIdx + 2] = NumOps.FromDouble(b);
                }
            }
        });
    }

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
        var positions = new T[numRays * numSamples * 3];
        var directions = new T[numRays * numSamples * 3];

        double near = NumOps.ToDouble(nearBound);
        double far = NumOps.ToDouble(farBound);
        double step = numSamples > 1 ? (far - near) / (numSamples - 1) : 0.0;

        var originData = rayOrigins.Data.Span;
        var dirData = rayDirections.Data.Span;

        for (int r = 0; r < numRays; r++)
        {
            double ox = NumOps.ToDouble(originData[r * 3]);
            double oy = NumOps.ToDouble(originData[r * 3 + 1]);
            double oz = NumOps.ToDouble(originData[r * 3 + 2]);
            double dx = NumOps.ToDouble(dirData[r * 3]);
            double dy = NumOps.ToDouble(dirData[r * 3 + 1]);
            double dz = NumOps.ToDouble(dirData[r * 3 + 2]);

            for (int s = 0; s < numSamples; s++)
            {
                double t = near + step * s;
                int baseIdx = (r * numSamples + s) * 3;
                positions[baseIdx] = NumOps.FromDouble(ox + t * dx);
                positions[baseIdx + 1] = NumOps.FromDouble(oy + t * dy);
                positions[baseIdx + 2] = NumOps.FromDouble(oz + t * dz);
                directions[baseIdx] = dirData[r * 3];
                directions[baseIdx + 1] = dirData[r * 3 + 1];
                directions[baseIdx + 2] = dirData[r * 3 + 2];
            }
        }

        return (new Tensor<T>(positions, [numRays * numSamples, 3]),
            new Tensor<T>(directions, [numRays * numSamples, 3]));
    }

    private Tensor<T> VolumeRendering(
        Tensor<T> rgb,
        Tensor<T> density,
        int numRays,
        int numSamples,
        T nearBound,
        T farBound)
    {
        var colors = new T[numRays * 3];
        var rgbData = rgb.Data.Span;
        var densityData = density.Data.Span;

        double near = NumOps.ToDouble(nearBound);
        double far = NumOps.ToDouble(farBound);
        double deltaT = numSamples > 0 ? (far - near) / numSamples : 0.0;

        for (int r = 0; r < numRays; r++)
        {
            double transmittance = 1.0;
            double accumR = 0.0;
            double accumG = 0.0;
            double accumB = 0.0;

            for (int s = 0; s < numSamples; s++)
            {
                int idx = r * numSamples + s;
                double sigma = NumOps.ToDouble(densityData[idx]);
                double alpha = 1.0 - Math.Exp(-sigma * deltaT);
                if (alpha <= 0.0)
                {
                    continue;
                }

                int rgbIdx = idx * 3;
                double rVal = NumOps.ToDouble(rgbData[rgbIdx]);
                double gVal = NumOps.ToDouble(rgbData[rgbIdx + 1]);
                double bVal = NumOps.ToDouble(rgbData[rgbIdx + 2]);

                accumR += transmittance * alpha * rVal;
                accumG += transmittance * alpha * gVal;
                accumB += transmittance * alpha * bVal;

                transmittance *= (1.0 - alpha);
                if (transmittance < 1e-4)
                {
                    break;
                }
            }

            int outIdx = r * 3;
            colors[outIdx] = NumOps.FromDouble(accumR);
            colors[outIdx + 1] = NumOps.FromDouble(accumG);
            colors[outIdx + 2] = NumOps.FromDouble(accumB);
        }

        return new Tensor<T>(colors, [numRays, 3]);
    }

    public override Tensor<T> ForwardWithMemory(Tensor<T> input)
    {
        if (input.Shape.Length != 2 || input.Shape[1] != 6)
        {
            throw new ArgumentException("Input must have shape [N, 6] (position + direction).", nameof(input));
        }

        int numPoints = input.Shape[0];
        var positions = new T[numPoints * 3];
        var directions = new T[numPoints * 3];

        for (int i = 0; i < numPoints; i++)
        {
            int baseIdx = i * 6;
            int posIdx = i * 3;
            positions[posIdx] = input.Data.Span[baseIdx];
            positions[posIdx + 1] = input.Data.Span[baseIdx + 1];
            positions[posIdx + 2] = input.Data.Span[baseIdx + 2];
            directions[posIdx] = input.Data.Span[baseIdx + 3];
            directions[posIdx + 1] = input.Data.Span[baseIdx + 4];
            directions[posIdx + 2] = input.Data.Span[baseIdx + 5];
        }

        var posTensor = new Tensor<T>(positions, [numPoints, 3]);
        var dirTensor = new Tensor<T>(directions, [numPoints, 3]);
        var (rgb, density) = QueryField(posTensor, dirTensor);

        if (IsTrainingMode)
        {
            _lastQueryPositions = posTensor;
            _lastQueryDirections = dirTensor;
        }

        var output = new T[numPoints * 4];
        for (int i = 0; i < numPoints; i++)
        {
            int outIdx = i * 4;
            int rgbIdx = i * 3;
            output[outIdx] = rgb.Data.Span[rgbIdx];
            output[outIdx + 1] = rgb.Data.Span[rgbIdx + 1];
            output[outIdx + 2] = rgb.Data.Span[rgbIdx + 2];
            output[outIdx + 3] = density.Data.Span[i];
        }

        return new Tensor<T>(output, [numPoints, 4]);
    }

    public override Tensor<T> Backpropagate(Tensor<T> outputGradient)
    {
        if (_lastQueryPositions == null || _lastQueryDirections == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward.");
        }

        ApplyQueryGradients(_lastQueryPositions, _lastQueryDirections, outputGradient);

        int numPoints = _lastQueryPositions.Shape[0];
        return new Tensor<T>(new T[numPoints * 6], [numPoints, 6]);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (LossFunction == null)
        {
            throw new InvalidOperationException("Loss function not set.");
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
        if (EnableDensification && _trainingStep % Math.Max(1, DensificationInterval) == 0)
        {
            DensifyAndPrune(gradients);
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        if (_gaussians.Count == 0)
        {
            if (parameters.Length == 0)
            {
                return;
            }

            throw new ArgumentException("No gaussians initialized to update.", nameof(parameters));
        }

        int colorDim = _useSphericalHarmonics ? 3 * GetShBasisCount() : 3;
        int perGaussian = 3 + 4 + 3 + 1 + colorDim;
        int expectedLength = perGaussian * _gaussians.Count;
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, got {parameters.Length}.", nameof(parameters));
        }

        int offset = 0;
        foreach (var gaussian in _gaussians)
        {
            for (int i = 0; i < 3; i++)
            {
                gaussian.Position[i] = parameters[offset++];
            }
            for (int i = 0; i < 4; i++)
            {
                gaussian.Rotation[i] = parameters[offset++];
            }
            for (int i = 0; i < 3; i++)
            {
                gaussian.Scale[i] = parameters[offset++];
            }
            gaussian.Opacity = parameters[offset++];
            for (int i = 0; i < colorDim; i++)
            {
                gaussian.Color[i] = parameters[offset++];
            }

            ComputeCovariance(gaussian);
        }

        MarkSpatialIndexDirty();
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        int colorDim = _useSphericalHarmonics ? 3 * GetShBasisCount() : 3;
        int perGaussian = 3 + 4 + 3 + 1 + colorDim;
        int totalParameters = _gaussians.Count * perGaussian;

        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GaussianCount", _gaussians.Count },
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
            ModelData = Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useSphericalHarmonics);
        writer.Write(_shDegree);
        writer.Write(_trainingStep);
        writer.Write(EnableDensification);
        writer.Write(DensificationInterval);
        writer.Write(PruneOpacityThreshold);
        writer.Write(SplitGradientThreshold);
        writer.Write(SplitPositionJitter);
        writer.Write(SplitScaleFactor);
        writer.Write(SplitOpacityFactor);
        writer.Write(SplitOpacityMax);
        writer.Write(MaxGaussians);
        writer.Write(PositionLearningRate);
        writer.Write(ColorLearningRate);
        writer.Write(OpacityLearningRate);
        writer.Write(ScaleLearningRate);
        writer.Write(RotationLearningRate);
        writer.Write(TileSize);
        writer.Write(EnableSpatialIndex);
        writer.Write(SpatialIndexRadius);
        writer.Write(InitialNeighborSearchScale);
        writer.Write(InitialScaleMultiplier);
        writer.Write(DefaultPointSpacing);
        writer.Write(MinScale);

        int colorDim = _useSphericalHarmonics ? 3 * GetShBasisCount() : 3;
        writer.Write(colorDim);
        writer.Write(_gaussians.Count);

        foreach (var gaussian in _gaussians)
        {
            for (int i = 0; i < 3; i++)
            {
                writer.Write(NumOps.ToDouble(gaussian.Position[i]));
            }
            for (int i = 0; i < 4; i++)
            {
                writer.Write(NumOps.ToDouble(gaussian.Rotation[i]));
            }
            for (int i = 0; i < 3; i++)
            {
                writer.Write(NumOps.ToDouble(gaussian.Scale[i]));
            }
            writer.Write(NumOps.ToDouble(gaussian.Opacity));
            for (int i = 0; i < colorDim; i++)
            {
                writer.Write(NumOps.ToDouble(gaussian.Color[i]));
            }
        }
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        bool useSh = reader.ReadBoolean();
        int shDegree = reader.ReadInt32();
        _trainingStep = reader.ReadInt32();
        EnableDensification = reader.ReadBoolean();
        DensificationInterval = reader.ReadInt32();
        PruneOpacityThreshold = reader.ReadDouble();
        SplitGradientThreshold = reader.ReadDouble();
        SplitPositionJitter = reader.ReadDouble();
        SplitScaleFactor = reader.ReadDouble();
        SplitOpacityFactor = reader.ReadDouble();
        SplitOpacityMax = reader.ReadDouble();
        MaxGaussians = reader.ReadInt32();
        PositionLearningRate = reader.ReadDouble();
        ColorLearningRate = reader.ReadDouble();
        OpacityLearningRate = reader.ReadDouble();
        ScaleLearningRate = reader.ReadDouble();
        RotationLearningRate = reader.ReadDouble();
        TileSize = reader.ReadInt32();
        EnableSpatialIndex = reader.ReadBoolean();
        SpatialIndexRadius = reader.ReadInt32();
        InitialNeighborSearchScale = reader.ReadDouble();
        InitialScaleMultiplier = reader.ReadDouble();
        DefaultPointSpacing = reader.ReadDouble();
        MinScale = reader.ReadDouble();

        if (useSh != _useSphericalHarmonics || shDegree != _shDegree)
        {
            throw new InvalidOperationException("Serialized GaussianSplatting configuration does not match this instance.");
        }

        int colorDim = reader.ReadInt32();
        int gaussianCount = reader.ReadInt32();
        _gaussians.Clear();

        for (int i = 0; i < gaussianCount; i++)
        {
            var gaussian = new Gaussian(colorDim, NumOps);

            for (int d = 0; d < 3; d++)
            {
                gaussian.Position[d] = NumOps.FromDouble(reader.ReadDouble());
            }
            for (int d = 0; d < 4; d++)
            {
                gaussian.Rotation[d] = NumOps.FromDouble(reader.ReadDouble());
            }
            for (int d = 0; d < 3; d++)
            {
                gaussian.Scale[d] = NumOps.FromDouble(reader.ReadDouble());
            }
            gaussian.Opacity = NumOps.FromDouble(reader.ReadDouble());
            for (int d = 0; d < colorDim; d++)
            {
                gaussian.Color[d] = NumOps.FromDouble(reader.ReadDouble());
            }

            ComputeCovariance(gaussian);
            _gaussians.Add(gaussian);
        }

        MarkSpatialIndexDirty();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new GaussianSplatting<T>(
            new GaussianSplattingOptions
            {
                UseSphericalHarmonics = _useSphericalHarmonics,
                ShDegree = _shDegree,
                EnableDensification = EnableDensification,
                DensificationInterval = DensificationInterval,
                PruneOpacityThreshold = PruneOpacityThreshold,
                SplitGradientThreshold = SplitGradientThreshold,
                SplitPositionJitter = SplitPositionJitter,
                SplitScaleFactor = SplitScaleFactor,
                SplitOpacityFactor = SplitOpacityFactor,
                SplitOpacityMax = SplitOpacityMax,
                MaxGaussians = MaxGaussians,
                PositionLearningRate = PositionLearningRate,
                ColorLearningRate = ColorLearningRate,
                OpacityLearningRate = OpacityLearningRate,
                ScaleLearningRate = ScaleLearningRate,
                RotationLearningRate = RotationLearningRate,
                TileSize = TileSize,
                EnableSpatialIndex = EnableSpatialIndex,
                SpatialIndexRadius = SpatialIndexRadius,
                InitialNeighborSearchScale = InitialNeighborSearchScale,
                InitialScaleMultiplier = InitialScaleMultiplier,
                DefaultPointSpacing = DefaultPointSpacing,
                MinScale = MinScale
            },
            initialPointCloud: null,
            initialColors: null,
            lossFunction: LossFunction);
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

        var data = input.Data.Span;
        cameraPosition = new Vector<T>(3)
        {
            [0] = data[0],
            [1] = data[1],
            [2] = data[2]
        };

        cameraRotation = new Matrix<T>(3, 3);
        int offset = 3;
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                cameraRotation[r, c] = data[offset++];
            }
        }

        focalLength = data[12];
    }

    private List<GaussianGradient> ApplyImageGradients(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength,
        Tensor<T> prediction,
        Tensor<T> lossGradient)
    {
        var transformed = TransformGaussiansToCamera(cameraPosition, cameraRotation);
        var projected = ProjectGaussiansTo2D(transformed, imageWidth, imageHeight, focalLength);
        var sorted = projected.OrderBy(g => g.Depth).ToList();

        int pixelCount = imageWidth * imageHeight;
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

        var totalR = new double[pixelCount];
        var totalG = new double[pixelCount];
        var totalB = new double[pixelCount];
        var gradR = new double[pixelCount];
        var gradG = new double[pixelCount];
        var gradB = new double[pixelCount];
        var predData = prediction.Data.Span;
        var gradData = lossGradient.Data.Span;

        for (int i = 0; i < pixelCount; i++)
        {
            int baseIdx = i * 3;
            totalR[i] = NumOps.ToDouble(predData[baseIdx]);
            totalG[i] = NumOps.ToDouble(predData[baseIdx + 1]);
            totalB[i] = NumOps.ToDouble(predData[baseIdx + 2]);
            gradR[i] = NumOps.ToDouble(gradData[baseIdx]);
            gradG[i] = NumOps.ToDouble(gradData[baseIdx + 1]);
            gradB[i] = NumOps.ToDouble(gradData[baseIdx + 2]);
        }

        var accumR = new double[pixelCount];
        var accumG = new double[pixelCount];
        var accumB = new double[pixelCount];
        var transmittance = new double[pixelCount];
        for (int i = 0; i < pixelCount; i++)
        {
            transmittance[i] = 1.0;
        }

        double r00 = NumOps.ToDouble(cameraRotation[0, 0]);
        double r01 = NumOps.ToDouble(cameraRotation[0, 1]);
        double r02 = NumOps.ToDouble(cameraRotation[0, 2]);
        double r10 = NumOps.ToDouble(cameraRotation[1, 0]);
        double r11 = NumOps.ToDouble(cameraRotation[1, 1]);
        double r12 = NumOps.ToDouble(cameraRotation[1, 2]);
        double r20 = NumOps.ToDouble(cameraRotation[2, 0]);
        double r21 = NumOps.ToDouble(cameraRotation[2, 1]);
        double r22 = NumOps.ToDouble(cameraRotation[2, 2]);

        var gradients = new List<GaussianGradient>(sorted.Count);
        int basisCount = GetShBasisCount();

        foreach (var gaussian in sorted)
        {
            var source = gaussian.Source;
            double gradPosX = 0.0;
            double gradPosY = 0.0;
            double gradPosZ = 0.0;
            double gradOpacity = 0.0;

            double gradColorR = 0.0;
            double gradColorG = 0.0;
            double gradColorB = 0.0;
            double gradInvA = 0.0;
            double gradInvB = 0.0;
            double gradInvC = 0.0;
            double[]? gradCoeffR = null;
            double[]? gradCoeffG = null;
            double[]? gradCoeffB = null;
            double[]? shBasis = null;

            if (_useSphericalHarmonics)
            {
                double dirX = -gaussian.CamX;
                double dirY = -gaussian.CamY;
                double dirZ = -gaussian.CamZ;
                NormalizeDirection(ref dirX, ref dirY, ref dirZ);
                shBasis = EvaluateSphericalHarmonicsBasis(dirX, dirY, dirZ);
                gradCoeffR = new double[basisCount];
                gradCoeffG = new double[basisCount];
                gradCoeffB = new double[basisCount];
            }

            double opacityParam = NumOps.ToDouble(source.Opacity);
            double alphaBase = Sigmoid(opacityParam);

            for (int y = gaussian.MinY; y <= gaussian.MaxY; y++)
            {
                int rowOffset = y * imageWidth;
                for (int x = gaussian.MinX; x <= gaussian.MaxX; x++)
                {
                    int idx = rowOffset + x;
                    double t = transmittance[idx];
                    if (t < 1e-4)
                    {
                        continue;
                    }

                    double dx = x - gaussian.MeanX;
                    double dy = y - gaussian.MeanY;
                    double exponent =
                        -0.5 * (gaussian.InvA * dx * dx + 2.0 * gaussian.InvB * dx * dy + gaussian.InvC * dy * dy);
                    if (exponent < -20.0)
                    {
                        continue;
                    }

                    double weight = Math.Exp(exponent);
                    double alpha = alphaBase * weight;
                    if (alpha < 1e-6)
                    {
                        continue;
                    }

                    double gR = gradR[idx];
                    double gG = gradG[idx];
                    double gB = gradB[idx];
                    double weightedAlpha = t * alpha;

                    if (_useSphericalHarmonics && shBasis != null && gradCoeffR != null && gradCoeffG != null && gradCoeffB != null)
                    {
                        for (int b = 0; b < basisCount; b++)
                        {
                            double basis = shBasis[b];
                            gradCoeffR[b] += weightedAlpha * gR * basis;
                            gradCoeffG[b] += weightedAlpha * gG * basis;
                            gradCoeffB[b] += weightedAlpha * gB * basis;
                        }
                    }
                    else
                    {
                        gradColorR += weightedAlpha * gR;
                        gradColorG += weightedAlpha * gG;
                        gradColorB += weightedAlpha * gB;
                    }

                    double contribR = weightedAlpha * gaussian.ColorR;
                    double contribG = weightedAlpha * gaussian.ColorG;
                    double contribB = weightedAlpha * gaussian.ColorB;

                    double afterR = totalR[idx] - accumR[idx] - contribR;
                    double afterG = totalG[idx] - accumG[idx] - contribG;
                    double afterB = totalB[idx] - accumB[idx] - contribB;

                    double invOneMinusAlpha = 1.0 / Math.Max(1e-6, 1.0 - alpha);
                    double dColorTermR = t * gaussian.ColorR - afterR * invOneMinusAlpha;
                    double dColorTermG = t * gaussian.ColorG - afterG * invOneMinusAlpha;
                    double dColorTermB = t * gaussian.ColorB - afterB * invOneMinusAlpha;
                    double dL_dalpha = gR * dColorTermR + gG * dColorTermG + gB * dColorTermB;
                    double dL_dweight = dL_dalpha * alphaBase;
                    double dL_dexponent = dL_dweight * weight;
                    gradInvA += -0.5 * dL_dexponent * dx * dx;
                    gradInvB += -dL_dexponent * dx * dy;
                    gradInvC += -0.5 * dL_dexponent * dy * dy;
                    gradOpacity += dL_dalpha * weight * alphaBase * (1.0 - alphaBase);

                    double invMuX = gaussian.InvA * dx + gaussian.InvB * dy;
                    double invMuY = gaussian.InvB * dx + gaussian.InvC * dy;
                    double gradMuX = dL_dalpha * alphaBase * weight * invMuX;
                    double gradMuY = dL_dalpha * alphaBase * weight * invMuY;

                    double invZ = 1.0 / gaussian.CamZ;
                    double f = gaussian.FocalLength;
                    double gradCamX = gradMuX * f * invZ;
                    double gradCamY = gradMuY * f * invZ;
                    double gradCamZ = gradMuX * (-f * gaussian.CamX * invZ * invZ)
                                      + gradMuY * (-f * gaussian.CamY * invZ * invZ);

                    gradPosX += r00 * gradCamX + r10 * gradCamY + r20 * gradCamZ;
                    gradPosY += r01 * gradCamX + r11 * gradCamY + r21 * gradCamZ;
                    gradPosZ += r02 * gradCamX + r12 * gradCamY + r22 * gradCamZ;

                    accumR[idx] += contribR;
                    accumG[idx] += contribG;
                    accumB[idx] += contribB;
                    transmittance[idx] = t * (1.0 - alpha);
                }
            }

            if (_useSphericalHarmonics && gradCoeffR != null && gradCoeffG != null && gradCoeffB != null)
            {
                for (int b = 0; b < basisCount; b++)
                {
                    int rIdx = b;
                    int gIdx = b + basisCount;
                    int bIdx = b + 2 * basisCount;

                    source.Color[rIdx] = NumOps.FromDouble(
                        NumOps.ToDouble(source.Color[rIdx]) - ColorLearningRate * gradCoeffR[b]);
                    source.Color[gIdx] = NumOps.FromDouble(
                        NumOps.ToDouble(source.Color[gIdx]) - ColorLearningRate * gradCoeffG[b]);
                    source.Color[bIdx] = NumOps.FromDouble(
                        NumOps.ToDouble(source.Color[bIdx]) - ColorLearningRate * gradCoeffB[b]);
                }
            }
            else
            {
                source.Color[0] = NumOps.FromDouble(Clamp01(NumOps.ToDouble(source.Color[0]) - ColorLearningRate * gradColorR));
                source.Color[1] = NumOps.FromDouble(Clamp01(NumOps.ToDouble(source.Color[1]) - ColorLearningRate * gradColorG));
                source.Color[2] = NumOps.FromDouble(Clamp01(NumOps.ToDouble(source.Color[2]) - ColorLearningRate * gradColorB));
            }

            source.Opacity = NumOps.FromDouble(NumOps.ToDouble(source.Opacity) - OpacityLearningRate * gradOpacity);
            source.Position[0] = NumOps.FromDouble(NumOps.ToDouble(source.Position[0]) - PositionLearningRate * gradPosX);
            source.Position[1] = NumOps.FromDouble(NumOps.ToDouble(source.Position[1]) - PositionLearningRate * gradPosY);
            source.Position[2] = NumOps.FromDouble(NumOps.ToDouble(source.Position[2]) - PositionLearningRate * gradPosZ);

            if (Math.Abs(gradInvA) > 0.0 || Math.Abs(gradInvB) > 0.0 || Math.Abs(gradInvC) > 0.0)
            {
                double invA = gaussian.InvA;
                double invB = gaussian.InvB;
                double invC = gaussian.InvC;

                double temp00 = gradInvA * invA + gradInvB * invB;
                double temp01 = gradInvA * invB + gradInvB * invC;
                double temp10 = gradInvB * invA + gradInvC * invB;
                double temp11 = gradInvB * invB + gradInvC * invC;

                double covGradA = -(invA * temp00 + invB * temp10);
                double covGradB = -(invA * temp01 + invB * temp11);
                double covGradC = -(invB * temp01 + invC * temp11);

                double invZ = 1.0 / gaussian.CamZ;
                double f = gaussian.FocalLength;
                double j00 = f * invZ;
                double j02 = -f * gaussian.CamX * invZ * invZ;
                double j11 = f * invZ;
                double j12 = -f * gaussian.CamY * invZ * invZ;

                double g00 = covGradA;
                double g01 = covGradB;
                double g11 = covGradC;

                double s00 = g00 * j00 * j00;
                double s01 = g01 * j00 * j11;
                double s02 = j00 * (g00 * j02 + g01 * j12);
                double s11 = g11 * j11 * j11;
                double s12 = j11 * (g01 * j02 + g11 * j12);
                double s22 = g00 * j02 * j02 + 2.0 * g01 * j02 * j12 + g11 * j12 * j12;

                double qw = NumOps.ToDouble(source.Rotation[0]);
                double qx = NumOps.ToDouble(source.Rotation[1]);
                double qy = NumOps.ToDouble(source.Rotation[2]);
                double qz = NumOps.ToDouble(source.Rotation[3]);

                double qNorm = Math.Sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
                if (qNorm < 1e-12)
                {
                    qw = 1.0;
                    qx = 0.0;
                    qy = 0.0;
                    qz = 0.0;
                }
                else
                {
                    double inv = 1.0 / qNorm;
                    qw *= inv;
                    qx *= inv;
                    qy *= inv;
                    qz *= inv;
                }

                double rot00 = 1.0 - 2.0 * (qy * qy + qz * qz);
                double rot01 = 2.0 * (qx * qy - qw * qz);
                double rot02 = 2.0 * (qx * qz + qw * qy);
                double rot10 = 2.0 * (qx * qy + qw * qz);
                double rot11 = 1.0 - 2.0 * (qx * qx + qz * qz);
                double rot12 = 2.0 * (qy * qz - qw * qx);
                double rot20 = 2.0 * (qx * qz - qw * qy);
                double rot21 = 2.0 * (qy * qz + qw * qx);
                double rot22 = 1.0 - 2.0 * (qx * qx + qy * qy);

                double gr00 = s00 * rot00 + s01 * rot10 + s02 * rot20;
                double gr01 = s00 * rot01 + s01 * rot11 + s02 * rot21;
                double gr02 = s00 * rot02 + s01 * rot12 + s02 * rot22;
                double gr10 = s01 * rot00 + s11 * rot10 + s12 * rot20;
                double gr11 = s01 * rot01 + s11 * rot11 + s12 * rot21;
                double gr12 = s01 * rot02 + s11 * rot12 + s12 * rot22;
                double gr20 = s02 * rot00 + s12 * rot10 + s22 * rot20;
                double gr21 = s02 * rot01 + s12 * rot11 + s22 * rot21;
                double gr22 = s02 * rot02 + s12 * rot12 + s22 * rot22;

                double rt00 = rot00 * gr00 + rot10 * gr10 + rot20 * gr20;
                double rt11 = rot01 * gr01 + rot11 * gr11 + rot21 * gr21;
                double rt22 = rot02 * gr02 + rot12 * gr12 + rot22 * gr22;

                double sx = Math.Max(MinScale, NumOps.ToDouble(source.Scale[0]));
                double sy = Math.Max(MinScale, NumOps.ToDouble(source.Scale[1]));
                double sz = Math.Max(MinScale, NumOps.ToDouble(source.Scale[2]));

                double gradScaleX = 2.0 * sx * rt00;
                double gradScaleY = 2.0 * sy * rt11;
                double gradScaleZ = 2.0 * sz * rt22;

                sx = Math.Max(MinScale, sx - ScaleLearningRate * gradScaleX);
                sy = Math.Max(MinScale, sy - ScaleLearningRate * gradScaleY);
                sz = Math.Max(MinScale, sz - ScaleLearningRate * gradScaleZ);

                source.Scale[0] = NumOps.FromDouble(sx);
                source.Scale[1] = NumOps.FromDouble(sy);
                source.Scale[2] = NumOps.FromDouble(sz);

                double sx2 = sx * sx;
                double sy2 = sy * sy;
                double sz2 = sz * sz;

                double rs00 = rot00 * sx2;
                double rs01 = rot01 * sy2;
                double rs02 = rot02 * sz2;
                double rs10 = rot10 * sx2;
                double rs11 = rot11 * sy2;
                double rs12 = rot12 * sz2;
                double rs20 = rot20 * sx2;
                double rs21 = rot21 * sy2;
                double rs22 = rot22 * sz2;

                double gR00 = 2.0 * (s00 * rs00 + s01 * rs10 + s02 * rs20);
                double gR01 = 2.0 * (s00 * rs01 + s01 * rs11 + s02 * rs21);
                double gR02 = 2.0 * (s00 * rs02 + s01 * rs12 + s02 * rs22);
                double gR10 = 2.0 * (s01 * rs00 + s11 * rs10 + s12 * rs20);
                double gR11 = 2.0 * (s01 * rs01 + s11 * rs11 + s12 * rs21);
                double gR12 = 2.0 * (s01 * rs02 + s11 * rs12 + s12 * rs22);
                double gR20 = 2.0 * (s02 * rs00 + s12 * rs10 + s22 * rs20);
                double gR21 = 2.0 * (s02 * rs01 + s12 * rs11 + s22 * rs21);
                double gR22 = 2.0 * (s02 * rs02 + s12 * rs12 + s22 * rs22);

                double gradW = gR01 * (-2.0 * qz) + gR02 * (2.0 * qy) + gR10 * (2.0 * qz)
                               + gR12 * (-2.0 * qx) + gR20 * (-2.0 * qy) + gR21 * (2.0 * qx);
                double gradX = gR01 * (2.0 * qy) + gR02 * (2.0 * qz) + gR10 * (2.0 * qy)
                               + gR11 * (-4.0 * qx) + gR12 * (-2.0 * qw) + gR20 * (2.0 * qz)
                               + gR21 * (2.0 * qw) + gR22 * (-4.0 * qx);
                double gradY = gR00 * (-4.0 * qy) + gR01 * (2.0 * qx) + gR02 * (2.0 * qw)
                               + gR10 * (2.0 * qx) + gR12 * (2.0 * qz) + gR20 * (-2.0 * qw)
                               + gR21 * (2.0 * qz) + gR22 * (-4.0 * qy);
                double gradZ = gR00 * (-4.0 * qz) + gR01 * (-2.0 * qw) + gR02 * (2.0 * qx)
                               + gR10 * (2.0 * qw) + gR11 * (-4.0 * qz) + gR12 * (2.0 * qy)
                               + gR20 * (2.0 * qx) + gR21 * (2.0 * qy);

                qw -= RotationLearningRate * gradW;
                qx -= RotationLearningRate * gradX;
                qy -= RotationLearningRate * gradY;
                qz -= RotationLearningRate * gradZ;

                double newNorm = Math.Sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
                if (newNorm < 1e-12)
                {
                    qw = 1.0;
                    qx = 0.0;
                    qy = 0.0;
                    qz = 0.0;
                }
                else
                {
                    double invNew = 1.0 / newNorm;
                    qw *= invNew;
                    qx *= invNew;
                    qy *= invNew;
                    qz *= invNew;
                }

                source.Rotation[0] = NumOps.FromDouble(qw);
                source.Rotation[1] = NumOps.FromDouble(qx);
                source.Rotation[2] = NumOps.FromDouble(qy);
                source.Rotation[3] = NumOps.FromDouble(qz);
                ComputeCovariance(source);
            }

            gradients.Add(new GaussianGradient(source, gradPosX, gradPosY, gradPosZ));
        }

        MarkSpatialIndexDirty();
        return gradients;
    }

    private void ApplyQueryGradients(
        Tensor<T> positions,
        Tensor<T> directions,
        Tensor<T> outputGradient)
    {
        if (outputGradient.Shape.Length != 2 || outputGradient.Shape[1] != 4)
        {
            throw new ArgumentException("Output gradient must have shape [N, 4].", nameof(outputGradient));
        }

        int numPoints = positions.Shape[0];
        var posData = positions.Data.Span;
        var dirData = directions.Data.Span;
        var gradData = outputGradient.Data.Span;

        for (int i = 0; i < numPoints; i++)
        {
            double px = NumOps.ToDouble(posData[i * 3]);
            double py = NumOps.ToDouble(posData[i * 3 + 1]);
            double pz = NumOps.ToDouble(posData[i * 3 + 2]);

            double dx = NumOps.ToDouble(dirData[i * 3]);
            double dy = NumOps.ToDouble(dirData[i * 3 + 1]);
            double dz = NumOps.ToDouble(dirData[i * 3 + 2]);
            NormalizeDirection(ref dx, ref dy, ref dz);

            double sumAlpha = 0.0;
            double sumR = 0.0;
            double sumG = 0.0;
            double sumB = 0.0;

            foreach (var gaussian in GetCandidateGaussians(px, py, pz))
            {
                if (gaussian.CovarianceInverse == null)
                {
                    ComputeCovariance(gaussian);
                }

                double gx = NumOps.ToDouble(gaussian.Position[0]);
                double gy = NumOps.ToDouble(gaussian.Position[1]);
                double gz = NumOps.ToDouble(gaussian.Position[2]);

                double ox = px - gx;
                double oy = py - gy;
                double oz = pz - gz;

                var inv = gaussian.CovarianceInverse;
                if (inv == null)
                {
                    throw new InvalidOperationException("Gaussian covariance inverse is not initialized.");
                }
                double inv00 = NumOps.ToDouble(inv[0, 0]);
                double inv01 = NumOps.ToDouble(inv[0, 1]);
                double inv02 = NumOps.ToDouble(inv[0, 2]);
                double inv11 = NumOps.ToDouble(inv[1, 1]);
                double inv12 = NumOps.ToDouble(inv[1, 2]);
                double inv22 = NumOps.ToDouble(inv[2, 2]);

                double q =
                    ox * (inv00 * ox + inv01 * oy + inv02 * oz) +
                    oy * (inv01 * ox + inv11 * oy + inv12 * oz) +
                    oz * (inv02 * ox + inv12 * oy + inv22 * oz);

                double weight = Math.Exp(-0.5 * q);
                double alphaBase = Sigmoid(NumOps.ToDouble(gaussian.Opacity));
                double alpha = alphaBase * weight;
                if (alpha <= 1e-8)
                {
                    continue;
                }

                var (r, g, b) = EvaluateGaussianColor(gaussian, dx, dy, dz);
                sumAlpha += alpha;
                sumR += alpha * r;
                sumG += alpha * g;
                sumB += alpha * b;
            }

            if (sumAlpha <= 1e-10)
            {
                continue;
            }

            int gradIdx = i * 4;
            double gradR = NumOps.ToDouble(gradData[gradIdx]);
            double gradG = NumOps.ToDouble(gradData[gradIdx + 1]);
            double gradB = NumOps.ToDouble(gradData[gradIdx + 2]);
            double gradDensity = NumOps.ToDouble(gradData[gradIdx + 3]);

            double invSumAlpha = 1.0 / sumAlpha;
            double dSumColorR = gradR * invSumAlpha;
            double dSumColorG = gradG * invSumAlpha;
            double dSumColorB = gradB * invSumAlpha;

            double colorDot = gradR * sumR + gradG * sumG + gradB * sumB;
            double dSumAlpha = gradDensity - colorDot * invSumAlpha * invSumAlpha;

            foreach (var gaussian in GetCandidateGaussians(px, py, pz))
            {
                if (gaussian.CovarianceInverse == null)
                {
                    ComputeCovariance(gaussian);
                }

                double gx = NumOps.ToDouble(gaussian.Position[0]);
                double gy = NumOps.ToDouble(gaussian.Position[1]);
                double gz = NumOps.ToDouble(gaussian.Position[2]);

                double ox = px - gx;
                double oy = py - gy;
                double oz = pz - gz;

                var inv = gaussian.CovarianceInverse;
                if (inv == null)
                {
                    throw new InvalidOperationException("Gaussian covariance inverse is not initialized.");
                }
                double inv00 = NumOps.ToDouble(inv[0, 0]);
                double inv01 = NumOps.ToDouble(inv[0, 1]);
                double inv02 = NumOps.ToDouble(inv[0, 2]);
                double inv11 = NumOps.ToDouble(inv[1, 1]);
                double inv12 = NumOps.ToDouble(inv[1, 2]);
                double inv22 = NumOps.ToDouble(inv[2, 2]);

                double q =
                    ox * (inv00 * ox + inv01 * oy + inv02 * oz) +
                    oy * (inv01 * ox + inv11 * oy + inv12 * oz) +
                    oz * (inv02 * ox + inv12 * oy + inv22 * oz);

                double weight = Math.Exp(-0.5 * q);
                double opacityParam = NumOps.ToDouble(gaussian.Opacity);
                double alphaBase = Sigmoid(opacityParam);
                double alpha = alphaBase * weight;
                if (alpha <= 1e-8)
                {
                    continue;
                }

                var (r, g, b) = EvaluateGaussianColor(gaussian, dx, dy, dz);
                double dAlpha = dSumAlpha + dSumColorR * r + dSumColorG * g + dSumColorB * b;

                if (_useSphericalHarmonics)
                {
                    var basis = EvaluateSphericalHarmonicsBasis(dx, dy, dz);
                    int basisCount = basis.Length;
                    for (int bIdx = 0; bIdx < basisCount; bIdx++)
                    {
                        int rIdx = bIdx;
                        int gIdx = bIdx + basisCount;
                        int bIdx2 = bIdx + 2 * basisCount;
                        double coeff = basis[bIdx];
                        gaussian.Color[rIdx] = NumOps.FromDouble(
                            NumOps.ToDouble(gaussian.Color[rIdx]) - ColorLearningRate * alpha * dSumColorR * coeff);
                        gaussian.Color[gIdx] = NumOps.FromDouble(
                            NumOps.ToDouble(gaussian.Color[gIdx]) - ColorLearningRate * alpha * dSumColorG * coeff);
                        gaussian.Color[bIdx2] = NumOps.FromDouble(
                            NumOps.ToDouble(gaussian.Color[bIdx2]) - ColorLearningRate * alpha * dSumColorB * coeff);
                    }
                }
                else
                {
                    gaussian.Color[0] = NumOps.FromDouble(Clamp01(NumOps.ToDouble(gaussian.Color[0]) - ColorLearningRate * alpha * dSumColorR));
                    gaussian.Color[1] = NumOps.FromDouble(Clamp01(NumOps.ToDouble(gaussian.Color[1]) - ColorLearningRate * alpha * dSumColorG));
                    gaussian.Color[2] = NumOps.FromDouble(Clamp01(NumOps.ToDouble(gaussian.Color[2]) - ColorLearningRate * alpha * dSumColorB));
                }

                double gradOpacity = dAlpha * weight * alphaBase * (1.0 - alphaBase);
                gaussian.Opacity = NumOps.FromDouble(opacityParam - OpacityLearningRate * gradOpacity);

                double gradPosX = dAlpha * alpha * (inv00 * ox + inv01 * oy + inv02 * oz);
                double gradPosY = dAlpha * alpha * (inv01 * ox + inv11 * oy + inv12 * oz);
                double gradPosZ = dAlpha * alpha * (inv02 * ox + inv12 * oy + inv22 * oz);

                gaussian.Position[0] = NumOps.FromDouble(NumOps.ToDouble(gaussian.Position[0]) - PositionLearningRate * gradPosX);
                gaussian.Position[1] = NumOps.FromDouble(NumOps.ToDouble(gaussian.Position[1]) - PositionLearningRate * gradPosY);
                gaussian.Position[2] = NumOps.FromDouble(NumOps.ToDouble(gaussian.Position[2]) - PositionLearningRate * gradPosZ);
            }
        }

        MarkSpatialIndexDirty();
    }

    private void DensifyAndPrune(List<GaussianGradient> gradients)
    {
        double pruneThreshold = Math.Max(0.0, PruneOpacityThreshold);
        for (int i = _gaussians.Count - 1; i >= 0; i--)
        {
            double alpha = Sigmoid(NumOps.ToDouble(_gaussians[i].Opacity));
            if (alpha < pruneThreshold)
            {
                _gaussians.RemoveAt(i);
            }
        }

        if (_gaussians.Count >= MaxGaussians)
        {
            return;
        }

        foreach (var gradient in gradients.OrderByDescending(g => g.Magnitude))
        {
            if (_gaussians.Count >= MaxGaussians)
            {
                break;
            }

            if (gradient.Magnitude < SplitGradientThreshold)
            {
                break;
            }

            var source = gradient.Gaussian;
            if (!_gaussians.Contains(source))
            {
                continue;
            }
            var clone = CloneGaussian(source);

            double gx = gradient.GradX;
            double gy = gradient.GradY;
            double gz = gradient.GradZ;
            NormalizeDirection(ref gx, ref gy, ref gz);

            double jitter = SplitPositionJitter;
            if (gx == 0.0 && gy == 0.0 && gz == 0.0)
            {
                gx = Random.NextDouble() - 0.5;
                gy = Random.NextDouble() - 0.5;
                gz = Random.NextDouble() - 0.5;
                NormalizeDirection(ref gx, ref gy, ref gz);
            }

            for (int d = 0; d < 3; d++)
            {
                double scale = NumOps.ToDouble(source.Scale[d]);
                double newScale = Math.Max(MinScale, scale * SplitScaleFactor);
                source.Scale[d] = NumOps.FromDouble(newScale);
                clone.Scale[d] = NumOps.FromDouble(newScale);
            }

            source.Position[0] = NumOps.FromDouble(NumOps.ToDouble(source.Position[0]) + jitter * gx);
            source.Position[1] = NumOps.FromDouble(NumOps.ToDouble(source.Position[1]) + jitter * gy);
            source.Position[2] = NumOps.FromDouble(NumOps.ToDouble(source.Position[2]) + jitter * gz);

            clone.Position[0] = NumOps.FromDouble(NumOps.ToDouble(clone.Position[0]) - jitter * gx);
            clone.Position[1] = NumOps.FromDouble(NumOps.ToDouble(clone.Position[1]) - jitter * gy);
            clone.Position[2] = NumOps.FromDouble(NumOps.ToDouble(clone.Position[2]) - jitter * gz);

            double alpha = Sigmoid(NumOps.ToDouble(source.Opacity));
            double newAlpha = Math.Min(SplitOpacityMax, alpha * SplitOpacityFactor);
            source.Opacity = NumOps.FromDouble(Logit(newAlpha));
            clone.Opacity = NumOps.FromDouble(Logit(newAlpha));

            ComputeCovariance(source);
            ComputeCovariance(clone);
            _gaussians.Add(clone);
        }

        MarkSpatialIndexDirty();
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        return ForwardWithMemory(input);
    }

    private Gaussian CloneGaussian(Gaussian source)
    {
        var clone = new Gaussian(source.Color.Length, NumOps);
        clone.Position[0] = source.Position[0];
        clone.Position[1] = source.Position[1];
        clone.Position[2] = source.Position[2];
        clone.Rotation[0] = source.Rotation[0];
        clone.Rotation[1] = source.Rotation[1];
        clone.Rotation[2] = source.Rotation[2];
        clone.Rotation[3] = source.Rotation[3];
        clone.Scale[0] = source.Scale[0];
        clone.Scale[1] = source.Scale[1];
        clone.Scale[2] = source.Scale[2];
        clone.Opacity = source.Opacity;
        clone.Covariance = source.Covariance?.Clone();
        clone.CovarianceInverse = source.CovarianceInverse?.Clone();

        for (int i = 0; i < source.Color.Length; i++)
        {
            clone.Color[i] = source.Color[i];
        }

        return clone;
    }

    private (double r, double g, double b) EvaluateGaussianColor(
        Gaussian gaussian,
        double dirX,
        double dirY,
        double dirZ)
    {
        if (!_useSphericalHarmonics)
        {
            return (
                Clamp01(NumOps.ToDouble(gaussian.Color[0])),
                Clamp01(NumOps.ToDouble(gaussian.Color[1])),
                Clamp01(NumOps.ToDouble(gaussian.Color[2])));
        }

        var basis = EvaluateSphericalHarmonicsBasis(dirX, dirY, dirZ);
        int basisCount = basis.Length;
        double r = 0.0;
        double g = 0.0;
        double b = 0.0;

        for (int i = 0; i < basisCount; i++)
        {
            double coeff = basis[i];
            r += coeff * NumOps.ToDouble(gaussian.Color[i]);
            g += coeff * NumOps.ToDouble(gaussian.Color[i + basisCount]);
            b += coeff * NumOps.ToDouble(gaussian.Color[i + 2 * basisCount]);
        }

        return (Clamp01(r), Clamp01(g), Clamp01(b));
    }

    private double[] EvaluateSphericalHarmonicsBasis(double x, double y, double z)
    {
        NormalizeDirection(ref x, ref y, ref z);
        int degree = _useSphericalHarmonics ? _shDegree : 0;
        int count = (degree + 1) * (degree + 1);
        var basis = new double[count];

        basis[0] = 0.282095;
        if (degree >= 1)
        {
            basis[1] = 0.488603 * y;
            basis[2] = 0.488603 * z;
            basis[3] = 0.488603 * x;
        }
        if (degree >= 2)
        {
            basis[4] = 1.092548 * x * y;
            basis[5] = 1.092548 * y * z;
            basis[6] = 0.315392 * (3.0 * z * z - 1.0);
            basis[7] = 1.092548 * x * z;
            basis[8] = 0.546274 * (x * x - y * y);
        }
        if (degree >= 3)
        {
            basis[9] = 0.590044 * y * (3.0 * x * x - y * y);
            basis[10] = 2.890611 * x * y * z;
            basis[11] = 0.457046 * y * (5.0 * z * z - 1.0);
            basis[12] = 0.373176 * z * (5.0 * z * z - 3.0);
            basis[13] = 0.457046 * x * (5.0 * z * z - 1.0);
            basis[14] = 1.445306 * z * (x * x - y * y);
            basis[15] = 0.590044 * x * (x * x - 3.0 * y * y);
        }

        return basis;
    }

    private int GetShBasisCount()
    {
        int degree = _useSphericalHarmonics ? _shDegree : 0;
        return (degree + 1) * (degree + 1);
    }

    private static void NormalizeDirection(ref double x, ref double y, ref double z)
    {
        double norm = Math.Sqrt(x * x + y * y + z * z);
        if (norm > 0.0)
        {
            double inv = 1.0 / norm;
            x *= inv;
            y *= inv;
            z *= inv;
        }
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    private static double Logit(double value)
    {
        double clamped = Math.Min(1.0 - 1e-6, Math.Max(1e-6, value));
        return Math.Log(clamped / (1.0 - clamped));
    }

    private static double Clamp01(double value)
    {
        if (value <= 0.0)
        {
            return 0.0;
        }
        if (value >= 1.0)
        {
            return 1.0;
        }

        return value;
    }

    /// <summary>
    /// Gets the number of Gaussians currently in the scene.
    /// </summary>
    public int GaussianCount => _gaussians.Count;
}
