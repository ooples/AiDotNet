using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralRadianceFields.Interfaces;

namespace AiDotNet.NeuralRadianceFields.Models;

/// <summary>
/// Implements Instant Neural Graphics Primitives (Instant-NGP) for fast NeRF training and rendering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Instant-NGP is a dramatically faster version of NeRF, making it practical for real-time use.
/// </para>
/// <para>
/// Speed comparison:
/// - Original NeRF: Hours to train, seconds to render each image
/// - Instant-NGP: Minutes to train, milliseconds to render each image
/// - Speedup: ~100× faster training, ~1000× faster rendering
/// </para>
/// <para>
/// Key innovations:
/// 1. Multiresolution hash encoding: Replace expensive positional encoding
/// 2. Tiny MLP: Much smaller network (2-4 layers vs 8 layers)
/// 3. CUDA optimization: Highly optimized GPU kernels
/// 4. Occupancy grids: Skip empty space efficiently
/// </para>
/// <para>
/// Multiresolution hash encoding explained:
/// - Traditional NeRF: Encode position with sin/cos functions (expensive)
/// - Instant-NGP: Look up features from a hash table (very fast)
///
/// How it works:
/// 1. Multiple levels of resolution (coarse to fine)
/// 2. At each level: Hash 3D position to table index
/// 3. Look up learned features at that index
/// 4. Interpolate features from nearby grid points
/// 5. Concatenate features from all levels
/// 6. Feed to small MLP for final color/density
///
/// Example with 3 levels:
/// - Level 0: Coarse grid (16³ cells) for large-scale features
/// - Level 1: Medium grid (64³ cells) for mid-scale features
/// - Level 2: Fine grid (256³ cells) for fine details
/// - Total table size: Much smaller than full 256³ grid
/// - Hash function maps 3D position to table index
/// </para>
/// <para>
/// Why hash tables are fast:
/// - No expensive trigonometric operations
/// - Direct memory lookup (O(1) time)
/// - Cache-friendly access patterns
/// - Parallelizes extremely well on GPU
///
/// Why hash collisions are okay:
/// - Multiple positions may hash to same index
/// - Network learns to handle collisions
/// - Collisions mostly affect similar regions
/// - Final quality is still excellent
/// </para>
/// <para>
/// Tiny MLP architecture:
/// - Original NeRF: 8 layers × 256 units = ~1M parameters
/// - Instant-NGP: 2 layers × 64 units = ~10K parameters
/// - Most representation power is in hash table, not MLP
/// - MLP just needs to combine hash features
/// </para>
/// <para>
/// Occupancy grids:
/// - Discretize space into voxel grid
/// - Mark which voxels contain geometry (occupancy)
/// - Skip sampling in empty voxels
/// - Huge speedup: Don't waste time on empty space
///
/// Example:
/// - Room scene: Most space is empty air
/// - Occupancy grid: Mark only voxels with walls/furniture
/// - Rendering: Skip ~90% of samples (the empty ones)
/// - Result: ~10× faster rendering
/// </para>
/// <para>
/// Training process:
/// 1. Initialize hash tables randomly
/// 2. Initialize occupancy grid (start assuming all occupied)
/// 3. For each training iteration:
///    - Sample rays from training images
///    - Use occupancy grid to skip empty space
///    - Query hash tables + tiny MLP
///    - Compute rendering loss
///    - Backprop to update hash tables and MLP
///    - Periodically update occupancy grid
/// 4. Converges in minutes instead of hours
/// </para>
/// <para>
/// Applications:
/// - Interactive 3D scanning: Scan object, view it seconds later
/// - Real-time novel view synthesis: Move camera and render instantly
/// - AR/VR: Low latency is critical for immersion
/// - Robotics: Build 3D maps in real-time
/// - Game development: Capture real objects for games
/// </para>
/// <para>
/// Limitations:
/// - Requires good GPU (CUDA implementation is fastest)
/// - Hash table size is a trade-off:
///   - Larger: Better quality, more memory
///   - Smaller: Faster, lower quality
/// - Still requires multiple images from different views
/// - Per-scene optimization (not a general model)
/// </para>
/// <para>
/// Comparison with NeRF:
///
/// Feature            | NeRF      | Instant-NGP
/// -------------------|-----------|-------------
/// Training time      | 1-2 days  | 5-10 minutes
/// Rendering speed    | 30s/image | 30ms/image
/// Model size         | ~5MB      | ~50MB (with hash tables)
/// Quality            | Excellent | Excellent
/// Memory usage       | Low       | Medium
/// GPU requirement    | Any       | CUDA (for best performance)
/// </para>
/// <para>
/// Reference: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
/// by Müller et al., ACM Transactions on Graphics 2022
/// </para>
/// </remarks>
public class InstantNGP<T> : NeuralNetworkBase<T>, IRadianceField<T>
{
    private readonly int _hashTableSize;
    private readonly int _numLevels;
    private readonly int _featuresPerLevel;
    private readonly int _finestResolution;
    private readonly int _coarsestResolution;
    private readonly int _mlpHiddenDim;
    private readonly int _mlpNumLayers;
    private readonly int _featureDim;
    private readonly int _colorHiddenDim;
    private readonly int _colorNumLayers;
    private readonly int _renderSamples;
    private readonly T _renderNearBound;
    private readonly T _renderFarBound;
    private readonly T _learningRate;
    private readonly double _occupancyDecay;
    private readonly double _occupancyThreshold;
    private readonly int _occupancyUpdateInterval;
    private readonly bool _useOccupancyGrid;
    private readonly int _occupancySamplesPerCell;
    private readonly double _occupancyJitter;
    private int _trainingStep;

    // Hash tables for multiresolution encoding
    private readonly Dictionary<int, Tensor<T>> _hashTables;

    private readonly List<DenseLayer<T>> _densityLayers = [];
    private DenseLayer<T>? _densityOutputLayer;
    private DenseLayer<T>? _featureLayer;
    private readonly List<DenseLayer<T>> _colorLayers = [];
    private DenseLayer<T>? _colorOutputLayer;

    // Occupancy grid for efficient sampling
    private Tensor<T>? _occupancyGrid;
    private uint[]? _occupancyBitfield;
    private readonly int _occupancyGridResolution;
    private readonly double[] _sceneMin = new double[3];
    private readonly double[] _sceneMax = new double[3];
    private readonly double[] _sceneSize = new double[3];
    private readonly double[] _sceneInvSize = new double[3];
    private Tensor<T>? _lastPositions;
    private Tensor<T>? _lastDirections;
    private Tensor<T>? _lastDensityRaw;
    private Tensor<T>? _lastRgbRaw;
    private Tensor<T>? _lastHashFeatureGradients;

    public override bool SupportsTraining => true;

    public InstantNGP()
        : this(new InstantNGPOptions<T>(), null)
    {
    }

    public InstantNGP(InstantNGPOptions<T> options, ILossFunction<T>? lossFunction = null)
        : base(CreateArchitecture(options.MlpHiddenDim), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression))
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (options.HashTableSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.HashTableSize), "Hash table size must be positive.");
        }
        if (options.NumLevels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.NumLevels), "Number of levels must be positive.");
        }
        if (options.FeaturesPerLevel <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.FeaturesPerLevel), "Features per level must be positive.");
        }
        if (options.FinestResolution <= 0 || options.CoarsestResolution <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.FinestResolution), "Resolutions must be positive.");
        }
        if (options.MlpHiddenDim <= 0 || options.MlpNumLayers <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.MlpHiddenDim), "MLP dimensions must be positive.");
        }
        if (options.FeatureDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.FeatureDim), "Feature dimension must be positive.");
        }
        if (options.ColorHiddenDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.ColorHiddenDim), "Color hidden dimension must be positive.");
        }
        if (options.ColorNumLayers < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.ColorNumLayers), "Color layer count cannot be negative.");
        }
        if (options.OccupancyGridResolution <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.OccupancyGridResolution), "Occupancy grid resolution must be positive.");
        }
        if (options.OccupancySamplesPerCell <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.OccupancySamplesPerCell), "Occupancy samples per cell must be positive.");
        }
        if (options.OccupancyJitter < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.OccupancyJitter), "Occupancy jitter cannot be negative.");
        }
        if (options.OccupancyDecay < 0.0 || options.OccupancyDecay > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.OccupancyDecay), "Occupancy decay must be in [0, 1].");
        }
        if (options.OccupancyThreshold < 0.0 || options.OccupancyThreshold > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.OccupancyThreshold), "Occupancy threshold must be in [0, 1].");
        }
        if (options.RenderSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.RenderSamples), "Render samples must be positive.");
        }

        _hashTableSize = options.HashTableSize;
        _numLevels = options.NumLevels;
        _featuresPerLevel = options.FeaturesPerLevel;
        _finestResolution = options.FinestResolution;
        _coarsestResolution = options.CoarsestResolution;
        _mlpHiddenDim = options.MlpHiddenDim;
        _mlpNumLayers = options.MlpNumLayers;
        _featureDim = options.FeatureDim;
        _colorHiddenDim = options.ColorHiddenDim;
        _colorNumLayers = options.ColorNumLayers;
        _occupancyGridResolution = options.OccupancyGridResolution;
        _learningRate = NumOps.FromDouble(options.LearningRate);
        _occupancyDecay = options.OccupancyDecay;
        _occupancyThreshold = options.OccupancyThreshold;
        _occupancyUpdateInterval = Math.Max(1, options.OccupancyUpdateInterval);
        _useOccupancyGrid = options.UseOccupancyGrid;
        _occupancySamplesPerCell = Math.Max(1, options.OccupancySamplesPerCell);
        _occupancyJitter = options.OccupancyJitter;
        _renderSamples = options.RenderSamples;
        _renderNearBound = NumOps.FromDouble(options.RenderNearBound);
        _renderFarBound = NumOps.FromDouble(options.RenderFarBound);

        _hashTables = new Dictionary<int, Tensor<T>>();

        InitializeHashTables();
        SetSceneBounds(options.SceneMin, options.SceneMax);
        InitializeOccupancyGrid();
        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the InstantNGP class.
    /// </summary>
    /// <param name="hashTableSize">Size of each hash table (typical: 2^19 = 524,288).</param>
    /// <param name="numLevels">Number of resolution levels (typical: 16).</param>
    /// <param name="featuresPerLevel">Number of features per hash table level (typical: 2).</param>
    /// <param name="finestResolution">Finest grid resolution (typical: 512-2048).</param>
    /// <param name="coarsestResolution">Coarsest grid resolution (typical: 16).</param>
    /// <param name="mlpHiddenDim">Hidden dimension of tiny MLP (typical: 64).</param>
    /// <param name="mlpNumLayers">Number of MLP layers (typical: 2).</param>
    /// <param name="occupancyGridResolution">Resolution of occupancy grid (typical: 128).</param>
    /// <param name="learningRate">Learning rate for hash table and MLP updates.</param>
    /// <param name="occupancyDecay">EMA decay for occupancy grid updates.</param>
    /// <param name="occupancyThreshold">Density threshold for occupancy.</param>
    /// <param name="occupancyUpdateInterval">Training steps between occupancy updates.</param>
    /// <param name="sceneMin">Optional scene bounds minimum (defaults to [0, 0, 0]).</param>
    /// <param name="sceneMax">Optional scene bounds maximum (defaults to [1, 1, 1]).</param>
    /// <param name="lossFunction">Optional loss function for training.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates an Instant-NGP model for fast 3D scene representation.
    ///
    /// Parameters explained:
    /// - hashTableSize: How many entries in each hash table
    ///   - Larger = better quality, more memory
    ///   - Typical: 2^19 = 524K entries
    ///   - Total memory: numLevels × hashTableSize × featuresPerLevel × 4 bytes
    ///   - Example: 16 × 524K × 2 × 4 = 64MB
    ///
    /// - numLevels: How many resolution scales
    ///   - More levels = capture more frequency details
    ///   - Typical: 16 levels
    ///   - Geometric spacing: each level is ~1.5× finer than previous
    ///
    /// - featuresPerLevel: Features stored per hash entry
    ///   - Typical: 2 (good balance of quality vs speed)
    ///   - Higher = more expressive but slower
    ///
    /// - finestResolution: Highest detail level
    ///   - Typical: 512-2048
    ///   - Higher = capture finer details
    ///   - Limited by hash table size (collisions increase)
    ///
    /// - coarsestResolution: Lowest detail level
    ///   - Typical: 16
    ///   - Captures overall structure
    ///
    /// - mlpHiddenDim: Size of tiny MLP hidden layers
    ///   - Typical: 64 (much smaller than NeRF's 256)
    ///   - Smaller is faster, sufficient because hash encoding does heavy lifting
    ///
    /// - mlpNumLayers: Depth of tiny MLP
    ///   - Typical: 2 (vs NeRF's 8)
    ///   - Simpler network is sufficient with good hash features
    ///
    /// - occupancyGridResolution: Voxel grid resolution for empty space skipping
    ///   - Typical: 128 (128³ = 2M voxels)
    ///   - Higher = more precise but more memory
    ///
    /// Standard Instant-NGP configuration for bounded scenes:
    /// new InstantNGP(
    ///     hashTableSize: 524288,     // 2^19
    ///     numLevels: 16,
    ///     featuresPerLevel: 2,
    ///     finestResolution: 2048,
    ///     coarsestResolution: 16,
    ///     mlpHiddenDim: 64,
    ///     mlpNumLayers: 2,
    ///     occupancyGridResolution: 128
    /// );
    ///
    /// Memory usage estimate:
    /// - Hash tables: ~64MB
    /// - Occupancy grid: ~2MB
    /// - MLP weights: ~50KB
    /// - Total: ~66MB (vs ~5MB for NeRF, but 100× faster)
    /// </remarks>
    public InstantNGP(
        int hashTableSize = 524288,
        int numLevels = 16,
        int featuresPerLevel = 2,
        int finestResolution = 2048,
        int coarsestResolution = 16,
        int mlpHiddenDim = 64,
        int mlpNumLayers = 2,
        int occupancyGridResolution = 128,
        double learningRate = 1e-2,
        double occupancyDecay = 0.95,
        double occupancyThreshold = 0.01,
        int occupancyUpdateInterval = 16,
        Vector<T>? sceneMin = null,
        Vector<T>? sceneMax = null,
        ILossFunction<T>? lossFunction = null)
        : this(
            new InstantNGPOptions<T>
            {
                HashTableSize = hashTableSize,
                NumLevels = numLevels,
                FeaturesPerLevel = featuresPerLevel,
                FinestResolution = finestResolution,
                CoarsestResolution = coarsestResolution,
                MlpHiddenDim = mlpHiddenDim,
                MlpNumLayers = mlpNumLayers,
                OccupancyGridResolution = occupancyGridResolution,
                LearningRate = learningRate,
                OccupancyDecay = occupancyDecay,
                OccupancyThreshold = occupancyThreshold,
                OccupancyUpdateInterval = occupancyUpdateInterval,
                SceneMin = sceneMin,
                SceneMax = sceneMax
            },
            lossFunction)
    {
    }

    private static NeuralNetworkArchitecture<T> CreateArchitecture(int hiddenDim)
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

    private void InitializeHashTables()
    {
        var random = Random;

        for (int level = 0; level < _numLevels; level++)
        {
            var table = new T[_hashTableSize * _featuresPerLevel];

            // Initialize with small random values
            for (int i = 0; i < table.Length; i++)
            {
                table[i] = NumOps.FromDouble(random.NextDouble() * 0.0001 - 0.00005);
            }

            _hashTables[level] = new Tensor<T>(table, [_hashTableSize, _featuresPerLevel]);
        }
    }

    private void InitializeOccupancyGrid()
    {
        if (!_useOccupancyGrid)
        {
            _occupancyGrid = null;
            _occupancyBitfield = null;
            return;
        }

        int gridSize = _occupancyGridResolution;
        var grid = new T[gridSize * gridSize * gridSize];

        // Initialize all voxels as potentially occupied
        // Will be refined during training
        for (int i = 0; i < grid.Length; i++)
        {
            grid[i] = NumOps.FromDouble(1.0);
        }

        _occupancyGrid = new Tensor<T>(grid, [gridSize, gridSize, gridSize]);

        int bitfieldLength = GetBitfieldLength(grid.Length);
        _occupancyBitfield = new uint[bitfieldLength];
        FillBitfield(_occupancyBitfield, grid.Length, true);
    }

    private void SetSceneBounds(Vector<T>? sceneMin, Vector<T>? sceneMax)
    {
        if (sceneMin == null || sceneMax == null)
        {
            _sceneMin[0] = 0.0;
            _sceneMin[1] = 0.0;
            _sceneMin[2] = 0.0;
            _sceneMax[0] = 1.0;
            _sceneMax[1] = 1.0;
            _sceneMax[2] = 1.0;
        }
        else
        {
            if (sceneMin.Length != 3 || sceneMax.Length != 3)
            {
                throw new ArgumentException("Scene bounds must be length 3.");
            }

            _sceneMin[0] = NumOps.ToDouble(sceneMin[0]);
            _sceneMin[1] = NumOps.ToDouble(sceneMin[1]);
            _sceneMin[2] = NumOps.ToDouble(sceneMin[2]);
            _sceneMax[0] = NumOps.ToDouble(sceneMax[0]);
            _sceneMax[1] = NumOps.ToDouble(sceneMax[1]);
            _sceneMax[2] = NumOps.ToDouble(sceneMax[2]);
        }

        for (int i = 0; i < 3; i++)
        {
            double size = _sceneMax[i] - _sceneMin[i];
            if (size <= 0.0)
            {
                throw new ArgumentException("Scene bounds must have positive size.");
            }

            _sceneSize[i] = size;
            _sceneInvSize[i] = 1.0 / size;
        }
    }

    private void NormalizePosition(
        double x,
        double y,
        double z,
        out double nx,
        out double ny,
        out double nz)
    {
        nx = Clamp01((x - _sceneMin[0]) * _sceneInvSize[0]);
        ny = Clamp01((y - _sceneMin[1]) * _sceneInvSize[1]);
        nz = Clamp01((z - _sceneMin[2]) * _sceneInvSize[2]);
    }

    private bool TryGetGridCell(
        double x,
        double y,
        double z,
        int gridSize,
        out int gx,
        out int gy,
        out int gz)
    {
        double nx = (x - _sceneMin[0]) * _sceneInvSize[0];
        double ny = (y - _sceneMin[1]) * _sceneInvSize[1];
        double nz = (z - _sceneMin[2]) * _sceneInvSize[2];

        if (nx < 0.0 || nx > 1.0 || ny < 0.0 || ny > 1.0 || nz < 0.0 || nz > 1.0)
        {
            gx = 0;
            gy = 0;
            gz = 0;
            return false;
        }

        gx = Math.Min((int)(nx * gridSize), gridSize - 1);
        gy = Math.Min((int)(ny * gridSize), gridSize - 1);
        gz = Math.Min((int)(nz * gridSize), gridSize - 1);
        return true;
    }

    private void RebuildOccupancyBitfield()
    {
        if (!_useOccupancyGrid || _occupancyGrid == null)
        {
            _occupancyBitfield = null;
            return;
        }

        var gridData = _occupancyGrid.Data;
        int cellCount = gridData.Length;
        int bitfieldLength = GetBitfieldLength(cellCount);
        if (_occupancyBitfield == null || _occupancyBitfield.Length != bitfieldLength)
        {
            _occupancyBitfield = new uint[bitfieldLength];
        }

        Array.Clear(_occupancyBitfield, 0, _occupancyBitfield.Length);
        for (int i = 0; i < cellCount; i++)
        {
            if (NumOps.ToDouble(gridData[i]) >= _occupancyThreshold)
            {
                SetBitfieldOccupied(_occupancyBitfield, i);
            }
        }

        ClearUnusedBits(_occupancyBitfield, cellCount);
    }

    private double AdvanceToNextVoxelBoundary(
        double ox,
        double oy,
        double oz,
        double dx,
        double dy,
        double dz,
        double t,
        int gx,
        int gy,
        int gz,
        int gridSize)
    {
        double cellSizeX = _sceneSize[0] / gridSize;
        double cellSizeY = _sceneSize[1] / gridSize;
        double cellSizeZ = _sceneSize[2] / gridSize;

        double cellMinX = _sceneMin[0] + gx * cellSizeX;
        double cellMinY = _sceneMin[1] + gy * cellSizeY;
        double cellMinZ = _sceneMin[2] + gz * cellSizeZ;

        double nextX = dx >= 0.0 ? cellMinX + cellSizeX : cellMinX;
        double nextY = dy >= 0.0 ? cellMinY + cellSizeY : cellMinY;
        double nextZ = dz >= 0.0 ? cellMinZ + cellSizeZ : cellMinZ;

        const double eps = 1e-9;
        double tX = Math.Abs(dx) < eps ? double.PositiveInfinity : (nextX - ox) / dx;
        double tY = Math.Abs(dy) < eps ? double.PositiveInfinity : (nextY - oy) / dy;
        double tZ = Math.Abs(dz) < eps ? double.PositiveInfinity : (nextZ - oz) / dz;

        double tNext = Math.Min(tX, Math.Min(tY, tZ));
        if (double.IsInfinity(tNext) || double.IsNaN(tNext))
        {
            return t + 1e-4;
        }

        if (tNext <= t + eps)
        {
            tNext = t + 1e-4;
        }

        return tNext + 1e-6;
    }

    protected override void InitializeLayers()
    {
        ClearLayers();
        _densityLayers.Clear();
        _colorLayers.Clear();

        int hashFeatureDim = _numLevels * _featuresPerLevel;
        int currentDim = hashFeatureDim;
        for (int i = 0; i < _mlpNumLayers; i++)
        {
            var hidden = new DenseLayer<T>(currentDim, _mlpHiddenDim, activationFunction: new ReLUActivation<T>());
            _densityLayers.Add(hidden);
            AddLayerToCollection(hidden);
            currentDim = _mlpHiddenDim;
        }

        _densityOutputLayer = new DenseLayer<T>(currentDim, 1, activationFunction: new IdentityActivation<T>());
        AddLayerToCollection(_densityOutputLayer);

        _featureLayer = new DenseLayer<T>(currentDim, _featureDim, activationFunction: new IdentityActivation<T>());
        AddLayerToCollection(_featureLayer);

        int colorInputDim = _featureDim + 3;
        int colorHiddenDim = _colorNumLayers > 0 ? _colorHiddenDim : colorInputDim;
        for (int i = 0; i < _colorNumLayers; i++)
        {
            int inputDim = i == 0 ? colorInputDim : colorHiddenDim;
            var hidden = new DenseLayer<T>(inputDim, colorHiddenDim, activationFunction: new ReLUActivation<T>());
            _colorLayers.Add(hidden);
            AddLayerToCollection(hidden);
        }

        _colorOutputLayer = new DenseLayer<T>(colorHiddenDim, 3, activationFunction: new IdentityActivation<T>());
        AddLayerToCollection(_colorOutputLayer);
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

        var hashFeatures = MultiresolutionHashEncoding(positions);
        var normalizedDirections = NormalizeDirections(viewingDirections);

        if (_densityOutputLayer == null || _featureLayer == null || _colorOutputLayer == null)
        {
            throw new InvalidOperationException("InstantNGP layers are not initialized.");
        }

        Tensor<T> densityHidden = hashFeatures;
        for (int i = 0; i < _densityLayers.Count; i++)
        {
            densityHidden = _densityLayers[i].Forward(densityHidden);
        }

        var densityRaw = _densityOutputLayer.Forward(densityHidden);
        var feature = _featureLayer.Forward(densityHidden);
        var density = ApplySoftplus(densityRaw);

        var colorInput = ConcatenateFeatures(feature, normalizedDirections);
        Tensor<T> colorHidden = colorInput;
        for (int i = 0; i < _colorLayers.Count; i++)
        {
            colorHidden = _colorLayers[i].Forward(colorHidden);
        }

        var rgbRaw = _colorOutputLayer.Forward(colorHidden);
        var rgb = Engine.Sigmoid(rgbRaw);

        if (IsTrainingMode)
        {
            _lastPositions = positions;
            _lastDirections = viewingDirections;
            _lastDensityRaw = densityRaw;
            _lastRgbRaw = rgbRaw;
        }

        return (rgb, density);
    }

    private Tensor<T> MultiresolutionHashEncoding(Tensor<T> positions)
    {
        int numPoints = positions.Shape[0];
        int totalFeatures = _numLevels * _featuresPerLevel;
        var features = new T[numPoints * totalFeatures];
        var posData = positions.Data;

        for (int level = 0; level < _numLevels; level++)
        {
            int resolution = GetLevelResolution(level);
            var table = _hashTables[level].Data;

            // For each point, hash and lookup features
            for (int i = 0; i < numPoints; i++)
            {
                double px = NumOps.ToDouble(posData[i * 3]);
                double py = NumOps.ToDouble(posData[i * 3 + 1]);
                double pz = NumOps.ToDouble(posData[i * 3 + 2]);
                NormalizePosition(px, py, pz, out double x, out double y, out double z);

                double gx = x * resolution;
                double gy = y * resolution;
                double gz = z * resolution;

                int x0 = (int)Math.Floor(gx);
                int y0 = (int)Math.Floor(gy);
                int z0 = (int)Math.Floor(gz);

                double fx = gx - x0;
                double fy = gy - y0;
                double fz = gz - z0;

                int x1 = x0 + 1;
                int y1 = y0 + 1;
                int z1 = z0 + 1;

                var w000 = (1 - fx) * (1 - fy) * (1 - fz);
                var w001 = (1 - fx) * (1 - fy) * fz;
                var w010 = (1 - fx) * fy * (1 - fz);
                var w011 = (1 - fx) * fy * fz;
                var w100 = fx * (1 - fy) * (1 - fz);
                var w101 = fx * (1 - fy) * fz;
                var w110 = fx * fy * (1 - fz);
                var w111 = fx * fy * fz;

                int h000 = HashIndex(x0, y0, z0);
                int h001 = HashIndex(x0, y0, z1);
                int h010 = HashIndex(x0, y1, z0);
                int h011 = HashIndex(x0, y1, z1);
                int h100 = HashIndex(x1, y0, z0);
                int h101 = HashIndex(x1, y0, z1);
                int h110 = HashIndex(x1, y1, z0);
                int h111 = HashIndex(x1, y1, z1);

                int featureBase = i * totalFeatures + level * _featuresPerLevel;

                for (int f = 0; f < _featuresPerLevel; f++)
                {
                    double value =
                        w000 * NumOps.ToDouble(table[h000 * _featuresPerLevel + f]) +
                        w001 * NumOps.ToDouble(table[h001 * _featuresPerLevel + f]) +
                        w010 * NumOps.ToDouble(table[h010 * _featuresPerLevel + f]) +
                        w011 * NumOps.ToDouble(table[h011 * _featuresPerLevel + f]) +
                        w100 * NumOps.ToDouble(table[h100 * _featuresPerLevel + f]) +
                        w101 * NumOps.ToDouble(table[h101 * _featuresPerLevel + f]) +
                        w110 * NumOps.ToDouble(table[h110 * _featuresPerLevel + f]) +
                        w111 * NumOps.ToDouble(table[h111 * _featuresPerLevel + f]);

                    features[featureBase + f] = NumOps.FromDouble(value);
                }
            }
        }

        return new Tensor<T>(features, [numPoints, totalFeatures]);
    }

    private static int SpatialHash(int x, int y, int z)
    {
        unchecked
        {
            const uint p1 = 73856093;
            const uint p2 = 19349663;
            const uint p3 = 83492791;

            uint hx = (uint)x * p1;
            uint hy = (uint)y * p2;
            uint hz = (uint)z * p3;

            return (int)(hx ^ hy ^ hz);
        }
    }

    private int HashIndex(int x, int y, int z)
    {
        unchecked
        {
            uint hash = (uint)SpatialHash(x, y, z);
            return (int)(hash % (uint)_hashTableSize);
        }
    }

    public Tensor<T> RenderImage(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength)
    {
        var (rayOrigins, rayDirections) = GenerateCameraRays(
            cameraPosition, cameraRotation, imageWidth, imageHeight, focalLength);
        double centerX = 0.5 * (_sceneMin[0] + _sceneMax[0]);
        double centerY = 0.5 * (_sceneMin[1] + _sceneMax[1]);
        double centerZ = 0.5 * (_sceneMin[2] + _sceneMax[2]);
        double dx = NumOps.ToDouble(cameraPosition[0]) - centerX;
        double dy = NumOps.ToDouble(cameraPosition[1]) - centerY;
        double dz = NumOps.ToDouble(cameraPosition[2]) - centerZ;
        double radius = 0.5 * Math.Sqrt(
            _sceneSize[0] * _sceneSize[0] +
            _sceneSize[1] * _sceneSize[1] +
            _sceneSize[2] * _sceneSize[2]);
        double dist = Math.Sqrt(dx * dx + dy * dy + dz * dz);
        double configuredFar = NumOps.ToDouble(_renderFarBound);
        double farBoundValue = Math.Max(Math.Max(1e-3, configuredFar), dist + radius);
        double nearBoundValue = Math.Max(0.0, NumOps.ToDouble(_renderNearBound));

        var rendered = RenderRays(
            rayOrigins,
            rayDirections,
            numSamples: _renderSamples,
            nearBound: NumOps.FromDouble(nearBoundValue),
            farBound: NumOps.FromDouble(farBoundValue));

        return rendered.Reshape(imageHeight, imageWidth, 3);
    }

    public Tensor<T> RenderRays(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        int numSamples,
        T nearBound,
        T farBound)
    {
        int numRays = rayOrigins.Shape[0];
        var (samplePositions, sampleDirections, occupancyMask, sampleTs, rayNear, rayFar) =
            SampleRaysWithOccupancy(rayOrigins, rayDirections, numSamples, nearBound, farBound);

        int totalSamples = numRays * numSamples;
        var rgbAll = new T[totalSamples * 3];
        var densityAll = new T[totalSamples];

        var occupiedIndices = new List<int>();
        for (int i = 0; i < totalSamples; i++)
        {
            if (occupancyMask[i])
            {
                occupiedIndices.Add(i);
            }
        }

        if (occupiedIndices.Count > 0)
        {
            var occPositions = new T[occupiedIndices.Count * 3];
            var occDirections = new T[occupiedIndices.Count * 3];
            var posData = samplePositions.Data;
            var dirData = sampleDirections.Data;

            for (int i = 0; i < occupiedIndices.Count; i++)
            {
                int src = occupiedIndices[i] * 3;
                int dst = i * 3;
                occPositions[dst] = posData[src];
                occPositions[dst + 1] = posData[src + 1];
                occPositions[dst + 2] = posData[src + 2];

                occDirections[dst] = dirData[src];
                occDirections[dst + 1] = dirData[src + 1];
                occDirections[dst + 2] = dirData[src + 2];
            }

            var occPosTensor = new Tensor<T>(occPositions, [occupiedIndices.Count, 3]);
            var occDirTensor = new Tensor<T>(occDirections, [occupiedIndices.Count, 3]);
            var (rgbOcc, densityOcc) = QueryField(occPosTensor, occDirTensor);

            var rgbOccData = rgbOcc.Data;
            var densityOccData = densityOcc.Data;

            for (int i = 0; i < occupiedIndices.Count; i++)
            {
                int dstSample = occupiedIndices[i];
                int dstRgb = dstSample * 3;
                int srcRgb = i * 3;
                rgbAll[dstRgb] = rgbOccData[srcRgb];
                rgbAll[dstRgb + 1] = rgbOccData[srcRgb + 1];
                rgbAll[dstRgb + 2] = rgbOccData[srcRgb + 2];
                densityAll[dstSample] = densityOccData[i];
            }
        }

        var rgbTensor = new Tensor<T>(rgbAll, [totalSamples, 3]);
        var densityTensor = new Tensor<T>(densityAll, [totalSamples, 1]);

        return VolumeRendering(rgbTensor, densityTensor, numRays, numSamples, rayNear, rayFar, sampleTs);
    }

    private (Tensor<T> positions, Tensor<T> directions, bool[] mask, double[] sampleTs, double[] rayNear, double[] rayFar)
        SampleRaysWithOccupancy(
            Tensor<T> rayOrigins,
            Tensor<T> rayDirections,
            int numSamples,
            T nearBound,
            T farBound)
    {
        int numRays = rayOrigins.Shape[0];
        int totalSamples = numRays * numSamples;
        var positions = new T[totalSamples * 3];
        var directions = new T[totalSamples * 3];
        var mask = new bool[totalSamples];
        var sampleTs = new double[totalSamples];
        var rayNear = new double[numRays];
        var rayFar = new double[numRays];

        var originData = rayOrigins.Data;
        var dirData = rayDirections.Data;
        double near = NumOps.ToDouble(nearBound);
        double far = NumOps.ToDouble(farBound);
        int gridSize = _occupancyGridResolution;
        int totalCells = gridSize * gridSize * gridSize;
        uint[]? bitfield = _occupancyBitfield;
        bool useOccupancy = _useOccupancyGrid &&
            bitfield != null &&
            bitfield.Length == GetBitfieldLength(totalCells);

        for (int r = 0; r < numRays; r++)
        {
            double ox = NumOps.ToDouble(originData[r * 3]);
            double oy = NumOps.ToDouble(originData[r * 3 + 1]);
            double oz = NumOps.ToDouble(originData[r * 3 + 2]);
            double dx = NumOps.ToDouble(dirData[r * 3]);
            double dy = NumOps.ToDouble(dirData[r * 3 + 1]);
            double dz = NumOps.ToDouble(dirData[r * 3 + 2]);

            if (!ComputeRayBounds(ox, oy, oz, dx, dy, dz, near, far, out double tMin, out double tMax))
            {
                rayNear[r] = 0.0;
                rayFar[r] = 0.0;
                continue;
            }

            rayNear[r] = tMin;
            rayFar[r] = tMax;
            double range = tMax - tMin;
            double step = (numSamples > 1 && range > 0.0) ? range / (numSamples - 1) : 0.0;

            if (!useOccupancy)
            {
                for (int s = 0; s < numSamples; s++)
                {
                    double t = tMin + step * s;
                    int sampleIdx = r * numSamples + s;
                    int baseIdx = sampleIdx * 3;

                    double px = ox + t * dx;
                    double py = oy + t * dy;
                    double pz = oz + t * dz;

                    positions[baseIdx] = NumOps.FromDouble(px);
                    positions[baseIdx + 1] = NumOps.FromDouble(py);
                    positions[baseIdx + 2] = NumOps.FromDouble(pz);

                    directions[baseIdx] = dirData[r * 3];
                    directions[baseIdx + 1] = dirData[r * 3 + 1];
                    directions[baseIdx + 2] = dirData[r * 3 + 2];
                    mask[sampleIdx] = true;
                    sampleTs[sampleIdx] = t;
                }
                continue;
            }

            int baseSampleIdx = r * numSamples;
            double tCurrent = tMin;
            int samplesTaken = 0;

            while (samplesTaken < numSamples && tCurrent <= tMax)
            {
                double px = ox + tCurrent * dx;
                double py = oy + tCurrent * dy;
                double pz = oz + tCurrent * dz;

                if (!TryGetGridCell(px, py, pz, gridSize, out int gx, out int gy, out int gz))
                {
                    if (step <= 0.0)
                    {
                        break;
                    }

                    tCurrent += step;
                    continue;
                }

                int gridIdx = (gx * gridSize + gy) * gridSize + gz;
                if (bitfield != null && IsBitfieldOccupied(bitfield, gridIdx))
                {
                    int sampleIdx = baseSampleIdx + samplesTaken;
                    int baseIdx = sampleIdx * 3;
                    positions[baseIdx] = NumOps.FromDouble(px);
                    positions[baseIdx + 1] = NumOps.FromDouble(py);
                    positions[baseIdx + 2] = NumOps.FromDouble(pz);
                    directions[baseIdx] = dirData[r * 3];
                    directions[baseIdx + 1] = dirData[r * 3 + 1];
                    directions[baseIdx + 2] = dirData[r * 3 + 2];
                    mask[sampleIdx] = true;
                    sampleTs[sampleIdx] = tCurrent;
                    samplesTaken++;

                    if (step <= 0.0)
                    {
                        break;
                    }

                    tCurrent += step;
                }
                else
                {
                    if (step <= 0.0)
                    {
                        break;
                    }

                    tCurrent = AdvanceToNextVoxelBoundary(
                        ox, oy, oz, dx, dy, dz, tCurrent, gx, gy, gz, gridSize);
                }
            }
        }

        return (new Tensor<T>(positions, [totalSamples, 3]),
            new Tensor<T>(directions, [totalSamples, 3]),
            mask,
            sampleTs,
            rayNear,
            rayFar);
    }

    private void ApplyHashTableGradients(Tensor<T> hashFeatureGradients)
    {
        if (_lastPositions == null)
        {
            throw new InvalidOperationException("Hash table gradients require cached positions from a forward pass.");
        }

        if (hashFeatureGradients.Shape.Length != 2)
        {
            throw new ArgumentException("Hash feature gradients must have shape [N, F].", nameof(hashFeatureGradients));
        }

        int numPoints = _lastPositions.Shape[0];
        int hashFeatureDim = _numLevels * _featuresPerLevel;
        if (hashFeatureGradients.Shape[0] != numPoints || hashFeatureGradients.Shape[1] != hashFeatureDim)
        {
            throw new ArgumentException("Hash feature gradients shape does not match cached positions.", nameof(hashFeatureGradients));
        }

        var posData = _lastPositions.Data;
        var gradData = hashFeatureGradients.Data;
        double learningRate = NumOps.ToDouble(_learningRate);

        for (int level = 0; level < _numLevels; level++)
        {
            int resolution = GetLevelResolution(level);
            var table = _hashTables[level].Data;

            for (int i = 0; i < numPoints; i++)
            {
                double px = NumOps.ToDouble(posData[i * 3]);
                double py = NumOps.ToDouble(posData[i * 3 + 1]);
                double pz = NumOps.ToDouble(posData[i * 3 + 2]);
                NormalizePosition(px, py, pz, out double x, out double y, out double z);

                double gx = x * resolution;
                double gy = y * resolution;
                double gz = z * resolution;

                int x0 = (int)Math.Floor(gx);
                int y0 = (int)Math.Floor(gy);
                int z0 = (int)Math.Floor(gz);

                double fx = gx - x0;
                double fy = gy - y0;
                double fz = gz - z0;

                int x1 = x0 + 1;
                int y1 = y0 + 1;
                int z1 = z0 + 1;

                double w000 = (1 - fx) * (1 - fy) * (1 - fz);
                double w001 = (1 - fx) * (1 - fy) * fz;
                double w010 = (1 - fx) * fy * (1 - fz);
                double w011 = (1 - fx) * fy * fz;
                double w100 = fx * (1 - fy) * (1 - fz);
                double w101 = fx * (1 - fy) * fz;
                double w110 = fx * fy * (1 - fz);
                double w111 = fx * fy * fz;

                int h000 = HashIndex(x0, y0, z0);
                int h001 = HashIndex(x0, y0, z1);
                int h010 = HashIndex(x0, y1, z0);
                int h011 = HashIndex(x0, y1, z1);
                int h100 = HashIndex(x1, y0, z0);
                int h101 = HashIndex(x1, y0, z1);
                int h110 = HashIndex(x1, y1, z0);
                int h111 = HashIndex(x1, y1, z1);

                int gradBase = i * hashFeatureDim + level * _featuresPerLevel;
                for (int f = 0; f < _featuresPerLevel; f++)
                {
                    double grad = NumOps.ToDouble(gradData[gradBase + f]);
                    if (grad == 0.0)
                    {
                        continue;
                    }

                    UpdateHashEntry(table, h000, f, grad, w000, learningRate);
                    UpdateHashEntry(table, h001, f, grad, w001, learningRate);
                    UpdateHashEntry(table, h010, f, grad, w010, learningRate);
                    UpdateHashEntry(table, h011, f, grad, w011, learningRate);
                    UpdateHashEntry(table, h100, f, grad, w100, learningRate);
                    UpdateHashEntry(table, h101, f, grad, w101, learningRate);
                    UpdateHashEntry(table, h110, f, grad, w110, learningRate);
                    UpdateHashEntry(table, h111, f, grad, w111, learningRate);
                }
            }
        }
    }

    private void UpdateHashEntry(
        Vector<T> table,
        int index,
        int feature,
        double grad,
        double weight,
        double learningRate)
    {
        int tableIndex = index * _featuresPerLevel + feature;
        double current = NumOps.ToDouble(table[tableIndex]);
        double update = learningRate * grad * weight;
        table[tableIndex] = NumOps.FromDouble(current - update);
    }

    private void UpdateOccupancyGrid()
    {
        if (!_useOccupancyGrid || _occupancyGrid == null)
        {
            return;
        }

        int gridSize = _occupancyGridResolution;
        int totalCells = gridSize * gridSize * gridSize;
        if (totalCells <= 0)
        {
            return;
        }

        var gridData = _occupancyGrid.Data;
        double decay = _occupancyDecay;
        if (decay < 0.0)
        {
            decay = 0.0;
        }
        else if (decay > 1.0)
        {
            decay = 1.0;
        }

        int batchSize = Math.Min(totalCells, Math.Max(4096, totalCells / 16));  
        var unitDirection = new[] { NumOps.Zero, NumOps.Zero, NumOps.One };     
        int samplesPerCell = Math.Max(1, _occupancySamplesPerCell);
        double jitter = _occupancyJitter;
        if (jitter < 0.0)
        {
            jitter = 0.0;
        }
        else if (jitter > 1.0)
        {
            jitter = 1.0;
        }

        var random = Random;
        double cellSize = Math.Max(_sceneSize[0], Math.Max(_sceneSize[1], _sceneSize[2])) / gridSize;
        if (cellSize <= 0.0)
        {
            cellSize = 1.0 / gridSize;
        }

        bool originalMode = IsTrainingMode;
        SetTrainingMode(false);

        for (int start = 0; start < totalCells; start += batchSize)
        {
            int count = Math.Min(batchSize, totalCells - start);
            int sampleCount = count * samplesPerCell;
            var positions = new T[sampleCount * 3];
            var directions = new T[sampleCount * 3];

            for (int i = 0; i < count; i++)
            {
                int cellIndex = start + i;
                int gx = cellIndex / (gridSize * gridSize);
                int rem = cellIndex - gx * gridSize * gridSize;
                int gy = rem / gridSize;
                int gz = rem - gy * gridSize;

                for (int s = 0; s < samplesPerCell; s++)
                {
                    double jx = jitter > 0.0 ? (random.NextDouble() - 0.5) * jitter : 0.0;
                    double jy = jitter > 0.0 ? (random.NextDouble() - 0.5) * jitter : 0.0;
                    double jz = jitter > 0.0 ? (random.NextDouble() - 0.5) * jitter : 0.0;

                    double sampleX = Math.Min(Math.Max(gx + 0.5 + jx, 0.0), gridSize - 1e-6);
                    double sampleY = Math.Min(Math.Max(gy + 0.5 + jy, 0.0), gridSize - 1e-6);
                    double sampleZ = Math.Min(Math.Max(gz + 0.5 + jz, 0.0), gridSize - 1e-6);

                    double nx = sampleX / gridSize;
                    double ny = sampleY / gridSize;
                    double nz = sampleZ / gridSize;

                    double px = _sceneMin[0] + nx * _sceneSize[0];
                    double py = _sceneMin[1] + ny * _sceneSize[1];
                    double pz = _sceneMin[2] + nz * _sceneSize[2];

                    int sampleIndex = i * samplesPerCell + s;
                    int baseIdx = sampleIndex * 3;
                    positions[baseIdx] = NumOps.FromDouble(px);
                    positions[baseIdx + 1] = NumOps.FromDouble(py);
                    positions[baseIdx + 2] = NumOps.FromDouble(pz);

                    directions[baseIdx] = unitDirection[0];
                    directions[baseIdx + 1] = unitDirection[1];
                    directions[baseIdx + 2] = unitDirection[2];
                }
            }

            var posTensor = new Tensor<T>(positions, [sampleCount, 3]);
            var dirTensor = new Tensor<T>(directions, [sampleCount, 3]);
            var (_, density) = QueryField(posTensor, dirTensor);
            var densityData = density.Data;

            for (int i = 0; i < count; i++)
            {
                int gridIndex = start + i;
                double previous = NumOps.ToDouble(gridData[gridIndex]);
                double maxAlpha = 0.0;
                int baseSample = i * samplesPerCell;
                for (int s = 0; s < samplesPerCell; s++)
                {
                    double sigma = NumOps.ToDouble(densityData[baseSample + s]);
                    if (sigma < 0.0)
                    {
                        continue;
                    }

                    double alpha = 1.0 - Math.Exp(-sigma * cellSize);
                    if (alpha > maxAlpha)
                    {
                        maxAlpha = alpha;
                    }
                }

                double updated = Math.Max(previous * decay, maxAlpha);
                gridData[gridIndex] = NumOps.FromDouble(updated);
            }
        }

        SetTrainingMode(originalMode);
        RebuildOccupancyBitfield();
    }

    private Tensor<T> NormalizeDirections(Tensor<T> directions)
    {
        var norm = Engine.TensorNorm(directions, axis: 1, keepDims: true);
        norm = Engine.TensorAddScalar(norm, NumOps.FromDouble(1e-8));
        var normBroadcast = Engine.TensorTile(norm, new[] { 1, directions.Shape[1] });
        return Engine.TensorDivide(directions, normBroadcast);
    }

    private Tensor<T> ConcatenateFeatures(Tensor<T> hashFeatures, Tensor<T> directions)
    {
        return Engine.TensorConcatenate(new[] { hashFeatures, directions }, axis: 1);
    }

    private Tensor<T> ApplySoftplus(Tensor<T> input)
    {
        var exp = Engine.TensorExp(input);
        var expPlus = Engine.TensorAddScalar(exp, NumOps.One);
        return Engine.TensorLog(expPlus);
    }

    private Tensor<T> ApplySigmoidGradient(Tensor<T> raw, Tensor<T> gradient)
    {
        var numOps = NumOps;
        var data = new T[gradient.Length];
        for (int i = 0; i < gradient.Length; i++)
        {
            double rawVal = numOps.ToDouble(raw.Data[i]);
            double sig = 1.0 / (1.0 + Math.Exp(-rawVal));
            double grad = numOps.ToDouble(gradient.Data[i]);
            data[i] = numOps.FromDouble(grad * sig * (1.0 - sig));
        }

        return new Tensor<T>(data, gradient.Shape);
    }

    private Tensor<T> ApplySoftplusGradient(Tensor<T> raw, Tensor<T> gradient)
    {
        var numOps = NumOps;
        var data = new T[gradient.Length];
        for (int i = 0; i < gradient.Length; i++)
        {
            double rawVal = numOps.ToDouble(raw.Data[i]);
            double sigmoid = 1.0 / (1.0 + Math.Exp(-rawVal));
            double grad = numOps.ToDouble(gradient.Data[i]);
            data[i] = numOps.FromDouble(grad * sigmoid);
        }

        return new Tensor<T>(data, gradient.Shape);
    }

    private Tensor<T> AddTensors(Tensor<T> left, Tensor<T> right)
    {
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Tensor lengths must match.");
        }

        var numOps = NumOps;
        var data = new T[left.Length];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = numOps.Add(left.Data[i], right.Data[i]);
        }

        return new Tensor<T>(data, left.Shape);
    }

    private static double Clamp01(double value)
    {
        if (value <= 0.0)
        {
            return 0.0;
        }
        if (value >= 1.0 - 1e-6)
        {
            return 1.0 - 1e-6;
        }

        return value;
    }

    private int GetLevelResolution(int level)
    {
        if (_numLevels <= 1)
        {
            return _finestResolution;
        }

        double scale = Math.Exp(level * Math.Log((double)_finestResolution / _coarsestResolution) / (_numLevels - 1));
        int resolution = (int)Math.Floor(_coarsestResolution * scale);
        return Math.Max(1, resolution);
    }

    private static int GetBitfieldLength(int cellCount)
    {
        return (cellCount + 31) / 32;
    }

    private static bool IsBitfieldOccupied(uint[] bitfield, int cellIndex)
    {
        int word = cellIndex >> 5;
        int bit = cellIndex & 31;
        return (bitfield[word] & (1u << bit)) != 0u;
    }

    private static void SetBitfieldOccupied(uint[] bitfield, int cellIndex)
    {
        int word = cellIndex >> 5;
        int bit = cellIndex & 31;
        bitfield[word] |= 1u << bit;
    }

    private static void FillBitfield(uint[] bitfield, int cellCount, bool value)
    {
        uint fill = value ? uint.MaxValue : 0u;
        for (int i = 0; i < bitfield.Length; i++)
        {
            bitfield[i] = fill;
        }

        if (value)
        {
            ClearUnusedBits(bitfield, cellCount);
        }
    }

    private static void ClearUnusedBits(uint[] bitfield, int cellCount)
    {
        int totalBits = bitfield.Length * 32;
        int extraBits = totalBits - cellCount;
        if (extraBits <= 0)
        {
            return;
        }

        uint mask = uint.MaxValue >> extraBits;
        bitfield[bitfield.Length - 1] &= mask;
    }

    private (Tensor<T> origins, Tensor<T> directions) GenerateCameraRays(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int width,
        int height,
        T focalLength)
    {
        int numRays = width * height;
        var origins = new T[numRays * 3];
        var directions = new T[numRays * 3];
        double f = NumOps.ToDouble(focalLength);
        double cx = (width - 1) * 0.5;
        double cy = (height - 1) * 0.5;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double px = (x - cx) / f;
                double py = (y - cy) / f;
                double pz = 1.0;

                double dx = NumOps.ToDouble(cameraRotation[0, 0]) * px +
                            NumOps.ToDouble(cameraRotation[0, 1]) * py +
                            NumOps.ToDouble(cameraRotation[0, 2]) * pz;
                double dy = NumOps.ToDouble(cameraRotation[1, 0]) * px +
                            NumOps.ToDouble(cameraRotation[1, 1]) * py +
                            NumOps.ToDouble(cameraRotation[1, 2]) * pz;
                double dz = NumOps.ToDouble(cameraRotation[2, 0]) * px +
                            NumOps.ToDouble(cameraRotation[2, 1]) * py +
                            NumOps.ToDouble(cameraRotation[2, 2]) * pz;

                double norm = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                if (norm > 0.0)
                {
                    double inv = 1.0 / norm;
                    dx *= inv;
                    dy *= inv;
                    dz *= inv;
                }

                int idx = (y * width + x) * 3;
                origins[idx] = cameraPosition[0];
                origins[idx + 1] = cameraPosition[1];
                origins[idx + 2] = cameraPosition[2];
                directions[idx] = NumOps.FromDouble(dx);
                directions[idx + 1] = NumOps.FromDouble(dy);
                directions[idx + 2] = NumOps.FromDouble(dz);
            }
        }

        return (new Tensor<T>(origins, [numRays, 3]), new Tensor<T>(directions, [numRays, 3]));
    }

    private bool ComputeRayBounds(
        double ox,
        double oy,
        double oz,
        double dx,
        double dy,
        double dz,
        double near,
        double far,
        out double tMin,
        out double tMax)
    {
        tMin = near;
        tMax = far;

        if (!IntersectAxis(ox, dx, _sceneMin[0], _sceneMax[0], ref tMin, ref tMax) ||
            !IntersectAxis(oy, dy, _sceneMin[1], _sceneMax[1], ref tMin, ref tMax) ||
            !IntersectAxis(oz, dz, _sceneMin[2], _sceneMax[2], ref tMin, ref tMax))
        {
            return false;
        }

        return tMax >= tMin;
    }

    private static bool IntersectAxis(
        double o,
        double d,
        double minBound,
        double maxBound,
        ref double tMin,
        ref double tMax)
    {
        const double eps = 1e-8;

        if (Math.Abs(d) < eps)
        {
            return o >= minBound && o <= maxBound;
        }

        double inv = 1.0 / d;
        double t1 = (minBound - o) * inv;
        double t2 = (maxBound - o) * inv;

        if (t1 > t2)
        {
            (t1, t2) = (t2, t1);
        }

        tMin = Math.Max(tMin, t1);
        tMax = Math.Min(tMax, t2);
        return tMax >= tMin;
    }

    private Tensor<T> VolumeRendering(
        Tensor<T> rgb,
        Tensor<T> density,
        int numRays,
        int numSamples,
        double[] rayNear,
        double[] rayFar,
        double[]? sampleTs = null)
    {
        var colors = new T[numRays * 3];
        var rgbData = rgb.Data;
        var densityData = density.Data;

        for (int r = 0; r < numRays; r++)
        {
            double transmittance = 1.0;
            double accumR = 0.0;
            double accumG = 0.0;
            double accumB = 0.0;
            double uniformDeltaT = numSamples > 0 ? (rayFar[r] - rayNear[r]) / numSamples : 0.0;

            if (uniformDeltaT < 0.0)
            {
                uniformDeltaT = 0.0;
            }

            for (int s = 0; s < numSamples; s++)
            {
                int idx = r * numSamples + s;
                double deltaT = uniformDeltaT;
                if (sampleTs != null)
                {
                    double t0 = sampleTs[idx];
                    double t1 = s + 1 < numSamples ? sampleTs[idx + 1] : rayFar[r];
                    if (t1 <= 0.0 || t1 < t0)
                    {
                        t1 = rayFar[r];
                    }

                    deltaT = t1 - t0;
                    if (deltaT < 0.0)
                    {
                        deltaT = 0.0;
                    }
                }

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
            for (int j = 0; j < 3; j++)
            {
                positions[i * 3 + j] = input.Data[i * 6 + j];
                directions[i * 3 + j] = input.Data[i * 6 + 3 + j];
            }
        }

        var posT = new Tensor<T>(positions, [numPoints, 3]);
        var dirT = new Tensor<T>(directions, [numPoints, 3]);

        var (rgb, density) = QueryField(posT, dirT);

        var output = new T[numPoints * 4];
        for (int i = 0; i < numPoints; i++)
        {
            output[i * 4] = rgb.Data[i * 3];
            output[i * 4 + 1] = rgb.Data[i * 3 + 1];
            output[i * 4 + 2] = rgb.Data[i * 3 + 2];
            output[i * 4 + 3] = density.Data[i];
        }

        return new Tensor<T>(output, [numPoints, 4]);
    }

    public override Tensor<T> Backpropagate(Tensor<T> outputGradient)
    {
        if (_lastDensityRaw == null || _lastRgbRaw == null || _lastPositions == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward.");
        }
        if (_densityOutputLayer == null || _featureLayer == null || _colorOutputLayer == null)
        {
            throw new InvalidOperationException("InstantNGP layers are not initialized.");
        }
        if (outputGradient.Shape.Length != 2 || outputGradient.Shape[1] != 4)
        {
            throw new ArgumentException("Output gradient must have shape [N, 4].", nameof(outputGradient));
        }

        int numPoints = _lastDensityRaw.Shape[0];
        var rgbGrad = new T[numPoints * 3];
        var densityGrad = new T[numPoints];
        for (int i = 0; i < numPoints; i++)
        {
            int baseIdx = i * 4;
            rgbGrad[i * 3] = outputGradient.Data[baseIdx];
            rgbGrad[i * 3 + 1] = outputGradient.Data[baseIdx + 1];
            rgbGrad[i * 3 + 2] = outputGradient.Data[baseIdx + 2];
            densityGrad[i] = outputGradient.Data[baseIdx + 3];
        }

        var rgbGradTensor = new Tensor<T>(rgbGrad, [numPoints, 3]);
        var densityGradTensor = new Tensor<T>(densityGrad, [numPoints, 1]);
        var rgbRawGrad = ApplySigmoidGradient(_lastRgbRaw, rgbGradTensor);
        var densityRawGrad = ApplySoftplusGradient(_lastDensityRaw, densityGradTensor);

        Tensor<T> gradColor = _colorOutputLayer.Backward(rgbRawGrad);
        for (int i = _colorLayers.Count - 1; i >= 0; i--)
        {
            gradColor = _colorLayers[i].Backward(gradColor);
        }

        var gradFeatures = new T[numPoints * _featureDim];
        int colorStride = _featureDim + 3;
        for (int i = 0; i < numPoints; i++)
        {
            int colorBase = i * colorStride;
            int featureBase = i * _featureDim;
            for (int f = 0; f < _featureDim; f++)
            {
                gradFeatures[featureBase + f] = gradColor.Data[colorBase + f];
            }
        }

        var gradFeatureTensor = new Tensor<T>(gradFeatures, [numPoints, _featureDim]);
        var gradFromFeatures = _featureLayer.Backward(gradFeatureTensor);
        var gradFromDensity = _densityOutputLayer.Backward(densityRawGrad);
        var gradDensityHidden = AddTensors(gradFromFeatures, gradFromDensity);

        Tensor<T> gradHash = gradDensityHidden;
        for (int i = _densityLayers.Count - 1; i >= 0; i--)
        {
            gradHash = _densityLayers[i].Backward(gradHash);
        }

        _lastHashFeatureGradients = gradHash;

        return new Tensor<T>(new T[numPoints * 6], [numPoints, 6]);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)       
    {
        var prediction = ForwardWithMemory(input);

        if (LossFunction == null)
        {
            throw new InvalidOperationException("Loss function not set.");
        }

        LastLoss = LossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        var lossGradient = LossFunction.ComputeGradient(prediction, expectedOutput);
        Backpropagate(lossGradient);

        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining && layer.ParameterCount > 0)
            {
                layer.UpdateParameters(_learningRate);
            }
        }

        if (_lastHashFeatureGradients != null)
        {
            ApplyHashTableGradients(_lastHashFeatureGradients);
        }

        _trainingStep++;
        if (_trainingStep % _occupancyUpdateInterval == 0)
        {
            UpdateOccupancyGrid();
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        int layerParameterCount = ParameterCount;
        int hashParameterCount = _numLevels * _hashTableSize * _featuresPerLevel;
        if (parameters.Length != layerParameterCount &&
            parameters.Length != layerParameterCount + hashParameterCount)
        {
            throw new ArgumentException(
                $"Expected {layerParameterCount} or {layerParameterCount + hashParameterCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        if (layerParameterCount > 0)
        {
            SetParameters(parameters.GetSubVector(0, layerParameterCount));
        }

        if (parameters.Length == layerParameterCount + hashParameterCount)
        {
            int offset = layerParameterCount;
            for (int level = 0; level < _numLevels; level++)
            {
                var data = new T[_hashTableSize * _featuresPerLevel];
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = parameters[offset++];
                }

                _hashTables[level] = new Tensor<T>(data, [_hashTableSize, _featuresPerLevel]);
            }
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        int hashParameterCount = _numLevels * _hashTableSize * _featuresPerLevel;

        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "HashTableSize", _hashTableSize },
                { "NumLevels", _numLevels },
                { "FeaturesPerLevel", _featuresPerLevel },
                { "FinestResolution", _finestResolution },
                { "CoarsestResolution", _coarsestResolution },
                { "MlpHiddenDim", _mlpHiddenDim },
                { "MlpNumLayers", _mlpNumLayers },
                { "FeatureDim", _featureDim },
                { "ColorHiddenDim", _colorHiddenDim },
                { "ColorNumLayers", _colorNumLayers },
                { "UseOccupancyGrid", _useOccupancyGrid },
                { "OccupancyGridResolution", _occupancyGridResolution },
                { "OccupancyDecay", _occupancyDecay },
                { "OccupancyThreshold", _occupancyThreshold },
                { "OccupancyUpdateInterval", _occupancyUpdateInterval },
                { "OccupancySamplesPerCell", _occupancySamplesPerCell },
                { "OccupancyJitter", _occupancyJitter },
                { "RenderSamples", _renderSamples },
                { "RenderNearBound", NumOps.ToDouble(_renderNearBound) },
                { "RenderFarBound", NumOps.ToDouble(_renderFarBound) },
                { "LearningRate", NumOps.ToDouble(_learningRate) },
                { "SceneBoundsMin", new[] { _sceneMin[0], _sceneMin[1], _sceneMin[2] } },
                { "SceneBoundsMax", new[] { _sceneMax[0], _sceneMax[1], _sceneMax[2] } },
                { "LayerCount", Layers.Count },
                { "TotalParameters", ParameterCount + hashParameterCount }
            },
            ModelData = Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_hashTableSize);
        writer.Write(_numLevels);
        writer.Write(_featuresPerLevel);
        writer.Write(_finestResolution);
        writer.Write(_coarsestResolution);
        writer.Write(_mlpHiddenDim);
        writer.Write(_mlpNumLayers);
        writer.Write(_featureDim);
        writer.Write(_colorHiddenDim);
        writer.Write(_colorNumLayers);
        writer.Write(_useOccupancyGrid);
        writer.Write(_occupancyGridResolution);
        writer.Write(NumOps.ToDouble(_learningRate));
        writer.Write(_occupancyDecay);
        writer.Write(_occupancyThreshold);
        writer.Write(_occupancyUpdateInterval);
        writer.Write(_occupancySamplesPerCell);
        writer.Write(_occupancyJitter);
        writer.Write(_renderSamples);
        writer.Write(NumOps.ToDouble(_renderNearBound));
        writer.Write(NumOps.ToDouble(_renderFarBound));
        writer.Write(_sceneMin[0]);
        writer.Write(_sceneMin[1]);
        writer.Write(_sceneMin[2]);
        writer.Write(_sceneMax[0]);
        writer.Write(_sceneMax[1]);
        writer.Write(_sceneMax[2]);
        writer.Write(_trainingStep);

        writer.Write(_hashTables.Count);
        for (int level = 0; level < _numLevels; level++)
        {
            var table = _hashTables[level].Data;
            writer.Write(table.Length);
            for (int i = 0; i < table.Length; i++)
            {
                writer.Write(NumOps.ToDouble(table[i]));
            }
        }

        if (_occupancyGrid == null)
        {
            writer.Write(false);
        }
        else
        {
            writer.Write(true);
            writer.Write(_occupancyGridResolution);
            var gridData = _occupancyGrid.Data;
            writer.Write(gridData.Length);
            for (int i = 0; i < gridData.Length; i++)
            {
                writer.Write(NumOps.ToDouble(gridData[i]));
            }
        }
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int hashTableSize = reader.ReadInt32();
        int numLevels = reader.ReadInt32();
        int featuresPerLevel = reader.ReadInt32();
        int finestResolution = reader.ReadInt32();
        int coarsestResolution = reader.ReadInt32();
        int mlpHiddenDim = reader.ReadInt32();
        int mlpNumLayers = reader.ReadInt32();
        int featureDim = reader.ReadInt32();
        int colorHiddenDim = reader.ReadInt32();
        int colorNumLayers = reader.ReadInt32();
        bool useOccupancyGrid = reader.ReadBoolean();
        int occupancyGridResolution = reader.ReadInt32();
        double learningRate = reader.ReadDouble();
        double occupancyDecay = reader.ReadDouble();
        double occupancyThreshold = reader.ReadDouble();
        int occupancyUpdateInterval = reader.ReadInt32();
        int occupancySamplesPerCell = reader.ReadInt32();
        double occupancyJitter = reader.ReadDouble();
        int renderSamples = reader.ReadInt32();
        double renderNear = reader.ReadDouble();
        double renderFar = reader.ReadDouble();
        double sceneMinX = reader.ReadDouble();
        double sceneMinY = reader.ReadDouble();
        double sceneMinZ = reader.ReadDouble();
        double sceneMaxX = reader.ReadDouble();
        double sceneMaxY = reader.ReadDouble();
        double sceneMaxZ = reader.ReadDouble();
        _trainingStep = reader.ReadInt32();

        if (hashTableSize != _hashTableSize ||
            numLevels != _numLevels ||
            featuresPerLevel != _featuresPerLevel ||
            finestResolution != _finestResolution ||
            coarsestResolution != _coarsestResolution ||
            mlpHiddenDim != _mlpHiddenDim ||
            mlpNumLayers != _mlpNumLayers ||
            featureDim != _featureDim ||
            colorHiddenDim != _colorHiddenDim ||
            colorNumLayers != _colorNumLayers ||
            useOccupancyGrid != _useOccupancyGrid ||
            occupancyGridResolution != _occupancyGridResolution ||
            Math.Abs(learningRate - NumOps.ToDouble(_learningRate)) > 1e-9 ||
            Math.Abs(occupancyDecay - _occupancyDecay) > 1e-9 ||
            Math.Abs(occupancyThreshold - _occupancyThreshold) > 1e-9 ||
            occupancyUpdateInterval != _occupancyUpdateInterval ||
            occupancySamplesPerCell != _occupancySamplesPerCell ||
            Math.Abs(occupancyJitter - _occupancyJitter) > 1e-9 ||
            renderSamples != _renderSamples ||
            Math.Abs(renderNear - NumOps.ToDouble(_renderNearBound)) > 1e-9 ||
            Math.Abs(renderFar - NumOps.ToDouble(_renderFarBound)) > 1e-9 ||
            Math.Abs(sceneMinX - _sceneMin[0]) > 1e-9 ||
            Math.Abs(sceneMinY - _sceneMin[1]) > 1e-9 ||
            Math.Abs(sceneMinZ - _sceneMin[2]) > 1e-9 ||
            Math.Abs(sceneMaxX - _sceneMax[0]) > 1e-9 ||
            Math.Abs(sceneMaxY - _sceneMax[1]) > 1e-9 ||
            Math.Abs(sceneMaxZ - _sceneMax[2]) > 1e-9)
        {
            throw new InvalidOperationException("Serialized InstantNGP configuration does not match this instance.");
        }

        int tableCount = reader.ReadInt32();
        if (tableCount != _numLevels)
        {
            throw new InvalidOperationException("Serialized hash table count does not match this instance.");
        }

        _hashTables.Clear();
        for (int level = 0; level < _numLevels; level++)
        {
            int length = reader.ReadInt32();
            int expectedLength = _hashTableSize * _featuresPerLevel;
            if (length != expectedLength)
            {
                throw new InvalidOperationException("Serialized hash table length does not match this instance.");
            }

            var data = new T[length];
            for (int i = 0; i < length; i++)
            {
                data[i] = NumOps.FromDouble(reader.ReadDouble());
            }

            _hashTables[level] = new Tensor<T>(data, [_hashTableSize, _featuresPerLevel]);
        }

        bool hasGrid = reader.ReadBoolean();
        if (hasGrid)
        {
            int gridSize = reader.ReadInt32();
            int gridLength = reader.ReadInt32();
            int expectedLength = gridSize * gridSize * gridSize;
            if (gridSize != _occupancyGridResolution || gridLength != expectedLength)
            {
                throw new InvalidOperationException("Serialized occupancy grid does not match this instance.");
            }

            var grid = new T[gridLength];
            for (int i = 0; i < gridLength; i++)
            {
                grid[i] = NumOps.FromDouble(reader.ReadDouble());
            }

            _occupancyGrid = _useOccupancyGrid
                ? new Tensor<T>(grid, [gridSize, gridSize, gridSize])
                : null;
        }
        else
        {
            _occupancyGrid = null;
        }

        RebuildOccupancyBitfield();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new InstantNGP<T>(
            new InstantNGPOptions<T>
            {
                HashTableSize = _hashTableSize,
                NumLevels = _numLevels,
                FeaturesPerLevel = _featuresPerLevel,
                FinestResolution = _finestResolution,
                CoarsestResolution = _coarsestResolution,
                MlpHiddenDim = _mlpHiddenDim,
                MlpNumLayers = _mlpNumLayers,
                FeatureDim = _featureDim,
                ColorHiddenDim = _colorHiddenDim,
                ColorNumLayers = _colorNumLayers,
                UseOccupancyGrid = _useOccupancyGrid,
                OccupancyGridResolution = _occupancyGridResolution,
                LearningRate = NumOps.ToDouble(_learningRate),
                OccupancyDecay = _occupancyDecay,
                OccupancyThreshold = _occupancyThreshold,
                OccupancyUpdateInterval = _occupancyUpdateInterval,
                OccupancySamplesPerCell = _occupancySamplesPerCell,
                OccupancyJitter = _occupancyJitter,
                RenderSamples = _renderSamples,
                RenderNearBound = NumOps.ToDouble(_renderNearBound),
                RenderFarBound = NumOps.ToDouble(_renderFarBound),
                SceneMin = new Vector<T>(3)
                {
                    [0] = NumOps.FromDouble(_sceneMin[0]),
                    [1] = NumOps.FromDouble(_sceneMin[1]),
                    [2] = NumOps.FromDouble(_sceneMin[2])
                },
                SceneMax = new Vector<T>(3)
                {
                    [0] = NumOps.FromDouble(_sceneMax[0]),
                    [1] = NumOps.FromDouble(_sceneMax[1]),
                    [2] = NumOps.FromDouble(_sceneMax[2])
                }
            },
            LossFunction);
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        return ForwardWithMemory(input);
    }
}
