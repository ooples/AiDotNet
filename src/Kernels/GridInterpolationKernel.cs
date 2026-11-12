namespace AiDotNet.Kernels;

/// <summary>
/// Grid Interpolation Kernel (KISS-GP) for scalable Gaussian Process inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> KISS-GP (Kernel Interpolation for Scalable Structured GPs) combines
/// inducing points with grid structure for highly scalable GP inference.
///
/// The key insight: Place inducing points on a regular grid, then use:
/// 1. Interpolation to map data points to the grid
/// 2. Kronecker/Toeplitz structure of the grid for fast computations
///
/// K ≈ W × K_grid × W'
///
/// Where:
/// - W is a sparse interpolation matrix (each data point → nearby grid points)
/// - K_grid has Kronecker structure (efficient to work with)
///
/// W uses the local cubic convolution interpolation of Keys (1981), as in Wilson &amp; Nickisch,
/// "Kernel Interpolation for Scalable Structured Gaussian Processes (KISS-GP)" (ICML 2015) and
/// GPyTorch's <c>Interpolation</c>: every data point draws on the 4 nearest grid points per dimension.
///
/// Complexity for N data points, M grid points:
/// - Standard GP: O(N³)
/// - Inducing points: O(NM² + M³)
/// - KISS-GP: O(N × 4^d + M × Σ n_d) per kernel-vector product
///
/// This enables GPs with millions of data points on commodity hardware.
/// </para>
/// </remarks>
public class GridInterpolationKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Grid points per dimension that cubic convolution draws on (GPyTorch's
    /// <c>interp_points = range(-2, 2)</c>): two on each side of the target.
    /// </summary>
    private const int InterpolationPoints = 4;

    /// <summary>
    /// Relative tolerance of the uniform-spacing check, loose enough for float grids.
    /// </summary>
    private const double SpacingTolerance = 1e-6;

    /// <summary>
    /// The base kernel.
    /// </summary>
    private readonly IKernelFunction<T> _baseKernel;

    /// <summary>
    /// Grid coordinates along each dimension.
    /// </summary>
    private readonly Vector<T>[] _gridCoordinates;

    /// <summary>
    /// Number of grid points per dimension.
    /// </summary>
    private readonly int[] _gridSizes;

    /// <summary>
    /// All grid coordinates, concatenated dimension after dimension.
    /// </summary>
    private readonly Tensor<T> _allCoordinates;

    /// <summary>
    /// First grid coordinate of each dimension, shape [d].
    /// </summary>
    private readonly Tensor<T> _gridStarts;

    /// <summary>
    /// Last grid coordinate of each dimension, shape [d].
    /// </summary>
    private readonly Tensor<T> _gridEnds;

    /// <summary>
    /// Grid spacing of each dimension, shape [d].
    /// </summary>
    private readonly Tensor<T> _gridSpacings;

    /// <summary>
    /// The first four coordinates of each dimension, shape [d, 4].
    /// </summary>
    private readonly Tensor<T> _firstPoints;

    /// <summary>
    /// The last four coordinates of each dimension, shape [d, 4].
    /// </summary>
    private readonly Tensor<T> _lastPoints;

    /// <summary>
    /// Highest valid left-most stencil index of each dimension (grid size minus 4), shape [d].
    /// </summary>
    private readonly Tensor<T> _lastStencilStarts;

    /// <summary>
    /// Row-major stride of each dimension in the flattened grid, shape [d].
    /// </summary>
    private readonly Tensor<T> _gridStrides;

    /// <summary>
    /// Offset of each dimension's coordinates inside <see cref="_allCoordinates"/>, shape [d].
    /// </summary>
    private readonly Tensor<T> _coordinateOffsets;

    /// <summary>
    /// For each dimension, the stencil position (0..3) used by every one of the 4^d stencil combinations.
    /// </summary>
    private readonly Tensor<int>[] _combinationDigits;

    /// <summary>
    /// Distances from the left-bounding grid point to the stencil points, [1, 0, -1, -2]
    /// (GPyTorch's flipped <c>interp_points</c>).
    /// </summary>
    private readonly Tensor<T> _stencilOffsets;

    /// <summary>
    /// Stencil positions [0, 1, 2, 3].
    /// </summary>
    private readonly Tensor<T> _stencilPositions;

    /// <summary>
    /// Precomputed per-dimension grid kernel matrices K_d (Toeplitz for stationary kernels).
    /// </summary>
    private Tensor<T>[]? _gridKernels;

    /// <summary>
    /// Whether precomputation is done.
    /// </summary>
    private bool _precomputed;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a Grid Interpolation Kernel.
    /// </summary>
    /// <param name="baseKernel">The base kernel function (should be stationary for best results).</param>
    /// <param name="gridCoordinates">Strictly increasing, uniformly spaced coordinates along each dimension,
    /// at least four per dimension.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a KISS-GP kernel.
    ///
    /// The grid should cover the range of your data with reasonable resolution.
    /// More grid points = better approximation but more computation.
    ///
    /// Each data point is interpolated from the 4 nearest grid points per dimension with
    /// cubic convolution, so every dimension needs at least 4 uniformly spaced grid points.
    ///
    /// Example for 2D data in [0, 10] × [0, 10]:
    /// var kernel = GridInterpolationKernel&lt;double&gt;.WithUniformGrid(
    ///     new GaussianKernel&lt;double&gt;(1.0),
    ///     new[] { (0.0, 10.0), (0.0, 10.0) },
    ///     gridPointsPerDim: 50);
    /// </para>
    /// </remarks>
    public GridInterpolationKernel(
        IKernelFunction<T> baseKernel,
        Vector<T>[] gridCoordinates)
    {
        Guard.NotNull(baseKernel);
        _baseKernel = baseKernel;
        Guard.NotNull(gridCoordinates);

        if (gridCoordinates.Length == 0)
            throw new ArgumentException("Must have at least one dimension.");

        // Validate each dimension has enough coordinates for the 4-point cubic stencil
        for (int i = 0; i < gridCoordinates.Length; i++)
        {
            if (gridCoordinates[i] is null)
                throw new ArgumentNullException($"gridCoordinates[{i}]", $"Grid coordinates for dimension {i} cannot be null.");
            if (gridCoordinates[i].Length < InterpolationPoints)
                throw new ArgumentException(
                    $"Grid coordinates for dimension {i} must have at least {InterpolationPoints} points " +
                    "for cubic convolution interpolation.");
        }

        _gridCoordinates = gridCoordinates;
        _gridSizes = gridCoordinates.Select(c => c.Length).ToArray();
        _numOps = MathHelper.GetNumericOperations<T>();
        _precomputed = false;

        var engine = AiDotNetEngine.Current;
        int d = gridCoordinates.Length;
        int[] offsets = new int[d];
        for (int dim = 1; dim < d; dim++)
            offsets[dim] = offsets[dim - 1] + _gridSizes[dim - 1];

        _allCoordinates = engine.Concat(gridCoordinates.Select(c => c.AsTensor()).ToList(), 0);
        _gridStarts = engine.TensorIndexSelect(_allCoordinates, IntTensor(offsets), 0);
        _gridEnds = engine.TensorIndexSelect(
            _allCoordinates, IntTensor(Enumerable.Range(0, d).Select(dim => offsets[dim] + _gridSizes[dim] - 1)), 0);
        _gridSpacings = engine.TensorSubtract(
            engine.TensorIndexSelect(_allCoordinates, IntTensor(offsets.Select(o => o + 1)), 0), _gridStarts);

        ValidateUniformSpacing(engine, offsets);

        _firstPoints = engine.Reshape(
            engine.TensorIndexSelect(
                _allCoordinates,
                IntTensor(Enumerable.Range(0, d).SelectMany(dim => Enumerable.Range(offsets[dim], InterpolationPoints))),
                0),
            new[] { d, InterpolationPoints });
        _lastPoints = engine.Reshape(
            engine.TensorIndexSelect(
                _allCoordinates,
                IntTensor(Enumerable.Range(0, d).SelectMany(dim =>
                    Enumerable.Range(offsets[dim] + _gridSizes[dim] - InterpolationPoints, InterpolationPoints))),
                0),
            new[] { d, InterpolationPoints });

        _lastStencilStarts = IntTensor(_gridSizes.Select(g => g - InterpolationPoints)).Cast<T>();
        _coordinateOffsets = IntTensor(offsets).Cast<T>();

        int[] strides = new int[d];
        strides[d - 1] = 1;
        for (int dim = d - 2; dim >= 0; dim--)
            strides[dim] = strides[dim + 1] * _gridSizes[dim + 1];
        _gridStrides = IntTensor(strides).Cast<T>();

        int combinations = IntPower(InterpolationPoints, d);
        _combinationDigits = Enumerable.Range(0, d)
            .Select(dim => IntTensor(Enumerable.Range(0, combinations)
                .Select(k => k / IntPower(InterpolationPoints, d - 1 - dim) % InterpolationPoints)))
            .ToArray();

        _stencilOffsets = engine.TensorArange<T>(1, 1 - InterpolationPoints, -1);
        _stencilPositions = engine.TensorArange<T>(0, InterpolationPoints, 1);
    }

    /// <summary>
    /// Gets the number of dimensions.
    /// </summary>
    public int NumDimensions => _gridCoordinates.Length;

    /// <summary>
    /// Gets the total number of grid points.
    /// </summary>
    public int TotalGridPoints => _gridSizes.Aggregate(1, (a, b) => a * b);

    /// <summary>
    /// Gets the base kernel.
    /// </summary>
    public IKernelFunction<T> BaseKernel => _baseKernel;

    /// <summary>
    /// Precomputes the per-dimension grid kernel matrices for efficient matrix-vector products.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For stationary kernels (where k(x, x') depends only on x - x'),
    /// the grid kernel matrix of each dimension has Toeplitz structure: entry (i, j) only depends
    /// on |i - j|. So we only evaluate the kernel for the first column and fill the matrix from it.
    ///
    /// The full grid kernel is the Kronecker product of these per-dimension matrices, which
    /// <see cref="FastGridMultiply"/> applies one dimension at a time.
    /// </para>
    /// </remarks>
    public void Precompute()
    {
        var engine = AiDotNetEngine.Current;
        _gridKernels = _gridCoordinates.Select(coords => ToeplitzGridKernel(engine, coords)).ToArray();
        _precomputed = true;
    }

    /// <summary>
    /// Calculates the kernel value between two points.
    /// </summary>
    /// <param name="x1">First point.</param>
    /// <param name="x2">Second point.</param>
    /// <returns>Approximate kernel value via grid interpolation.</returns>
    /// <remarks>
    /// <para>Points outside the grid are clamped to its boundary, as the nearest-grid-point interpolation did.</para>
    /// <para>
    /// <b>For Beginners:</b> Approximates k(x1, x2) using interpolation:
    /// k(x1, x2) ≈ w(x1)' × K_grid × w(x2)
    ///
    /// Where w(x) are the cubic interpolation weights to nearby grid points.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != NumDimensions || x2.Length != NumDimensions)
            throw new ArgumentException($"Expected {NumDimensions}-dimensional vectors.");

        var engine = AiDotNetEngine.Current;
        int d = NumDimensions;
        var stencil1 = Interpolate(engine, engine.Reshape(x1.AsTensor(), new[] { 1, d }));
        var stencil2 = Interpolate(engine, engine.Reshape(x2.AsTensor(), new[] { 1, d }));

        // Kernel between the grid points of the two stencils: k(z_i, z_j)
        var points1 = GridPointRows(engine, stencil1.Points);
        var points2 = GridPointRows(engine, stencil2.Points);
        var gridKernel = Tensor<T>.FromMatrix(Matrix<T>.FromRowVectors(
            points1.Select(z1 => points2.Select(z2 => _baseKernel.Calculate(z1, z2)))));

        // w1' × K × w2
        var left = engine.TensorMatMul(stencil1.Weights, gridKernel);
        return engine.TensorSum(engine.TensorMultiply(left, stencil2.Weights));
    }

    /// <summary>
    /// Computes the interpolation matrix W for a set of data points.
    /// </summary>
    /// <param name="X">Data points (n × d); points outside the grid are clamped to its boundary.</param>
    /// <returns>Sparse interpolation matrix W (n × M): the flat grid indices and the weights of the
    /// 4^d stencil entries of each row, both n × 4^d.</returns>
    /// <remarks>
    /// <para>Points outside the grid are clamped to its boundary, as the nearest-grid-point interpolation did.</para>
    /// <para>
    /// <b>For Beginners:</b> The interpolation matrix W maps each data point to its
    /// interpolation weights on the grid. Each row has only a few non-zeros
    /// (4^d entries).
    ///
    /// This sparsity is crucial for efficiency: W × v is O(N × 4^d) not O(N × M).
    /// </para>
    /// </remarks>
    public (Tensor<int> Indices, Matrix<T> Weights) ComputeInterpolationMatrix(Matrix<T> X)
    {
        Guard.NotNull(X);
        if (X.Columns != NumDimensions)
            throw new ArgumentException($"Expected {NumDimensions} columns.");

        var stencil = Interpolate(AiDotNetEngine.Current, Tensor<T>.FromMatrix(X));
        return (stencil.Indices, stencil.Weights.ToMatrix());
    }

    /// <summary>
    /// Performs fast matrix-vector product using the Kronecker structure of the grid kernel.
    /// </summary>
    /// <param name="v">Vector to multiply (length = total grid points).</param>
    /// <returns>K_grid × v.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The grid kernel is a Kronecker product K_1 ⊗ ... ⊗ K_d of small
    /// per-dimension matrices, so K_grid × v is computed by multiplying each K_d along its own axis
    /// of v reshaped to the grid. For M grid points that costs O(M × Σ n_d) instead of O(M²).
    /// </para>
    /// </remarks>
    public Vector<T> FastGridMultiply(Vector<T> v)
    {
        if (!_precomputed || _gridKernels is null)
            throw new InvalidOperationException("Call Precompute() first.");

        int total = TotalGridPoints;
        if (v.Length != total)
            throw new ArgumentException($"Vector length must be {total}.");

        var engine = AiDotNetEngine.Current;
        int d = NumDimensions;
        var grid = engine.Reshape(v.AsTensor(), _gridSizes.ToArray());

        for (int dim = 0; dim < d; dim++)
        {
            // Move this dimension to the front, multiply by K_dim, move it back.
            int axis = dim;
            int[] permutation = new[] { axis }.Concat(Enumerable.Range(0, d).Where(a => a != axis)).ToArray();
            int[] inverse = new int[d];
            for (int a = 0; a < d; a++)
                inverse[permutation[a]] = a;

            int[] movedShape = permutation.Select(a => _gridSizes[a]).ToArray();
            var flat = engine.Reshape(engine.TensorPermute(grid, permutation), new[] { _gridSizes[dim], total / _gridSizes[dim] });
            var product = engine.TensorMatMul(_gridKernels[dim], flat);
            grid = engine.TensorPermute(engine.Reshape(product, movedShape), inverse);
        }

        return engine.Reshape(grid, new[] { total }).ToVector();
    }

    /// <summary>
    /// Performs full KISS-GP kernel-vector product: W × K_grid × W' × v.
    /// </summary>
    /// <param name="v">Vector to multiply (length = number of data points).</param>
    /// <param name="interpolationIndices">Interpolation indices from ComputeInterpolationMatrix.</param>
    /// <param name="interpolationWeights">Interpolation weights from ComputeInterpolationMatrix.</param>
    /// <returns>Approximate K × v.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main operation for GP inference.
    ///
    /// Steps:
    /// 1. W' × v: Scatter data to grid (O(N × 4^d))
    /// 2. K_grid × (W' × v): Fast grid multiply
    /// 3. W × result: Gather from grid to data (O(N × 4^d))
    /// </para>
    /// </remarks>
    public Vector<T> KissGpMultiply(
        Vector<T> v,
        Tensor<int> interpolationIndices,
        Matrix<T> interpolationWeights)
    {
        Guard.NotNull(v);
        Guard.NotNull(interpolationIndices);
        Guard.NotNull(interpolationWeights);

        int n = v.Length;
        int stencilSize = interpolationWeights.Columns;
        if (interpolationIndices.Rank != 2 || interpolationIndices.Shape[0] != n
            || interpolationWeights.Rows != n || interpolationIndices.Shape[1] != stencilSize)
            throw new ArgumentException("Interpolation arrays must match vector length.");

        var engine = AiDotNetEngine.Current;
        int m = TotalGridPoints;
        int entries = n * stencilSize;
        var weights = Tensor<T>.FromMatrix(interpolationWeights);
        var flatIndices = engine.Reshape(interpolationIndices, new[] { entries });

        // Step 1: W' × v (scatter to grid)
        var scaled = engine.TensorMultiply(
            weights,
            engine.TensorBroadcastTo(engine.Reshape(v.AsTensor(), new[] { n, 1 }), new[] { n, stencilSize }));
        var gridVec = engine.TensorIndexAdd(
            new Tensor<T>(new[] { m }), 0, flatIndices, engine.Reshape(scaled, new[] { entries }));

        // Step 2: K_grid × (W' × v)
        var gridOutput = FastGridMultiply(gridVec.ToVector());

        // Step 3: W × result (gather from grid)
        var gathered = engine.Reshape(
            engine.TensorIndexSelect(gridOutput.AsTensor(), flatIndices, 0), new[] { n, stencilSize });
        return engine.ReduceSum(engine.TensorMultiply(weights, gathered), new[] { 1 }, keepDims: false).ToVector();
    }

    /// <summary>
    /// Creates a Grid Interpolation Kernel with automatic grid spacing.
    /// </summary>
    /// <param name="baseKernel">The base kernel.</param>
    /// <param name="bounds">Min and max for each dimension: [(min1, max1), (min2, max2), ...]</param>
    /// <param name="gridPointsPerDim">Number of grid points per dimension (at least 4).</param>
    /// <returns>A new Grid Interpolation Kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Convenience factory that creates a uniform grid.
    ///
    /// Choosing grid points:
    /// - More points = better approximation, but more memory/computation
    /// - Rule of thumb: grid spacing should be smaller than kernel lengthscale
    /// - For RBF with lengthscale 1.0: spacing ≤ 0.25 is usually good
    /// </para>
    /// </remarks>
    public static GridInterpolationKernel<T> WithUniformGrid(
        IKernelFunction<T> baseKernel,
        (T Min, T Max)[] bounds,
        int gridPointsPerDim)
    {
        Guard.NotNull(bounds);
        if (gridPointsPerDim < InterpolationPoints)
            throw new ArgumentOutOfRangeException(nameof(gridPointsPerDim),
                $"gridPointsPerDim must be at least {InterpolationPoints} for cubic convolution interpolation.");

        var engine = AiDotNetEngine.Current;
        var gridCoordinates = bounds
            .Select(b => engine.TensorLinspace(b.Min, b.Max, gridPointsPerDim).ToVector())
            .ToArray();

        return new GridInterpolationKernel<T>(baseKernel, gridCoordinates);
    }

    #region Private Methods

    /// <summary>
    /// Cubic convolution stencils of a batch of points, as GPyTorch's <c>Interpolation.interpolate</c>.
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="x">Points, shape [n, d].</param>
    /// <returns>Flat grid indices [n, 4^d], weights [n, 4^d] and grid point coordinates [n, 4^d, d].</returns>
    private (Tensor<int> Indices, Tensor<T> Weights, Tensor<T> Points) Interpolate(IEngine engine, Tensor<T> x)
    {
        int n = x.Shape[0];
        int d = NumDimensions;
        int[] pointShape = { n, d };
        int[] stencilShape = { n, d, InterpolationPoints };
        var ones = Tensor<T>.CreateDefault(pointShape, _numOps.One);
        var starts = PerDimension(engine, _gridStarts, pointShape);

        // A target outside the grid is clamped to the grid's boundary in each dimension, so it takes the boundary
        // stencil (all weight on the nearest boundary grid point). This keeps the previous nearest-grid-point
        // behaviour: prediction slightly past the training range returns the boundary value instead of throwing.
        // (GPyTorch instead refits the grid bounds to cover the inputs; a fixed grid can only clamp.)
        x = engine.TensorClampTensor(x, starts, PerDimension(engine, _gridEnds, pointShape));

        // Position in grid-index space, its left-bounding grid point and the distance from it.
        var spacing = engine.TensorClampMin(_gridSpacings, _numOps.FromDouble(1e-10));
        var position = engine.TensorDivide(engine.TensorSubtract(x, starts), PerDimension(engine, spacing, pointShape));
        var lowerPoint = engine.TensorFloor(position);
        var relative = engine.TensorSubtract(position, lowerPoint);
        var stencilStart = engine.TensorSubtract(lowerPoint, ones);

        // Keys' cubic convolution kernel at the four stencil distances.
        var scaled = engine.TensorAdd(
            Expand(engine, relative, stencilShape),
            engine.TensorBroadcastTo(engine.Reshape(_stencilOffsets, new[] { 1, 1, InterpolationPoints }), stencilShape));
        var weights = CubicConvolution(engine, scaled, stencilShape);

        // Boundary stencils: GPyTorch puts all weight on the closest of the first or last four grid points.
        var leftBoundary = engine.TensorLessThan(stencilStart, _numOps.Zero);
        var lastStart = PerDimension(engine, _lastStencilStarts, pointShape);
        var rightBoundary = engine.TensorMultiply(
            engine.TensorGreaterThan(stencilStart, lastStart), engine.TensorSubtract(ones, leftBoundary));
        var interior = engine.TensorSubtract(ones, engine.TensorAdd(leftBoundary, rightBoundary));

        var closestFirst = ClosestStencilPoint(engine, x, _firstPoints, stencilShape);
        var closestLast = ClosestStencilPoint(engine, x, _lastPoints, stencilShape);
        weights = engine.TensorAdd(
            engine.TensorMultiply(weights, Expand(engine, interior, stencilShape)),
            engine.TensorAdd(
                engine.TensorMultiply(closestFirst, Expand(engine, leftBoundary, stencilShape)),
                engine.TensorMultiply(closestLast, Expand(engine, rightBoundary, stencilShape))));
        stencilStart = engine.TensorAdd(
            engine.TensorMultiply(stencilStart, interior),
            engine.TensorMultiply(lastStart, rightBoundary));

        // Grid index of every stencil point per dimension, its flat-index contribution and its coordinate.
        var gridIndex = engine.TensorAdd(
            Expand(engine, stencilStart, stencilShape),
            engine.TensorBroadcastTo(engine.Reshape(_stencilPositions, new[] { 1, 1, InterpolationPoints }), stencilShape));
        var flatContribution = engine.TensorMultiply(
            gridIndex, engine.TensorBroadcastTo(engine.Reshape(_gridStrides, new[] { 1, d, 1 }), stencilShape));
        var coordinateIndex = engine.TensorAdd(
            gridIndex, engine.TensorBroadcastTo(engine.Reshape(_coordinateOffsets, new[] { 1, d, 1 }), stencilShape));
        var coordinates = engine.Reshape(
            engine.TensorIndexSelect(
                _allCoordinates, engine.Reshape(coordinateIndex, new[] { n * d * InterpolationPoints }).Cast<int>(), 0),
            stencilShape);

        // Combine the per-dimension stencils into the 4^d tensor-product stencil.
        int combinations = _combinationDigits[0].Length;
        var combinedWeights = Tensor<T>.CreateDefault(new[] { n, combinations }, _numOps.One);
        var combinedIndices = new Tensor<T>(new[] { n, combinations });
        var pointCoordinates = new Tensor<T>[d];
        for (int dim = 0; dim < d; dim++)
        {
            var digits = _combinationDigits[dim];
            combinedWeights = engine.TensorMultiply(
                combinedWeights, engine.TensorIndexSelect(engine.TensorSelect(weights, 1, dim), digits, 1));
            combinedIndices = engine.TensorAdd(
                combinedIndices, engine.TensorIndexSelect(engine.TensorSelect(flatContribution, 1, dim), digits, 1));
            pointCoordinates[dim] = engine.TensorIndexSelect(engine.TensorSelect(coordinates, 1, dim), digits, 1);
        }

        return (combinedIndices.Cast<int>(), combinedWeights, engine.TensorStack(pointCoordinates, 2));
    }

    /// <summary>
    /// Keys' cubic convolution kernel with a = -0.5, written as GPyTorch writes it:
    /// u(s) = 1.5|s|³ - 2.5|s|² + 1 for |s| &lt; 1 and -0.5|s|³ + 2.5|s|² - 4|s| + 2 otherwise.
    /// </summary>
    private Tensor<T> CubicConvolution(IEngine engine, Tensor<T> distance, int[] shape)
    {
        var u = engine.TensorAbs(distance);
        var inner = engine.TensorLessThan(u, _numOps.One);
        var outer = engine.TensorSubtract(Tensor<T>.CreateDefault(shape, _numOps.One), inner);

        var nearCoefficient = engine.TensorAddScalar(engine.TensorMultiplyScalar(u, _numOps.FromDouble(1.5)), _numOps.FromDouble(-2.5));
        var near = engine.TensorAddScalar(engine.TensorMultiply(engine.TensorMultiply(nearCoefficient, u), u), _numOps.One);

        var farCoefficient = engine.TensorAddScalar(engine.TensorMultiplyScalar(u, _numOps.FromDouble(-0.5)), _numOps.FromDouble(2.5));
        var farLinear = engine.TensorAddScalar(engine.TensorMultiply(farCoefficient, u), _numOps.FromDouble(-4.0));
        var far = engine.TensorAddScalar(engine.TensorMultiply(farLinear, u), _numOps.FromDouble(2.0));

        return engine.TensorAdd(engine.TensorMultiply(near, inner), engine.TensorMultiply(far, outer));
    }

    /// <summary>
    /// One-hot weights [n, d, 4] on the closest of four boundary grid points (first one on ties, as torch.min).
    /// </summary>
    private Tensor<T> ClosestStencilPoint(IEngine engine, Tensor<T> x, Tensor<T> boundaryPoints, int[] stencilShape)
    {
        int n = stencilShape[0];
        int d = stencilShape[1];
        var distance = engine.TensorAbs(engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(boundaryPoints, new[] { 1, d, InterpolationPoints }), stencilShape),
            Expand(engine, x, stencilShape)));
        var closest = engine.TensorArgMin(distance, 2).Cast<T>();
        return engine.TensorEquals(
            engine.TensorBroadcastTo(engine.Reshape(_stencilPositions, new[] { 1, 1, InterpolationPoints }), stencilShape),
            Expand(engine, engine.Reshape(closest, new[] { n, d }), stencilShape));
    }

    /// <summary>
    /// Broadcasts a per-dimension [d] tensor to [n, d].
    /// </summary>
    private static Tensor<T> PerDimension(IEngine engine, Tensor<T> values, int[] pointShape)
        => engine.TensorBroadcastTo(engine.Reshape(values, new[] { 1, pointShape[1] }), pointShape);

    /// <summary>
    /// Repeats a [n, d] tensor along a new trailing stencil axis.
    /// </summary>
    private static Tensor<T> Expand(IEngine engine, Tensor<T> perDimension, int[] stencilShape)
        => engine.TensorBroadcastTo(
            engine.Reshape(perDimension, new[] { stencilShape[0], stencilShape[1], 1 }), stencilShape);

    /// <summary>
    /// Dense grid kernel of one dimension from its Toeplitz first column k(z_i, z_0).
    /// </summary>
    private Tensor<T> ToeplitzGridKernel(IEngine engine, Vector<T> coords)
    {
        int n = coords.Length;
        var origin = coords.GetSubVector(0, 1);
        var column = new Vector<T>(Enumerable.Range(0, n).Select(i => _baseKernel.Calculate(coords.GetSubVector(i, 1), origin)));

        // K[i, j] = column[|i - j|]
        var positions = engine.TensorArange<T>(0, n, 1);
        var lag = engine.TensorAbs(engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(positions, new[] { n, 1 }), new[] { n, n }),
            engine.TensorBroadcastTo(engine.Reshape(positions, new[] { 1, n }), new[] { n, n })));
        var gathered = engine.TensorIndexSelect(column.AsTensor(), engine.Reshape(lag, new[] { n * n }).Cast<int>(), 0);
        return engine.Reshape(gathered, new[] { n, n });
    }

    /// <summary>
    /// Rows of a [1, c, d] stencil coordinate tensor as grid point vectors.
    /// </summary>
    private Vector<T>[] GridPointRows(IEngine engine, Tensor<T> points)
    {
        var matrix = engine.Reshape(points, new[] { points.Shape[1], NumDimensions }).ToMatrix();
        return Enumerable.Range(0, matrix.Rows).Select(row => matrix.GetRow(row)).ToArray();
    }

    /// <summary>
    /// Rejects grids that are not strictly increasing and uniformly spaced (cubic convolution needs both).
    /// </summary>
    private void ValidateUniformSpacing(IEngine engine, int[] offsets)
    {
        if (!_numOps.GreaterThan(engine.TensorMinValue(_gridSpacings), _numOps.Zero))
            throw new ArgumentException("Grid coordinates must be strictly increasing in every dimension.");

        // Spacing between neighbours inside each dimension, against that dimension's first spacing.
        int[] pairDimension = Enumerable.Range(0, NumDimensions)
            .SelectMany(dim => Enumerable.Repeat(dim, _gridSizes[dim] - 1)).ToArray();
        int[] pairStart = Enumerable.Range(0, NumDimensions)
            .SelectMany(dim => Enumerable.Range(offsets[dim], _gridSizes[dim] - 1)).ToArray();

        var steps = engine.TensorIndexSelect(engine.TensorDiff(_allCoordinates, 1, 0), IntTensor(pairStart), 0);
        var expected = engine.TensorIndexSelect(_gridSpacings, IntTensor(pairDimension), 0);
        var allowed = engine.TensorMultiplyScalar(
            engine.TensorClampMin(engine.TensorAbs(expected), _numOps.One), _numOps.FromDouble(SpacingTolerance));
        var excess = engine.TensorSubtract(engine.TensorAbs(engine.TensorSubtract(steps, expected)), allowed);

        int worst = engine.TensorArgMax(excess, 0).ToArray()[0];
        T[] excessValues = excess.ToArray();
        if (_numOps.GreaterThan(excessValues[worst], _numOps.Zero))
        {
            int dim = pairDimension[worst];
            int i = pairStart[worst] - offsets[dim] + 1;
            throw new ArgumentException(
                $"Grid dimension {dim} is not uniformly spaced. Cubic convolution interpolation requires uniform spacing. " +
                $"Expected spacing {expected.ToArray()[worst]}, got {steps.ToArray()[worst]} between indices {i - 1} and {i}.");
        }
    }

    /// <summary>
    /// Wraps integer indices as a 1-D index tensor.
    /// </summary>
    private static Tensor<int> IntTensor(IEnumerable<int> values)
    {
        int[] data = values.ToArray();
        return new Tensor<int>(data, new[] { data.Length });
    }

    /// <summary>
    /// Integer power for small stencil counts.
    /// </summary>
    private static int IntPower(int value, int exponent)
    {
        int result = 1;
        for (int i = 0; i < exponent; i++)
            result *= value;
        return result;
    }

    #endregion
}
