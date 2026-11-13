namespace AiDotNet.Kernels;

/// <summary>
/// Grid Kernel for exploiting Kronecker structure in regularly-spaced data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When your input data lies on a regular grid (like pixels in an image,
/// or time series sampled at regular intervals), the kernel matrix has special structure
/// that can be exploited for massive computational savings.
///
/// For data on a D-dimensional grid with n_1 × n_2 × ... × n_D points:
/// K = K_1 ⊗ K_2 ⊗ ... ⊗ K_D (Kronecker product)
///
/// This means:
/// - Storage: O(sum of n_i) instead of O((product of n_i)²)
/// - Matrix-vector multiply: O(product of n_i × sum of n_i) instead of O((product of n_i)²)
///
/// Example: 100×100 image grid
/// - Full kernel: 10,000 × 10,000 = 100M entries
/// - Grid kernel: 100 + 100 = 200 entries (500,000× less memory!)
///
/// Limitations:
/// - Only works for data on regular grids
/// - All dimensions must use the same base kernel (but can have different lengthscales)
/// </para>
/// </remarks>
public class GridKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Base kernels for each dimension.
    /// </summary>
    private readonly IKernelFunction<T>[] _dimensionKernels;

    /// <summary>
    /// Grid coordinates along each dimension.
    /// </summary>
    private readonly Vector<T>[] _gridCoordinates;

    /// <summary>
    /// Precomputed 1D kernel matrices for each dimension.
    /// </summary>
    private Matrix<T>[]? _dimensionKernelMatrices;

    /// <summary>
    /// Eigenvalues of each 1D kernel matrix.
    /// </summary>
    private Vector<T>[]? _eigenvalues;

    /// <summary>
    /// Eigenvectors of each 1D kernel matrix.
    /// </summary>
    private Matrix<T>[]? _eigenvectors;

    /// <summary>
    /// Whether precomputation has been done.
    /// </summary>
    private bool _precomputed;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a Grid Kernel with specified kernels per dimension.
    /// </summary>
    /// <param name="dimensionKernels">Kernel for each dimension.</param>
    /// <param name="gridCoordinates">Coordinates along each dimension.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a grid kernel with custom kernels per dimension.
    ///
    /// Example for 2D spatial data:
    /// var kernel = new GridKernel&lt;double&gt;(
    ///     new IKernelFunction&lt;double&gt;[]
    ///     {
    ///         new GaussianKernel&lt;double&gt;(1.0),  // x-dimension
    ///         new GaussianKernel&lt;double&gt;(2.0)   // y-dimension (different lengthscale)
    ///     },
    ///     new[]
    ///     {
    ///         new Vector&lt;double&gt;(Enumerable.Range(0, 10).Select(i => (double)i)),  // x: 0,1,...,9
    ///         new Vector&lt;double&gt;(Enumerable.Range(0, 20).Select(i => (double)i))   // y: 0,1,...,19
    ///     }
    /// );
    /// </para>
    /// </remarks>
    public GridKernel(IKernelFunction<T>[] dimensionKernels, Vector<T>[] gridCoordinates)
    {
        if (dimensionKernels is null)
            throw new ArgumentNullException(nameof(dimensionKernels));
        if (gridCoordinates is null)
            throw new ArgumentNullException(nameof(gridCoordinates));
        if (dimensionKernels.Length != gridCoordinates.Length)
            throw new ArgumentException("Must have same number of kernels and coordinate arrays.");
        if (dimensionKernels.Length == 0)
            throw new ArgumentException("Must have at least one dimension.");

        // Validate each dimension to prevent null/empty which causes divide-by-zero in Kronecker paths
        for (int i = 0; i < dimensionKernels.Length; i++)
        {
            if (dimensionKernels[i] is null)
                throw new ArgumentNullException($"dimensionKernels[{i}]", $"Kernel for dimension {i} cannot be null.");
            if (gridCoordinates[i] is null)
                throw new ArgumentNullException($"gridCoordinates[{i}]", $"Grid coordinates for dimension {i} cannot be null.");
            if (gridCoordinates[i].Length == 0)
                throw new ArgumentException($"Grid coordinates for dimension {i} cannot be empty.");
        }

        _dimensionKernels = (IKernelFunction<T>[])dimensionKernels.Clone();
        _gridCoordinates = gridCoordinates.Select(c => c.Clone()).ToArray();
        _numOps = MathHelper.GetNumericOperations<T>();
        _precomputed = false;
    }

    /// <summary>
    /// Gets the number of dimensions.
    /// </summary>
    public int NumDimensions => _dimensionKernels.Length;

    /// <summary>
    /// Gets the grid sizes along each dimension.
    /// </summary>
    public int[] GridSizes => _gridCoordinates.Select(c => c.Length).ToArray();

    /// <summary>
    /// Gets the total number of grid points.
    /// </summary>
    public int TotalGridPoints => GridSizes.Aggregate(1, (a, b) => a * b);

    /// <summary>
    /// Precomputes the 1D kernel matrices and their eigendecompositions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this once after creating the kernel to precompute
    /// the 1D matrices and their eigendecompositions. This enables efficient
    /// matrix-vector products via the Kronecker structure.
    /// </para>
    /// </remarks>
    public void Precompute()
    {
        int d = NumDimensions;
        _dimensionKernelMatrices = new Matrix<T>[d];
        _eigenvalues = new Vector<T>[d];
        _eigenvectors = new Matrix<T>[d];

        for (int dim = 0; dim < d; dim++)
        {
            // Build 1D kernel matrix over the dimension's grid coordinates
            var coordinates = _gridCoordinates[dim];
            var points = Enumerable.Range(0, coordinates.Length).Select(i => coordinates.GetSubVector(i, 1)).ToArray();
            var K = KernelVectorOps<T>.SymmetricGram(_dimensionKernels[dim], points);

            _dimensionKernelMatrices[dim] = K;

            // Compute eigendecomposition
            (_eigenvalues[dim], _eigenvectors[dim]) = EigenDecomposition(K);
        }

        _precomputed = true;
    }

    /// <summary>
    /// Calculates the kernel value between two grid points.
    /// </summary>
    /// <param name="x1">First point (indices or coordinates).</param>
    /// <param name="x2">Second point (indices or coordinates).</param>
    /// <returns>The kernel value (product of 1D kernels).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For grid kernels, the kernel between two points is:
    /// k(x, x') = k_1(x_1, x'_1) × k_2(x_2, x'_2) × ... × k_D(x_D, x'_D)
    ///
    /// This separability is what enables the Kronecker structure.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have same length.");
        if (x1.Length != NumDimensions)
            throw new ArgumentException($"Expected {NumDimensions}-dimensional vectors.");

        // Product of the one-dimensional kernels k_d(x1[d], x2[d]).
        var dimensionValues = new Vector<T>(Enumerable.Range(0, NumDimensions)
            .Select(dim => _dimensionKernels[dim].Calculate(x1.GetSubVector(dim, 1), x2.GetSubVector(dim, 1))));

        return AiDotNetEngine.Current.Product(dimensionValues);
    }

    /// <summary>
    /// Performs efficient matrix-vector product K * v using Kronecker structure.
    /// </summary>
    /// <param name="v">Vector to multiply (length = total grid points).</param>
    /// <returns>Result K * v.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instead of forming the full N×N kernel matrix and multiplying,
    /// we use the identity: (A ⊗ B) * vec(V) = vec(B * V * A')
    ///
    /// This reduces complexity from O(N²) to O(N × sum of n_i).
    ///
    /// For a 100×100 grid (N = 10,000):
    /// - Naive: 10,000 × 10,000 = 100M operations
    /// - Grid: 10,000 × (100 + 100) = 2M operations (50× faster!)
    /// </para>
    /// </remarks>
    public Vector<T> KroneckerMultiply(Vector<T> v)
    {
        if (!_precomputed || _dimensionKernelMatrices is null)
        {
            throw new InvalidOperationException("Call Precompute() first.");
        }

        int totalPoints = TotalGridPoints;
        if (v.Length != totalPoints)
            throw new ArgumentException($"Vector length must be {totalPoints}.");

        // Use the vec trick: (K_1 ⊗ K_2 ⊗ ... ⊗ K_D) * v
        // = vec(K_D * mat(v) * K_{D-1}' * ... * K_1')
        // We apply each K_i in sequence using reshaping

        return KernelVectorOps<T>.KroneckerMultiply(_dimensionKernelMatrices, v);
    }

    /// <summary>
    /// Gets the eigenvalues of the full Kronecker kernel matrix.
    /// </summary>
    /// <returns>Eigenvalues in sorted order.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The eigenvalues of a Kronecker product are all products
    /// of eigenvalues from each factor: λ_{i,j,...} = λ_1^i × λ_2^j × ...
    ///
    /// This allows computing the spectrum without forming the full matrix.
    /// Useful for:
    /// - Computing log-determinant: log|K| = sum(log(λ_i))
    /// - Checking condition number
    /// - Understanding spectral properties
    /// </para>
    /// </remarks>
    public Vector<T> GetEigenvalues()
    {
        if (!_precomputed || _eigenvalues is null)
        {
            throw new InvalidOperationException("Call Precompute() first.");
        }

        // All products of eigenvalues: the Kronecker product of the per-dimension spectra,
        // flattened with the last dimension fastest, then sorted in descending order.
        var engine = AiDotNetEngine.Current;
        var allEigenvalues = _eigenvalues[0].AsTensor();
        for (int dim = 1; dim < NumDimensions; dim++)
        {
            var outer = engine.TensorOuter(allEigenvalues, _eigenvalues[dim].AsTensor());
            allEigenvalues = engine.Reshape(outer, new[] { allEigenvalues.Length * _eigenvalues[dim].Length });
        }

        return engine.TensorSort(allEigenvalues, -1, descending: true).Values.ToVector();
    }

    /// <summary>
    /// Computes log-determinant using Kronecker structure.
    /// </summary>
    /// <returns>log|K| computed efficiently.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The log-determinant is needed for GP marginal likelihood:
    /// log|K| = sum_i log(λ_i)
    ///
    /// For Kronecker structure: log|K| = sum over dimensions of (product of other sizes) × sum(log(λ_dim))
    /// </para>
    /// </remarks>
    public T LogDeterminant()
    {
        if (!_precomputed || _eigenvalues is null)
        {
            throw new InvalidOperationException("Call Precompute() first.");
        }

        // log|K_1 ⊗ ... ⊗ K_D| = sum_d (N / n_d) × sum(log(max(eig_d, 1e-10)))
        var eigenvalues = _eigenvalues;
        var dimensionTerms = new Vector<T>(Enumerable.Range(0, NumDimensions)
            .Select(dim => ScaledLogEigenvalueSum(eigenvalues[dim], TotalGridPoints / eigenvalues[dim].Length)));

        return AiDotNetEngine.Current.Sum(dimensionTerms);
    }

    /// <summary>
    /// otherProduct × sum(log(max(eigenvalues, 1e-10))) for one dimension.
    /// </summary>
    private T ScaledLogEigenvalueSum(Vector<T> eigenvalues, int otherProduct)
    {
        var engine = AiDotNetEngine.Current;
        var clamped = engine.TensorClampMin(eigenvalues.AsTensor(), _numOps.FromDouble(1e-10));
        return _numOps.Multiply(_numOps.FromDouble(otherProduct), engine.TensorSum(engine.TensorLog(clamped)));
    }

    /// <summary>
    /// Creates a Grid Kernel with RBF kernels for all dimensions.
    /// </summary>
    /// <param name="gridCoordinates">Coordinates along each dimension.</param>
    /// <param name="lengthscales">Lengthscale for each dimension (default 1.0 for all).</param>
    /// <returns>A new Grid Kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Convenience factory for the common case of RBF kernels.
    /// </para>
    /// </remarks>
    public static GridKernel<T> WithRBF(Vector<T>[] gridCoordinates, double[]? lengthscales = null)
    {
        int d = gridCoordinates.Length;
        lengthscales ??= Enumerable.Repeat(1.0, d).ToArray();

        if (lengthscales.Length != d)
            throw new ArgumentException("Lengthscales must match number of dimensions.");

        var kernels = new IKernelFunction<T>[d];
        for (int i = 0; i < d; i++)
        {
            kernels[i] = new GaussianKernel<T>(lengthscales[i]);
        }

        return new GridKernel<T>(kernels, gridCoordinates);
    }

    #region Private Methods

    /// <summary>
    /// Simple eigendecomposition for symmetric matrices.
    /// </summary>
    private (Vector<T> eigenvalues, Matrix<T> eigenvectors) EigenDecomposition(Matrix<T> A)
    {
        // Symmetric eigendecomposition (Saatçi 2011, Kronecker GP inference). Eigh returns ascending
        // eigenvalues; flip to the descending order (eigenvector columns flipped to match).
        var engine = AiDotNetEngine.Current;
        var (values, vectors) = LinalgDispatch<T>.Eigh(Tensor<T>.FromMatrix(A));

        return (engine.TensorFlip(values, new[] { 0 }).ToVector(), engine.TensorFlip(vectors, new[] { 1 }).ToMatrix());
    }

    #endregion
}
