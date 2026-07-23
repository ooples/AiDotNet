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
    private readonly double[][] _gridCoordinates;

    /// <summary>
    /// Precomputed 1D kernel matrices for each dimension.
    /// </summary>
    private Matrix<T>[]? _dimensionKernelMatrices;

    /// <summary>
    /// Eigenvalues of each 1D kernel matrix.
    /// </summary>
    private double[][]? _eigenvalues;

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
    ///     new double[][]
    ///     {
    ///         Enumerable.Range(0, 10).Select(i => (double)i).ToArray(),  // x: 0,1,...,9
    ///         Enumerable.Range(0, 20).Select(i => (double)i).ToArray()   // y: 0,1,...,19
    ///     }
    /// );
    /// </para>
    /// </remarks>
    public GridKernel(IKernelFunction<T>[] dimensionKernels, double[][] gridCoordinates)
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
        _gridCoordinates = gridCoordinates.Select(c => (double[])c.Clone()).ToArray();
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
        _eigenvalues = new double[d][];
        _eigenvectors = new Matrix<T>[d];

        for (int dim = 0; dim < d; dim++)
        {
            int n = _gridCoordinates[dim].Length;
            var K = new Matrix<T>(n, n);

            // Build 1D kernel matrix
            for (int i = 0; i < n; i++)
            {
                for (int j = i; j < n; j++)
                {
                    var xi = new Vector<T>(1);
                    var xj = new Vector<T>(1);
                    xi[0] = _numOps.FromDouble(_gridCoordinates[dim][i]);
                    xj[0] = _numOps.FromDouble(_gridCoordinates[dim][j]);

                    T kval = _dimensionKernels[dim].Calculate(xi, xj);
                    K[i, j] = kval;
                    K[j, i] = kval;
                }
            }

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

        double product = 1.0;

        for (int dim = 0; dim < NumDimensions; dim++)
        {
            var xi = new Vector<T>(1);
            var xj = new Vector<T>(1);
            xi[0] = x1[dim];
            xj[0] = x2[dim];

            double kval = _numOps.ToDouble(_dimensionKernels[dim].Calculate(xi, xj));
            product *= kval;
        }

        return _numOps.FromDouble(product);
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

        var result = new double[totalPoints];
        for (int i = 0; i < totalPoints; i++)
            result[i] = _numOps.ToDouble(v[i]);

        int[] sizes = GridSizes;

        for (int dim = NumDimensions - 1; dim >= 0; dim--)
        {
            int n = sizes[dim];

            // Compute the product of sizes after this dimension
            int rightSize = 1;
            for (int d = dim + 1; d < NumDimensions; d++)
                rightSize *= sizes[d];

            // Compute the product of sizes before this dimension
            int leftSize = totalPoints / (n * rightSize);

            // Apply K_dim to the appropriate mode
            var newResult = new double[totalPoints];

            for (int left = 0; left < leftSize; left++)
            {
                for (int right = 0; right < rightSize; right++)
                {
                    // Extract the fiber along dimension 'dim'
                    var fiber = new double[n];
                    for (int i = 0; i < n; i++)
                    {
                        int idx = left * (n * rightSize) + i * rightSize + right;
                        fiber[i] = result[idx];
                    }

                    // Multiply by K_dim
                    var newFiber = new double[n];
                    for (int i = 0; i < n; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < n; j++)
                        {
                            sum += _numOps.ToDouble(_dimensionKernelMatrices[dim][i, j]) * fiber[j];
                        }
                        newFiber[i] = sum;
                    }

                    // Write back
                    for (int i = 0; i < n; i++)
                    {
                        int idx = left * (n * rightSize) + i * rightSize + right;
                        newResult[idx] = newFiber[i];
                    }
                }
            }

            result = newResult;
        }

        var output = new Vector<T>(totalPoints);
        for (int i = 0; i < totalPoints; i++)
            output[i] = _numOps.FromDouble(result[i]);

        return output;
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
    public double[] GetEigenvalues()
    {
        if (!_precomputed || _eigenvalues is null)
        {
            throw new InvalidOperationException("Call Precompute() first.");
        }

        // All products of eigenvalues
        int total = TotalGridPoints;
        var allEigenvalues = new double[total];
        int[] sizes = GridSizes;

        for (int idx = 0; idx < total; idx++)
        {
            double product = 1.0;
            int remaining = idx;

            for (int dim = NumDimensions - 1; dim >= 0; dim--)
            {
                int dimIdx = remaining % sizes[dim];
                remaining /= sizes[dim];
                product *= _eigenvalues[dim][dimIdx];
            }

            allEigenvalues[idx] = product;
        }

        Array.Sort(allEigenvalues);
        Array.Reverse(allEigenvalues);
        return allEigenvalues;
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
    public double LogDeterminant()
    {
        if (!_precomputed || _eigenvalues is null)
        {
            throw new InvalidOperationException("Call Precompute() first.");
        }

        int[] sizes = GridSizes;
        double logDet = 0;

        for (int dim = 0; dim < NumDimensions; dim++)
        {
            // Product of all other sizes
            int otherProduct = TotalGridPoints / sizes[dim];

            // Sum of log eigenvalues for this dimension
            double sumLogEig = _eigenvalues[dim].Sum(e => Math.Log(Math.Max(e, 1e-10)));

            logDet += otherProduct * sumLogEig;
        }

        return logDet;
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
    public static GridKernel<T> WithRBF(double[][] gridCoordinates, double[]? lengthscales = null)
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
    private (double[] eigenvalues, Matrix<T> eigenvectors) EigenDecomposition(Matrix<T> A)
    {
        int n = A.Rows;
        var eigenvalues = new double[n];
        var eigenvectors = new Matrix<T>(n, n);

        // Use power iteration with deflation (simple but works for small matrices)
        var workMatrix = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                workMatrix[i, j] = _numOps.ToDouble(A[i, j]);

        var rand = RandomHelper.CreateSecureRandom();

        for (int k = 0; k < n; k++)
        {
            // Power iteration
            var v = new double[n];
            for (int i = 0; i < n; i++)
                v[i] = rand.NextDouble();

            double eigenvalue = 0;
            for (int iter = 0; iter < 100; iter++)
            {
                // w = A * v
                var w = new double[n];
                for (int i = 0; i < n; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < n; j++)
                        sum += workMatrix[i, j] * v[j];
                    w[i] = sum;
                }

                // Eigenvalue estimate
                double dot = 0, normV = 0;
                for (int i = 0; i < n; i++)
                {
                    dot += v[i] * w[i];
                    normV += v[i] * v[i];
                }

                // Guard against zero norm (matrix became zero after deflation)
                const double minNorm = 1e-15;
                if (normV < minNorm)
                {
                    eigenvalue = 0;
                    break;
                }
                eigenvalue = dot / normV;

                // Normalize w
                double normW = Math.Sqrt(w.Sum(x => x * x));
                if (normW < minNorm)
                {
                    // w is effectively zero, eigenvalue is zero
                    eigenvalue = 0;
                    break;
                }
                for (int i = 0; i < n; i++)
                    v[i] = w[i] / normW;
            }

            eigenvalues[k] = eigenvalue;
            for (int i = 0; i < n; i++)
                eigenvectors[i, k] = _numOps.FromDouble(v[i]);

            // Deflate
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    workMatrix[i, j] -= eigenvalue * v[i] * v[j];
        }

        return (eigenvalues, eigenvectors);
    }

    #endregion
}
