namespace AiDotNet.Kernels;

/// <summary>
/// Product Structure Kernel for modeling multiplicative interactions between feature groups.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Product Structure Kernel assumes the underlying function has
/// a multiplicative structure: f(x) = f_1(x_G1) × f_2(x_G2) × ... × f_K(x_GK)
///
/// Where x_Gi is the subset of features in group i.
///
/// The kernel is: k(x, x') = k_1(x_G1, x'_G1) × k_2(x_G2, x'_G2) × ... × k_K(x_GK, x'_GK)
///
/// This is useful when:
/// - Features naturally group together (e.g., spatial × temporal)
/// - There's known multiplicative interaction structure
/// - You want to reduce computation via Kronecker structure
///
/// Example: Modeling sales as (location effects) × (time effects) × (product effects)
///
/// Compare to Additive Structure:
/// - Additive: f = f_1 + f_2 + ... (sum of independent effects)
/// - Product: f = f_1 × f_2 × ... (multiplicative interaction)
///
/// Product kernels can capture stronger interactions but are less interpretable.
/// </para>
/// </remarks>
public class ProductStructureKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Kernels for each feature group.
    /// </summary>
    private readonly IKernelFunction<T>[] _groupKernels;

    /// <summary>
    /// Feature indices for each group.
    /// </summary>
    private readonly int[][] _featureGroups;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a Product Structure Kernel.
    /// </summary>
    /// <param name="groupKernels">Kernel for each feature group.</param>
    /// <param name="featureGroups">Feature indices for each group.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a kernel with multiplicative structure.
    ///
    /// Example for 5D data where dims [0,1] interact multiplicatively with dims [2,3,4]:
    /// var kernel = new ProductStructureKernel&lt;double&gt;(
    ///     new IKernelFunction&lt;double&gt;[]
    ///     {
    ///         new GaussianKernel&lt;double&gt;(1.0),  // For dims 0,1
    ///         new GaussianKernel&lt;double&gt;(2.0)   // For dims 2,3,4
    ///     },
    ///     new int[][]
    ///     {
    ///         new[] { 0, 1 },      // Group 1: dims 0,1
    ///         new[] { 2, 3, 4 }    // Group 2: dims 2,3,4
    ///     }
    /// );
    /// </para>
    /// </remarks>
    public ProductStructureKernel(IKernelFunction<T>[] groupKernels, int[][] featureGroups)
    {
        if (groupKernels is null)
            throw new ArgumentNullException(nameof(groupKernels));
        if (featureGroups is null)
            throw new ArgumentNullException(nameof(featureGroups));
        if (groupKernels.Length != featureGroups.Length)
            throw new ArgumentException("Must have same number of kernels and feature groups.");
        if (groupKernels.Length == 0)
            throw new ArgumentException("Must have at least one group.");

        _groupKernels = (IKernelFunction<T>[])groupKernels.Clone();
        _featureGroups = featureGroups.Select(g => (int[])g.Clone()).ToArray();
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the number of feature groups.
    /// </summary>
    public int NumGroups => _groupKernels.Length;

    /// <summary>
    /// Gets the feature indices for each group.
    /// </summary>
    public int[][] FeatureGroups => _featureGroups.Select(g => (int[])g.Clone()).ToArray();

    /// <summary>
    /// Calculates the product kernel value.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>Product of kernel values over all groups.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes:
    /// k(x, x') = k_1(x_G1, x'_G1) × k_2(x_G2, x'_G2) × ...
    ///
    /// Each group kernel sees only its designated features.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have same length.");

        double product = 1.0;

        for (int g = 0; g < NumGroups; g++)
        {
            var indices = _featureGroups[g];

            // Extract features for this group
            var x1g = new Vector<T>(indices.Length);
            var x2g = new Vector<T>(indices.Length);

            for (int i = 0; i < indices.Length; i++)
            {
                x1g[i] = x1[indices[i]];
                x2g[i] = x2[indices[i]];
            }

            // Compute group kernel
            double kg = _numOps.ToDouble(_groupKernels[g].Calculate(x1g, x2g));
            product *= kg;
        }

        return _numOps.FromDouble(product);
    }

    /// <summary>
    /// Computes the kernel matrix with Kronecker structure (for gridded data).
    /// </summary>
    /// <param name="Xgroups">Data split by group: Xgroups[g] is (n_g × d_g) for group g.</param>
    /// <returns>Kernel matrices for each group (can use Kronecker product).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When data has Kronecker structure (each group's features
    /// vary independently), the full kernel matrix is:
    /// K = K_1 ⊗ K_2 ⊗ ... ⊗ K_G
    ///
    /// This method returns the individual K_g matrices. Use them with Kronecker
    /// operations for efficient computation.
    ///
    /// For example, if you have spatial×temporal data:
    /// - K_spatial: 100 × 100
    /// - K_temporal: 24 × 24
    /// - Full K would be: 2400 × 2400
    /// - But Kronecker structure lets you work with the small matrices directly
    /// </para>
    /// </remarks>
    public Matrix<T>[] ComputeGroupKernelMatrices(Matrix<T>[] Xgroups)
    {
        if (Xgroups.Length != NumGroups)
            throw new ArgumentException($"Expected {NumGroups} group matrices.");

        var kernelMatrices = new Matrix<T>[NumGroups];

        for (int g = 0; g < NumGroups; g++)
        {
            var X = Xgroups[g];
            int n = X.Rows;
            var K = new Matrix<T>(n, n);

            for (int i = 0; i < n; i++)
            {
                var xi = GetRow(X, i);
                for (int j = i; j < n; j++)
                {
                    var xj = GetRow(X, j);
                    T kval = _groupKernels[g].Calculate(xi, xj);
                    K[i, j] = kval;
                    K[j, i] = kval;
                }
            }

            kernelMatrices[g] = K;
        }

        return kernelMatrices;
    }

    /// <summary>
    /// Performs efficient Kronecker matrix-vector product.
    /// </summary>
    /// <param name="groupKernelMatrices">Kernel matrices from ComputeGroupKernelMatrices.</param>
    /// <param name="v">Vector to multiply (length = product of group sizes).</param>
    /// <returns>(K_1 ⊗ K_2 ⊗ ... ⊗ K_G) × v.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uses the vec-trick for efficient Kronecker multiplication.
    ///
    /// Instead of forming the full Kronecker product (which could be huge),
    /// we apply each factor matrix in sequence with appropriate reshaping.
    ///
    /// Complexity: O(N × sum of n_g) instead of O(N²)
    /// </para>
    /// </remarks>
    public Vector<T> KroneckerMultiply(Matrix<T>[] groupKernelMatrices, Vector<T> v)
    {
        if (groupKernelMatrices.Length != NumGroups)
            throw new ArgumentException($"Expected {NumGroups} kernel matrices.");

        // Validate kernel matrices are square
        for (int g = 0; g < groupKernelMatrices.Length; g++)
        {
            var K = groupKernelMatrices[g];
            if (K.Rows != K.Columns)
                throw new ArgumentException($"Kernel matrix for group {g} must be square. Got {K.Rows}x{K.Columns}.");
        }

        int[] sizes = groupKernelMatrices.Select(K => K.Rows).ToArray();

        // Check for overflow in total size computation
        long totalLong = 1;
        foreach (int sz in sizes)
        {
            totalLong *= sz;
            if (totalLong > int.MaxValue)
                throw new OverflowException(
                    $"Total Kronecker product size exceeds int.MaxValue ({int.MaxValue}). " +
                    $"Consider using smaller group sizes.");
        }
        int total = (int)totalLong;

        if (v.Length != total)
            throw new ArgumentException($"Vector length must be {total}.");

        var result = new double[total];
        for (int i = 0; i < total; i++)
            result[i] = _numOps.ToDouble(v[i]);

        // Apply each kernel matrix
        for (int g = NumGroups - 1; g >= 0; g--)
        {
            int n = sizes[g];
            var K = groupKernelMatrices[g];

            // Compute sizes before and after this group
            int rightSize = 1;
            for (int gg = g + 1; gg < NumGroups; gg++)
                rightSize *= sizes[gg];

            int leftSize = total / (n * rightSize);

            var newResult = new double[total];

            // Apply K to mode g
            for (int left = 0; left < leftSize; left++)
            {
                for (int right = 0; right < rightSize; right++)
                {
                    // Extract fiber
                    var fiber = new double[n];
                    for (int i = 0; i < n; i++)
                    {
                        int idx = left * (n * rightSize) + i * rightSize + right;
                        fiber[i] = result[idx];
                    }

                    // Multiply by K
                    for (int i = 0; i < n; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < n; j++)
                            sum += _numOps.ToDouble(K[i, j]) * fiber[j];

                        int idx = left * (n * rightSize) + i * rightSize + right;
                        newResult[idx] = sum;
                    }
                }
            }

            result = newResult;
        }

        var output = new Vector<T>(total);
        for (int i = 0; i < total; i++)
            output[i] = _numOps.FromDouble(result[i]);

        return output;
    }

    /// <summary>
    /// Computes log-determinant using Kronecker structure.
    /// </summary>
    /// <param name="groupKernelMatrices">Kernel matrices from ComputeGroupKernelMatrices.</param>
    /// <returns>log|K_1 ⊗ K_2 ⊗ ... ⊗ K_G|.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For Kronecker products:
    /// log|K_1 ⊗ ... ⊗ K_G| = sum_g (N/n_g) × log|K_g|
    ///
    /// Where N = total points, n_g = points in group g.
    ///
    /// This is much cheaper than computing the full determinant!
    /// </para>
    /// </remarks>
    public double LogDeterminant(Matrix<T>[] groupKernelMatrices)
    {
        if (groupKernelMatrices.Length != NumGroups)
            throw new ArgumentException($"Expected {NumGroups} kernel matrices.");

        int[] sizes = groupKernelMatrices.Select(K => K.Rows).ToArray();
        int total = sizes.Aggregate(1, (a, b) => a * b);

        double logDet = 0;

        for (int g = 0; g < NumGroups; g++)
        {
            var K = groupKernelMatrices[g];
            int n = sizes[g];
            int otherProduct = total / n;

            // Compute log-determinant of K via Cholesky
            try
            {
                var L = CholeskyDecomposition(K);
                double logDetG = 0;
                for (int i = 0; i < n; i++)
                    logDetG += Math.Log(_numOps.ToDouble(L[i, i]));
                logDetG *= 2; // log|K| = 2 × sum(log(diag(L)))

                logDet += otherProduct * logDetG;
            }
            catch (InvalidOperationException)
            {
                // Cholesky failed - matrix is not positive definite
                return double.NegativeInfinity;
            }
            catch (ArithmeticException)
            {
                // Numerical issues (e.g., NaN/Inf values)
                return double.NegativeInfinity;
            }
        }

        return logDet;
    }

    /// <summary>
    /// Creates a Product Structure Kernel with RBF for all groups.
    /// </summary>
    /// <param name="featureGroups">Feature indices for each group.</param>
    /// <param name="lengthscales">Lengthscale for each group (default 1.0).</param>
    /// <returns>A new Product Structure Kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Convenience factory using RBF kernels.
    /// </para>
    /// </remarks>
    public static ProductStructureKernel<T> WithRBF(int[][] featureGroups, double[]? lengthscales = null)
    {
        int numGroups = featureGroups.Length;
        lengthscales ??= Enumerable.Repeat(1.0, numGroups).ToArray();

        if (lengthscales.Length != numGroups)
            throw new ArgumentException("Lengthscales must match number of groups.");

        var kernels = new IKernelFunction<T>[numGroups];
        for (int g = 0; g < numGroups; g++)
        {
            kernels[g] = new GaussianKernel<T>(lengthscales[g]);
        }

        return new ProductStructureKernel<T>(kernels, featureGroups);
    }

    /// <summary>
    /// Creates a Product Structure Kernel with automatic grouping (one feature per group).
    /// </summary>
    /// <param name="numFeatures">Total number of features.</param>
    /// <param name="baseLengthscale">Lengthscale for all groups.</param>
    /// <returns>A fully factorized Product Kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates an ARD-like kernel where:
    /// k(x, x') = k(x_1, x'_1) × k(x_2, x'_2) × ... × k(x_d, x'_d)
    ///
    /// This assumes features are independent but interact multiplicatively.
    /// </para>
    /// </remarks>
    public static ProductStructureKernel<T> FullyFactorized(int numFeatures, double baseLengthscale = 1.0)
    {
        var groups = new int[numFeatures][];
        for (int i = 0; i < numFeatures; i++)
        {
            groups[i] = new[] { i };
        }

        return WithRBF(groups, Enumerable.Repeat(baseLengthscale, numFeatures).ToArray());
    }

    #region Private Methods

    /// <summary>
    /// Extracts a row from a matrix.
    /// </summary>
    private Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var result = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
            result[j] = matrix[row, j];
        return result;
    }

    /// <summary>
    /// Cholesky decomposition.
    /// </summary>
    private Matrix<T> CholeskyDecomposition(Matrix<T> A)
    {
        int n = A.Rows;
        var L = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                    sum += _numOps.ToDouble(L[i, k]) * _numOps.ToDouble(L[j, k]);

                if (i == j)
                {
                    double diag = _numOps.ToDouble(A[i, i]) - sum;
                    if (diag <= 0)
                        throw new InvalidOperationException("Matrix not positive definite.");
                    L[i, j] = _numOps.FromDouble(Math.Sqrt(diag));
                }
                else
                {
                    double ljj = _numOps.ToDouble(L[j, j]);
                    L[i, j] = _numOps.FromDouble((_numOps.ToDouble(A[i, j]) - sum) / ljj);
                }
            }
        }

        return L;
    }

    #endregion
}
