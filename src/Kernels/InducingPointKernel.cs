namespace AiDotNet.Kernels;

/// <summary>
/// Inducing Point Kernel for sparse Gaussian Process approximations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Inducing Point Kernel is fundamental to making GPs scale to large datasets.
/// Instead of working with all N training points (O(N³) complexity), we use a smaller set
/// of M "inducing points" (O(NM² + M³) complexity, where M &lt;&lt; N).
///
/// The key idea: Instead of modeling the full function f, we model:
/// 1. The function values at M inducing points: u = f(Z)
/// 2. The conditional distribution: f(x) | u
///
/// The approximation makes f(x) conditionally independent given the inducing values u.
/// The closer the inducing points are to your data, the better the approximation.
///
/// This kernel computes the "Q_ff" approximation:
/// Q(x, x') = k(x, Z) × k(Z, Z)⁻¹ × k(Z, x')
///
/// Where:
/// - Z are the inducing point locations (M × d)
/// - k(x, Z) is the cross-covariance (1 × M)
/// - k(Z, Z) is the inducing point covariance (M × M)
///
/// This is the Nyström approximation of the full kernel matrix.
/// </para>
/// <para>
/// Usage patterns:
/// - Use with SparseVariationalGaussianProcess for scalable GP regression
/// - Inducing points can be learned via optimization
/// - Good initialization: subset of training data or k-means centroids
/// </para>
/// </remarks>
public class InducingPointKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The base kernel function.
    /// </summary>
    private readonly IKernelFunction<T> _baseKernel;

    /// <summary>
    /// The inducing point locations.
    /// </summary>
    [AiDotNet.Attributes.Buffer]
    private Matrix<T> _inducingPoints;

    /// <summary>
    /// Number of inducing points.
    /// </summary>
    private int _numInducingPoints;

    /// <summary>
    /// Precomputed k(Z, Z).
    /// </summary>
    private Matrix<T>? _Kzz;

    /// <summary>
    /// Cholesky decomposition of k(Z, Z).
    /// </summary>
    private Matrix<T>? _Lzz;

    /// <summary>
    /// Whether precomputed matrices are valid.
    /// </summary>
    private bool _matricesValid;

    /// <summary>
    /// Small regularization constant.
    /// </summary>
    private readonly double _jitter;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes an Inducing Point Kernel.
    /// </summary>
    /// <param name="baseKernel">The base kernel function.</param>
    /// <param name="inducingPoints">Initial inducing point locations (M × d).</param>
    /// <param name="jitter">Regularization constant for numerical stability.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates an inducing point kernel with specified locations.
    ///
    /// Choosing inducing points:
    /// - Start with M = 100-500 for most problems
    /// - Good initialization: k-means centroids of training data
    /// - Inducing points can be optimized during training
    ///
    /// The jitter parameter adds a small value to the diagonal of k(Z, Z) for
    /// numerical stability when inverting.
    /// </para>
    /// </remarks>
    public InducingPointKernel(IKernelFunction<T> baseKernel, Matrix<T> inducingPoints, double jitter = 1e-6)
    {
        if (baseKernel is null) throw new ArgumentNullException(nameof(baseKernel));
        if (inducingPoints is null) throw new ArgumentNullException(nameof(inducingPoints));
        if (inducingPoints.Rows < 1)
            throw new ArgumentException("Must have at least one inducing point.", nameof(inducingPoints));
        if (jitter <= 0)
            throw new ArgumentException("Jitter must be positive.", nameof(jitter));

        _baseKernel = baseKernel;
        _inducingPoints = inducingPoints;
        _numInducingPoints = inducingPoints.Rows;
        _jitter = jitter;
        _numOps = MathHelper.GetNumericOperations<T>();
        _matricesValid = false;
    }

    /// <summary>
    /// Gets the base kernel.
    /// </summary>
    public IKernelFunction<T> BaseKernel => _baseKernel;

    /// <summary>
    /// Gets the number of inducing points.
    /// </summary>
    public int NumInducingPoints => _numInducingPoints;

    /// <summary>
    /// Gets a copy of the inducing points.
    /// </summary>
    /// <returns>Matrix of inducing point locations.</returns>
    public Matrix<T> GetInducingPoints()
    {
        return _inducingPoints.Clone();
    }

    /// <summary>
    /// Updates the inducing point locations.
    /// </summary>
    /// <param name="newInducingPoints">New inducing point locations.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this to update inducing points during optimization.
    /// The precomputed matrices will be invalidated and recomputed on next use.
    /// </para>
    /// </remarks>
    public void UpdateInducingPoints(Matrix<T> newInducingPoints)
    {
        if (newInducingPoints is null) throw new ArgumentNullException(nameof(newInducingPoints));
        if (newInducingPoints.Rows == 0)
            throw new ArgumentException("Inducing points matrix cannot be empty. Must have at least one inducing point.", nameof(newInducingPoints));
        _inducingPoints = newInducingPoints;
        _numInducingPoints = newInducingPoints.Rows;
        _matricesValid = false;
    }

    /// <summary>
    /// Precomputes the inducing point matrices for efficiency.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This precomputes k(Z, Z) and its Cholesky decomposition.
    /// Call this once after setting inducing points, before making many kernel evaluations.
    /// If you modify inducing points, call this again.
    /// </para>
    /// </remarks>
    public void PrecomputeMatrices()
    {
        int m = _numInducingPoints;
        var engine = AiDotNetEngine.Current;

        // Compute k(Z, Z) + jitter * I
        var inducingRows = Enumerable.Range(0, m).Select(i => _inducingPoints.GetRow(i)).ToArray();
        var gram = KernelVectorOps<T>.SymmetricGram(_baseKernel, inducingRows);
        _Kzz = engine.MatrixAdd(gram, engine.MatrixMultiplyScalar(Matrix<T>.CreateIdentity(m), _numOps.FromDouble(_jitter)));

        // Compute Cholesky decomposition; fall back to the clamped factorisation when K_zz is not
        // numerically positive definite.
        var (factor, info) = LinalgDispatch<T>.CholeskyEx(Tensor<T>.FromMatrix(_Kzz));
        _Lzz = info.GetFlatIndexValue(0) == 0 ? factor.ToMatrix() : CholeskyDecomposition(_Kzz);
        _matricesValid = true;
    }

    /// <summary>
    /// Calculates the approximate kernel value (Nyström approximation).
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The approximate kernel value: k(x1, Z) × k(Z, Z)⁻¹ × k(Z, x2).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes the low-rank approximation of the kernel.
    ///
    /// Q(x, x') = k(x, Z) × k(Z, Z)⁻¹ × k(Z, x')
    ///
    /// This approximates k(x, x') using only the inducing points.
    /// The approximation is exact when x and x' are inducing points.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (!_matricesValid)
        {
            PrecomputeMatrices();
        }

        if (_Lzz is null)
        {
            throw new InvalidOperationException("Matrices not properly initialized.");
        }

        // Compute k(x1, Z) and k(x2, Z)
        var inducingRows = Enumerable.Range(0, _numInducingPoints).Select(i => _inducingPoints.GetRow(i)).ToArray();
        var kx1z = new Vector<T>(inducingRows.Select(zi => _baseKernel.Calculate(x1, zi)));
        var kx2z = new Vector<T>(inducingRows.Select(zi => _baseKernel.Calculate(x2, zi)));

        // Solve L * v1 = k(x1, Z) for v1
        var v1 = SolveLowerTriangular(_Lzz, kx1z);
        var v2 = SolveLowerTriangular(_Lzz, kx2z);

        // Q(x1, x2) = v1^T * v2
        return AiDotNetEngine.Current.DotProduct(v1, v2);
    }

    /// <summary>
    /// Computes the cross-covariance between test points and inducing points.
    /// </summary>
    /// <param name="X">Test points (N × d).</param>
    /// <returns>Cross-covariance matrix k(X, Z) of shape (N × M).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is useful for making predictions in sparse GPs.
    /// It computes k(X, Z), the covariance between your test points and the inducing points.
    /// </para>
    /// </remarks>
    public Matrix<T> ComputeCrossCovariance(Matrix<T> X)
    {
        int n = X.Rows;
        if (n == 0)
            return new Matrix<T>(0, _numInducingPoints);

        var inducingRows = Enumerable.Range(0, _numInducingPoints).Select(j => _inducingPoints.GetRow(j)).ToArray();
        return new Matrix<T>(Enumerable.Range(0, n)
            .Select(i => X.GetRow(i))
            .Select(xi => inducingRows.Select(zj => _baseKernel.Calculate(xi, zj))));
    }

    /// <summary>
    /// Gets the inducing point covariance matrix k(Z, Z).
    /// </summary>
    /// <returns>The M × M covariance matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns the precomputed k(Z, Z) matrix.
    /// Useful for sparse GP computations that need this directly.
    /// </para>
    /// </remarks>
    public Matrix<T> GetInducingPointCovariance()
    {
        if (!_matricesValid)
        {
            PrecomputeMatrices();
        }

        if (_Kzz is null)
        {
            throw new InvalidOperationException("Matrices not properly initialized.");
        }

        // Return a copy
        var copy = new Matrix<T>(_Kzz.Rows, _Kzz.Columns);
        for (int i = 0; i < _Kzz.Rows; i++)
        {
            for (int j = 0; j < _Kzz.Columns; j++)
            {
                copy[i, j] = _Kzz[i, j];
            }
        }
        return copy;
    }

    /// <summary>
    /// Gets the Cholesky factor of k(Z, Z).
    /// </summary>
    /// <returns>The lower triangular Cholesky factor L where LL^T = k(Z, Z).</returns>
    public Matrix<T> GetCholeskyFactor()
    {
        if (!_matricesValid)
        {
            PrecomputeMatrices();
        }

        if (_Lzz is null)
        {
            throw new InvalidOperationException("Matrices not properly initialized.");
        }

        // Return a copy
        var copy = new Matrix<T>(_Lzz.Rows, _Lzz.Columns);
        for (int i = 0; i < _Lzz.Rows; i++)
        {
            for (int j = 0; j < _Lzz.Columns; j++)
            {
                copy[i, j] = _Lzz[i, j];
            }
        }
        return copy;
    }

    /// <summary>
    /// Estimates the approximation quality at a set of test points.
    /// </summary>
    /// <param name="X">Test points to evaluate.</param>
    /// <returns>Mean relative error between exact and approximate kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to check if your inducing points provide
    /// a good approximation. Lower values mean better approximation.
    ///
    /// If the error is too high, consider:
    /// - Adding more inducing points
    /// - Better initializing their locations (e.g., via k-means)
    /// - Optimizing inducing point locations during training
    /// </para>
    /// </remarks>
    public T EstimateApproximationError(Matrix<T> X)
    {
        int n = Math.Min(X.Rows, 100); // Sample at most 100 points
        int count = (int)Math.Min((long)n * n, 500);
        if (count == 0)
            return _numOps.Zero;

        // Draw the random pairs first (same stream order as one pair per trial).
        var rand = RandomHelper.CreateSecureRandom();
        var pairs = new (int I, int J)[count];
        for (int trial = 0; trial < count; trial++)
        {
            int i = rand.Next(X.Rows);
            int j = rand.Next(X.Rows);
            pairs[trial] = (i, j);
        }

        var exact = new Vector<T>(pairs.Select(p => _baseKernel.Calculate(X.GetRow(p.I), X.GetRow(p.J))));
        var approx = new Vector<T>(pairs.Select(p => Calculate(X.GetRow(p.I), X.GetRow(p.J))));

        // Mean relative error |exact - approx| / max(|exact|, 1e-10)
        var engine = AiDotNetEngine.Current;
        var exactTensor = exact.AsTensor();
        var denominator = engine.TensorClampMin(engine.TensorAbs(exactTensor), _numOps.FromDouble(1e-10));
        var errors = engine.TensorDivide(engine.TensorAbs(engine.TensorSubtract(exactTensor, approx.AsTensor())), denominator);
        return engine.TensorMean(errors);
    }

    #region Helper Methods

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
                {
                    sum += _numOps.ToDouble(L[i, k]) * _numOps.ToDouble(L[j, k]);
                }

                if (i == j)
                {
                    double val = _numOps.ToDouble(A[i, i]) - sum;
                    L[i, j] = _numOps.FromDouble(Math.Sqrt(Math.Max(val, 1e-10)));
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

    /// <summary>
    /// Solves Lx = b for x.
    /// </summary>
    private Vector<T> SolveLowerTriangular(Matrix<T> L, Vector<T> b)
    {
        int n = b.Length;
        var solution = LinalgDispatch<T>.SolveTriangular(Tensor<T>.FromMatrix(L), b.AsTensor().Reshape(new[] { n, 1 }), upper: false);
        return solution.Reshape(new[] { n }).ToVector();
    }

    #endregion
}
