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
        var copy = new Matrix<T>(_inducingPoints.Rows, _inducingPoints.Columns);
        for (int i = 0; i < _inducingPoints.Rows; i++)
        {
            for (int j = 0; j < _inducingPoints.Columns; j++)
            {
                copy[i, j] = _inducingPoints[i, j];
            }
        }
        return copy;
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
        _Kzz = new Matrix<T>(m, m);

        // Compute k(Z, Z)
        for (int i = 0; i < m; i++)
        {
            var zi = GetRow(_inducingPoints, i);
            for (int j = i; j < m; j++)
            {
                var zj = GetRow(_inducingPoints, j);
                T kval = _baseKernel.Calculate(zi, zj);

                if (i == j)
                {
                    double kd = _numOps.ToDouble(kval) + _jitter;
                    _Kzz[i, j] = _numOps.FromDouble(kd);
                }
                else
                {
                    _Kzz[i, j] = kval;
                    _Kzz[j, i] = kval;
                }
            }
        }

        // Compute Cholesky decomposition
        _Lzz = CholeskyDecomposition(_Kzz);
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
        var kx1z = new Vector<T>(_numInducingPoints);
        var kx2z = new Vector<T>(_numInducingPoints);

        for (int i = 0; i < _numInducingPoints; i++)
        {
            var zi = GetRow(_inducingPoints, i);
            kx1z[i] = _baseKernel.Calculate(x1, zi);
            kx2z[i] = _baseKernel.Calculate(x2, zi);
        }

        // Solve L * v1 = k(x1, Z) for v1
        var v1 = SolveLowerTriangular(_Lzz, kx1z);
        var v2 = SolveLowerTriangular(_Lzz, kx2z);

        // Q(x1, x2) = v1^T * v2
        double result = 0;
        for (int i = 0; i < _numInducingPoints; i++)
        {
            result += _numOps.ToDouble(v1[i]) * _numOps.ToDouble(v2[i]);
        }

        return _numOps.FromDouble(result);
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
        var Kxz = new Matrix<T>(n, _numInducingPoints);

        for (int i = 0; i < n; i++)
        {
            var xi = GetRow(X, i);
            for (int j = 0; j < _numInducingPoints; j++)
            {
                var zj = GetRow(_inducingPoints, j);
                Kxz[i, j] = _baseKernel.Calculate(xi, zj);
            }
        }

        return Kxz;
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
    public double EstimateApproximationError(Matrix<T> X)
    {
        int n = Math.Min(X.Rows, 100); // Sample at most 100 points
        double totalError = 0;
        int count = 0;

        var rand = RandomHelper.CreateSecureRandom();

        for (int trial = 0; trial < n * n && count < 500; trial++)
        {
            int i = rand.Next(X.Rows);
            int j = rand.Next(X.Rows);

            var xi = GetRow(X, i);
            var xj = GetRow(X, j);

            double exact = _numOps.ToDouble(_baseKernel.Calculate(xi, xj));
            double approx = _numOps.ToDouble(Calculate(xi, xj));

            double error = Math.Abs(exact - approx) / Math.Max(Math.Abs(exact), 1e-10);
            totalError += error;
            count++;
        }

        return count > 0 ? totalError / count : 0;
    }

    #region Helper Methods

    /// <summary>
    /// Extracts a row from a matrix.
    /// </summary>
    private Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var result = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            result[j] = matrix[row, j];
        }
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
        var x = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < i; j++)
            {
                sum += _numOps.ToDouble(L[i, j]) * _numOps.ToDouble(x[j]);
            }
            x[i] = _numOps.FromDouble((_numOps.ToDouble(b[i]) - sum) / _numOps.ToDouble(L[i, i]));
        }
        return x;
    }

    #endregion
}
