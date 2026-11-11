using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.CausalDiscovery;

/// <summary>
/// Abstract base class for causal discovery algorithms with shared statistical utilities.
/// </summary>
/// <remarks>
/// <para>
/// Provides common functionality needed by most causal discovery algorithms:
/// conditional independence testing, BIC scoring, partial correlation computation,
/// covariance estimation, and OLS regression.
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class contains the "toolbox" of statistical tests
/// and helper methods that all causal discovery algorithms share. Each specific algorithm
/// (like NOTEARS or PC) extends this class and adds its own unique logic.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class CausalDiscoveryBase<T> : ICausalDiscoveryAlgorithm<T>
{
    /// <summary>
    /// Numeric operations helper for generic math on type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the global execution engine for accelerated vector/matrix operations.
    /// Subclasses should prefer using specific helper methods (e.g., DotProduct) when available.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <inheritdoc/>
    public abstract CausalDiscoveryCategory Category { get; }

    /// <inheritdoc/>
    public virtual bool SupportsLatentConfounders => false;

    /// <inheritdoc/>
    public virtual bool SupportsTimeSeries => false;

    /// <inheritdoc/>
    public virtual bool SupportsNonlinear => false;

    /// <inheritdoc/>
    public virtual bool SupportsMixedData => false;

    /// <inheritdoc/>
    public CausalGraph<T> DiscoverStructure(Matrix<T> data, string[]? featureNames = null)
    {
        ValidateInput(data);
        string[] names = featureNames ?? GenerateDefaultNames(data.Columns);

        if (names.Length != data.Columns)
        {
            throw new ArgumentException(
                $"Feature names length ({names.Length}) must match number of columns ({data.Columns}).",
                nameof(featureNames));
        }

        Matrix<T> adjacency = DiscoverStructureCore(data);
        return new CausalGraph<T>(adjacency, names);
    }

    /// <inheritdoc/>
    public virtual CausalGraph<T> DiscoverStructure(Matrix<T> data, Vector<T> target, string[]? featureNames = null)
    {
        // Default: append target as last column and run standard discovery
        int n = data.Rows;
        int d = data.Columns;

        if (target.Length != n)
            throw new ArgumentException($"Target length ({target.Length}) must match data row count ({n}).", nameof(target));
        if (featureNames != null && featureNames.Length != d)
            throw new ArgumentException($"Feature names length ({featureNames.Length}) must match data column count ({d}).", nameof(featureNames));
        var combined = Engine.TensorConcatenate(
            new[] { Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target).Reshape(new[] { n, 1 }) }, 1).ToMatrix();

        string[] names = featureNames ?? GenerateDefaultNames(d);
        string[] allNames = [.. names, "target"];

        return DiscoverStructure(combined, allNames);
    }

    /// <summary>
    /// Core implementation that each algorithm must provide.
    /// </summary>
    /// <param name="data">Validated data matrix [n_samples, n_variables].</param>
    /// <returns>Weighted adjacency matrix [d x d].</returns>
    protected abstract Matrix<T> DiscoverStructureCore(Matrix<T> data);

    /// <summary>
    /// Validates that the input data matrix is suitable for causal discovery.
    /// </summary>
    protected virtual void ValidateInput(Matrix<T> data)
    {
        if (data.Rows < 2)
        {
            throw new ArgumentException("Data must have at least 2 samples.", nameof(data));
        }

        if (data.Columns < 2)
        {
            throw new ArgumentException("Data must have at least 2 variables.", nameof(data));
        }
    }

    /// <summary>
    /// The standard deviation below which a column counts as constant and carries no information.
    /// </summary>
    /// <remarks>
    /// Dividing by a standard deviation this small would amplify floating-point noise into a
    /// unit-variance column, and every downstream correlation would then read that noise as signal.
    /// </remarks>
    protected const double ConstantColumnStdDevTolerance = 1e-12;

    /// <summary>
    /// Returns a copy of <paramref name="data"/> with every column standardized to zero mean and
    /// unit variance, so a downstream covariance is the scale-invariant correlation matrix.
    /// </summary>
    /// <param name="data">Data matrix [n x d].</param>
    /// <returns>A new [n x d] matrix; <paramref name="data"/> is not modified.</returns>
    /// <remarks>
    /// <para>
    /// A column whose standard deviation falls below <see cref="ConstantColumnStdDevTolerance"/> is
    /// written as zeros rather than divided through: it holds no information, and rescaling its
    /// residual noise to unit variance would manufacture correlations that are not in the data.
    /// </para>
    /// <para>
    /// Population standard deviation (divide by n, not n-1), because the result feeds covariance
    /// estimates that use the same convention.
    /// </para>
    /// </remarks>
    protected Matrix<T> StandardizeColumns(Matrix<T> data)
    {
        var x = Tensor<T>.FromMatrix(data);
        var centered = CenterColumns(x);
        var standardDeviation = Engine.TensorSqrt(
            Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 0 }, true));

        // Constant columns are written as zeros: their scale factor is 0 instead of 1/std.
        var isConstant = Engine.TensorLessThan(standardDeviation, NumOps.FromDouble(ConstantColumnStdDevTolerance));
        var safeDeviation = Engine.TensorWhere(isConstant, Engine.TensorOnesLike(standardDeviation), standardDeviation);
        var scale = Engine.TensorWhere(isConstant, Engine.TensorZerosLike(standardDeviation),
            Engine.TensorReciprocal(safeDeviation));

        return Engine.TensorMultiply(centered, Engine.TensorBroadcastTo(scale, centered.Shape.ToArray())).ToMatrix();
    }

    /// <summary>
    /// Subtracts each column's mean from an [n x d] tensor.
    /// </summary>
    private protected Tensor<T> CenterColumns(Tensor<T> x)
    {
        var mean = Engine.ReduceMean(x, new[] { 0 }, true);
        return Engine.TensorSubtract(x, Engine.TensorBroadcastTo(mean, x.Shape.ToArray()));
    }

    /// <summary>
    /// Computes the sample covariance matrix from data.
    /// </summary>
    /// <param name="data">Data matrix [n x d].</param>
    /// <returns>Covariance matrix [d x d].</returns>
    protected Matrix<T> ComputeCovarianceMatrix(Matrix<T> data)
    {
        int n = data.Rows;

        // Compute covariance: (1/n) * (X - mean)^T * (X - mean)
        var centered = CenterColumns(Tensor<T>.FromMatrix(data));
        T nInv = NumOps.Divide(NumOps.One, NumOps.FromDouble(n));
        var gram = Engine.TensorMatMul(Engine.TensorTranspose(centered), centered);

        // Mirror the upper triangle so the result is exactly symmetric.
        var upper = Engine.TensorTriu(gram, 0);
        var symmetric = Engine.TensorAdd(upper, Engine.TensorTranspose(Engine.TensorTriu(gram, 1)));
        return Engine.TensorMultiplyScalar(symmetric, nInv).ToMatrix();
    }

    /// <summary>
    /// Computes the partial correlation between variables i and j given a conditioning set.
    /// </summary>
    /// <param name="correlationMatrix">The correlation matrix [d x d].</param>
    /// <param name="i">First variable index.</param>
    /// <param name="j">Second variable index.</param>
    /// <param name="conditioningSet">Indices of conditioning variables.</param>
    /// <returns>The partial correlation coefficient.</returns>
    protected double ComputePartialCorrelation(
        Matrix<T> correlationMatrix, int i, int j, int[] conditioningSet)
    {
        if (conditioningSet.Length == 0)
        {
            return NumOps.ToDouble(correlationMatrix[i, j]);
        }

        // Build the submatrix for [i, j, condSet] and compute via matrix inversion
        int[] indices = [i, j, .. conditioningSet];
        var subMatrix = SelectSubmatrix(correlationMatrix, indices);

        // Invert the submatrix
        var precision = InvertSmallMatrix(subMatrix);
        if (precision == null)
        {
            return double.NaN; // Singular matrix — caller should handle
        }

        // Partial correlation = -precision[0,1] / sqrt(precision[0,0] * precision[1,1])
        double p00 = NumOps.ToDouble(precision[0, 0]);
        double p11 = NumOps.ToDouble(precision[1, 1]);
        double denom = Math.Sqrt(Math.Abs(p00 * p11));
        if (denom < 1e-15)
        {
            return 0.0;
        }

        return -NumOps.ToDouble(precision[0, 1]) / denom;
    }

    /// <summary>
    /// Tests conditional independence between variables i and j given a conditioning set.
    /// Uses Fisher's z-transform of the partial correlation.
    /// </summary>
    /// <param name="correlationMatrix">The correlation matrix.</param>
    /// <param name="i">First variable.</param>
    /// <param name="j">Second variable.</param>
    /// <param name="conditioningSet">Conditioning set indices.</param>
    /// <param name="nSamples">Number of data samples.</param>
    /// <param name="alpha">Significance level (default 0.05).</param>
    /// <returns>True if i and j are conditionally independent given the conditioning set.</returns>
    protected bool TestConditionalIndependence(
        Matrix<T> correlationMatrix, int i, int j, int[] conditioningSet,
        int nSamples, double alpha = 0.05)
    {
        double partialCorr = ComputePartialCorrelation(correlationMatrix, i, j, conditioningSet);

        // NaN means singular correlation sub-matrix — conservatively assume independence
        if (double.IsNaN(partialCorr))
            return true;

        // Clamp to avoid log(0) or log(negative) in Fisher's z-transform
        partialCorr = Math.Max(-1 + 1e-15, Math.Min(1 - 1e-15, partialCorr));

        // Fisher's z-transform
        double z = 0.5 * Math.Log((1 + partialCorr) / (1 - partialCorr));

        // Guard: degrees of freedom must be positive for the test to be valid
        int dof = nSamples - conditioningSet.Length - 3;
        if (dof <= 0)
            return true; // insufficient data — cannot reject independence

        double testStat = Math.Abs(z) * Math.Sqrt(dof);

        // Compare with standard normal critical value
        double criticalValue = NormalQuantile(1 - alpha / 2);
        return testStat < criticalValue;
    }

    /// <summary>
    /// Computes the BIC (Bayesian Information Criterion) score for a variable given its parents.
    /// </summary>
    /// <param name="data">Data matrix.</param>
    /// <param name="variable">Target variable index.</param>
    /// <param name="parents">Parent variable indices.</param>
    /// <returns>BIC score (lower is better).</returns>
    protected double ComputeBICScore(Matrix<T> data, int variable, int[] parents)
    {
        int n = data.Rows;
        int k = parents.Length;

        // Use consistent BIC formula: BIC = n * log(residual_variance) + k * log(n)
        // where k = number of parameters (parents + intercept for with-parents case).
        // The n*log(2*pi) + n terms cancel when comparing models.
        if (k == 0)
        {
            // No parents: 1 parameter (mean), residual variance = sample variance
            double variance = ComputeVariance(data, variable);
            return n * Math.Log(Math.Max(variance, 1e-15)) + 1.0 * Math.Log(n);
        }

        // With parents: k+1 parameters (k regression coefficients + intercept)
        double residualVariance = ComputeResidualVariance(data, variable, parents);
        return n * Math.Log(Math.Max(residualVariance, 1e-15)) + (k + 1) * Math.Log(n);
    }

    /// <summary>
    /// Fits OLS regression and returns the residual variance.
    /// </summary>
    protected double ComputeResidualVariance(Matrix<T> data, int target, int[] predictors)
    {
        int n = data.Rows;
        int dim = predictors.Length + 1; // +1 for intercept

        // Design matrix [1 | data[:, predictors]] and target column.
        var source = Tensor<T>.FromMatrix(data);
        var y = SelectColumns(source, new[] { target });
        var predictorColumns = SelectColumns(source, predictors);
        var X = Engine.TensorConcatenate(
            new[] { Engine.TensorOnesLike(y), predictorColumns }, 1);
        var Xt = Engine.TensorTranspose(X);

        // Normal equations with ridge regularization: (XtX + ridge*I) * beta = Xty
        var XtX = Engine.TensorAdd(Engine.TensorMatMul(Xt, X),
            Engine.TensorMultiplyScalar(Engine.TensorEye<T>(dim), NumOps.FromDouble(1e-4)));
        var Xty = Engine.TensorMatMul(Xt, y).Reshape(new[] { dim }).ToVector();

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(XtX.ToMatrix(), Xty, MatrixDecompositionType.Lu);

        // Residual variance
        var prediction = Engine.TensorMatMul(X, Tensor<T>.FromVector(beta).Reshape(new[] { dim, 1 }));
        var residual = Engine.TensorSubtract(y, prediction);
        T rss = Engine.TensorSum(Engine.TensorSquare(residual));
        return NumOps.ToDouble(rss) / n;
    }

    /// <summary>
    /// Gathers the given columns of an [n x d] tensor into an [n x k] tensor.
    /// </summary>
    private protected Tensor<T> SelectColumns(Tensor<T> source, int[] columns)
        => Engine.TensorIndexSelect(source, new Tensor<int>((int[])columns.Clone(), new[] { columns.Length }), 1);

    /// <summary>
    /// Gathers the rows and columns <paramref name="indices"/> of a matrix into a square submatrix.
    /// </summary>
    private protected Matrix<T> SelectSubmatrix(Matrix<T> matrix, int[] indices)
    {
        var index = new Tensor<int>((int[])indices.Clone(), new[] { indices.Length });
        var rows = Engine.TensorIndexSelect(Tensor<T>.FromMatrix(matrix), index, 0);
        return Engine.TensorIndexSelect(rows, index, 1).ToMatrix();
    }

    /// <summary>
    /// Computes the variance of a single column.
    /// </summary>
    protected double ComputeVariance(Matrix<T> data, int column)
    {
        var centered = CenterColumns(SelectColumns(Tensor<T>.FromMatrix(data), new[] { column }));
        return NumOps.ToDouble(Engine.TensorMean(Engine.TensorSquare(centered)));
    }

    /// <summary>
    /// Converts a covariance matrix to a correlation matrix.
    /// </summary>
    protected Matrix<T> CovarianceToCorrelation(Matrix<T> covariance)
    {
        var cov = Tensor<T>.FromMatrix(covariance);
        var variances = Engine.TensorDiagonal(cov);
        var varianceProducts = Engine.TensorOuter(variances, variances);
        var denom = Engine.TensorSqrt(Engine.TensorMax(varianceProducts, NumOps.FromDouble(1e-30)));
        return Engine.TensorDivide(cov, denom).ToMatrix();
    }

    /// <summary>
    /// Applies a threshold to the adjacency matrix, zeroing out entries below the threshold.
    /// </summary>
    protected Matrix<T> ThresholdMatrix(Matrix<T> matrix, double threshold)
    {
        var values = Tensor<T>.FromMatrix(matrix);
        var keep = Engine.TensorGreaterEqual(Engine.TensorAbs(values),
            Engine.TensorFullLike(values, NumOps.FromDouble(threshold)));
        return Engine.TensorWhere(keep, values, Engine.TensorZerosLike(values)).ToMatrix();
    }

    /// <summary>
    /// Generates default variable names like "X0", "X1", "X2", etc.
    /// </summary>
    protected static string[] GenerateDefaultNames(int count)
    {
        string[] names = new string[count];
        for (int i = 0; i < count; i++)
        {
            names[i] = $"X{i}";
        }

        return names;
    }

    /// <summary>
    /// Standard normal quantile (inverse CDF) approximation using rational approximation.
    /// </summary>
    protected static double NormalQuantile(double p)
    {
        // Abramowitz and Stegun approximation 26.2.23
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;
        if (Math.Abs(p - 0.5) < 1e-15) return 0.0;

        double t;
        if (p < 0.5)
        {
            t = Math.Sqrt(-2.0 * Math.Log(p));
        }
        else
        {
            t = Math.Sqrt(-2.0 * Math.Log(1.0 - p));
        }

        // Coefficients for rational approximation
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        double result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);

        return p < 0.5 ? -result : result;
    }

    /// <summary>
    /// Accelerated matrix-matrix multiplication: C = A * B using Engine when available.
    /// </summary>
    /// <param name="a">Left matrix [m x k].</param>
    /// <param name="b">Right matrix [k x n].</param>
    /// <returns>Result matrix [m x n].</returns>
    protected Matrix<T> MatMul(Matrix<T> a, Matrix<T> b)
    {
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException(
                $"Matrix dimension mismatch in MatMul: A is [{a.Rows}x{a.Columns}] but B is [{b.Rows}x{b.Columns}]. " +
                $"A.Columns ({a.Columns}) must equal B.Rows ({b.Rows}).");
        }

        return Engine.MatrixMultiply(a, b);
    }

    /// <summary>
    /// Accelerated dot product between a row of matrix A and a row of matrix B.
    /// </summary>
    protected T RowDotProduct(Matrix<T> a, int rowA, Matrix<T> b, int rowB, int length)
    {
        var va = Engine.GetRow(a, rowA).GetSubVector(0, length);
        var vb = Engine.GetRow(b, rowB).GetSubVector(0, length);
        return Engine.DotProduct(va, vb);
    }

    /// <summary>
    /// Accelerated element-wise addition of two vectors.
    /// </summary>
    protected Vector<T> VectorAdd(Vector<T> a, Vector<T> b)
    {
        return (Vector<T>)Engine.Add<T>(a, b);
    }

    /// <summary>
    /// Accelerated element-wise subtraction: a - b.
    /// </summary>
    protected Vector<T> VectorSubtract(Vector<T> a, Vector<T> b)
    {
        return (Vector<T>)Engine.Subtract<T>(a, b);
    }

    /// <summary>
    /// Accelerated scalar multiplication: a * scalar.
    /// </summary>
    protected Vector<T> VectorScale(Vector<T> a, T scalar)
    {
        return (Vector<T>)Engine.Multiply(a, scalar);
    }

    /// <summary>
    /// Extract a row from a matrix as a Vector for accelerated operations.
    /// </summary>
    protected Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        return Engine.GetRow(matrix, row);
    }

    /// <summary>
    /// Extract a column from a matrix as a Vector for accelerated operations.
    /// </summary>
    protected Vector<T> GetColumn(Matrix<T> matrix, int col)
    {
        return Engine.GetColumn(matrix, col);
    }

    /// <summary>
    /// Accelerated matrix-vector multiplication: result = M * v.
    /// </summary>
    protected Vector<T> MatVecMul(Matrix<T> m, Vector<T> v)
    {
        if (m.Columns != v.Length)
        {
            throw new ArgumentException(
                $"Dimension mismatch in MatVecMul: matrix is [{m.Rows}x{m.Columns}] but vector length is {v.Length}. " +
                $"Matrix columns ({m.Columns}) must equal vector length ({v.Length}).");
        }

        return Engine.MatrixVectorMultiply(m, v);
    }

    /// <summary>
    /// Computes the matrix exponential via Taylor series: exp(M) = sum_{k=0}^{terms} M^k / k!
    /// </summary>
    /// <param name="M">The input square matrix.</param>
    /// <param name="d">Dimension of the matrix.</param>
    /// <param name="terms">Number of Taylor series terms (default 10).</param>
    /// <returns>The matrix exponential approximation.</returns>
    protected Matrix<T> MatrixExponentialTaylor(Matrix<T> M, int d, int terms = 10)
    {
        var m = Tensor<T>.FromMatrix(M);
        var result = Engine.TensorEye<T>(d);
        var power = Engine.TensorEye<T>(d);

        // runningFactorial holds k! as a [1, 1] tensor, broadcast over the power matrix.
        var one = Engine.TensorOnesLike(Engine.TensorSlice(result, new[] { 0, 0 }, new[] { 1, 1 }));
        var counter = Engine.TensorZerosLike(one);
        var runningFactorial = one;
        for (int k = 1; k <= terms; k++)
        {
            power = Engine.TensorMatMul(power, m);
            counter = Engine.TensorAdd(counter, one);
            runningFactorial = Engine.TensorMultiply(runningFactorial, counter);
            result = Engine.TensorAdd(result,
                Engine.TensorDivide(power, Engine.TensorBroadcastTo(runningFactorial, new[] { d, d })));
        }

        return result.ToMatrix();
    }

    /// <summary>
    /// Solves a small linear system Ax = b using Gaussian elimination with partial pivoting.
    /// This shared utility replaces duplicate implementations across causal discovery algorithms.
    /// </summary>
    /// <param name="A">Coefficient matrix (p x p).</param>
    /// <param name="b">Right-hand side vector (length p).</param>
    /// <param name="p">System dimension.</param>
    /// <returns>Solution vector x, with zero for near-singular pivots.</returns>
    protected static double[] SolveSmallSystem(double[,] A, double[] b, int p)
    {
        if (A.GetLength(0) < p || A.GetLength(1) < p)
            throw new ArgumentException($"Matrix A must be at least {p}x{p}, got {A.GetLength(0)}x{A.GetLength(1)}.");
        if (b.Length < p)
            throw new ArgumentException($"Vector b must have at least {p} elements, got {b.Length}.");

        const double pivotTolerance = 1e-15;
        var aug = new double[p, p + 1];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++) aug[i, j] = A[i, j];
            aug[i, p] = b[i];
        }
        for (int col = 0; col < p; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < p; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col])) maxRow = row;
            if (maxRow != col)
                for (int j = col; j <= p; j++)
                    (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);
            double pivot = aug[col, col];
            if (Math.Abs(pivot) < pivotTolerance) continue;
            for (int row = col + 1; row < p; row++)
            {
                double factor = aug[row, col] / pivot;
                for (int j = col; j <= p; j++) aug[row, j] -= factor * aug[col, j];
            }
        }
        var x = new double[p];
        for (int row = p - 1; row >= 0; row--)
        {
            double sum = aug[row, p];
            for (int j = row + 1; j < p; j++) sum -= aug[row, j] * x[j];
            double diag = aug[row, row];
            if (Math.Abs(diag) <= pivotTolerance)
                return new double[p]; // Singular — return zero coefficients (caller handles)
            x[row] = sum / diag;
        }
        return x;
    }

    /// <summary>
    /// Inverts a small matrix using Gauss-Jordan elimination. Returns null if singular.
    /// </summary>
    private Matrix<T>? InvertSmallMatrix(Matrix<T> matrix)
    {
        // Partial-pivoting LU: its U diagonal holds exactly the pivots Gauss-Jordan elimination
        // with partial pivoting would divide by, so the singularity test is unchanged.
        var a = Tensor<T>.FromMatrix(matrix);
        var (_, _, u) = LinalgDispatch<T>.LU(a);
        var smallPivots = Engine.TensorLessThan(Engine.TensorAbs(Engine.TensorDiagonal(u)), NumOps.FromDouble(1e-12));
        if (NumOps.GreaterThan(Engine.TensorSum(smallPivots), NumOps.Zero))
        {
            return null; // Singular
        }

        return LinalgDispatch<T>.Inv(a).ToMatrix();
    }

}
