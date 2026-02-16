using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
        var combined = new Matrix<T>(n, d + 1);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                combined[i, j] = data[i, j];
            }

            combined[i, d] = target[i];
        }

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
    /// Computes the sample covariance matrix from data.
    /// </summary>
    /// <param name="data">Data matrix [n x d].</param>
    /// <returns>Covariance matrix [d x d].</returns>
    protected Matrix<T> ComputeCovarianceMatrix(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        // Compute column means
        var means = new T[d];
        for (int j = 0; j < d; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, data[i, j]);
            }

            means[j] = NumOps.Divide(sum, NumOps.FromDouble(n));
        }

        // Compute covariance: (1/n) * (X - mean)^T * (X - mean)
        var cov = new Matrix<T>(d, d);
        T nInv = NumOps.Divide(NumOps.One, NumOps.FromDouble(n));

        for (int j1 = 0; j1 < d; j1++)
        {
            for (int j2 = j1; j2 < d; j2++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    T diff1 = NumOps.Subtract(data[i, j1], means[j1]);
                    T diff2 = NumOps.Subtract(data[i, j2], means[j2]);
                    sum = NumOps.Add(sum, NumOps.Multiply(diff1, diff2));
                }

                T covVal = NumOps.Multiply(sum, nInv);
                cov[j1, j2] = covVal;
                cov[j2, j1] = covVal; // Symmetric
            }
        }

        return cov;
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
        int size = indices.Length;
        var subMatrix = new Matrix<T>(size, size);

        for (int r = 0; r < size; r++)
        {
            for (int c = 0; c < size; c++)
            {
                subMatrix[r, c] = correlationMatrix[indices[r], indices[c]];
            }
        }

        // Invert the submatrix
        var precision = InvertSmallMatrix(subMatrix);
        if (precision == null)
        {
            return 0.0; // Singular matrix — assume independence
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

        if (k == 0)
        {
            // No parents: BIC = n * log(var(variable))
            double variance = ComputeVariance(data, variable);
            return n * Math.Log(Math.Max(variance, 1e-15)) + Math.Log(n);
        }

        // Fit OLS: variable ~ parents
        double residualVariance = ComputeResidualVariance(data, variable, parents);
        double logLikelihood = -0.5 * n * Math.Log(2 * Math.PI * Math.Max(residualVariance, 1e-15)) - 0.5 * n;
        double bic = -2 * logLikelihood + (k + 1) * Math.Log(n);

        return bic;
    }

    /// <summary>
    /// Fits OLS regression and returns the residual variance.
    /// </summary>
    protected double ComputeResidualVariance(Matrix<T> data, int target, int[] predictors)
    {
        int n = data.Rows;
        int p = predictors.Length;
        int dim = p + 1; // +1 for intercept

        // Build design matrix and target vector using Matrix<T>/Vector<T>
        var X = new Matrix<T>(n, dim);
        var y = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            y[i] = data[i, target];
            X[i, 0] = NumOps.One; // intercept
            for (int j = 0; j < p; j++)
            {
                X[i, j + 1] = data[i, predictors[j]];
            }
        }

        // Normal equations: XtX * beta = Xty
        var XtX = new Matrix<T>(dim, dim);
        var Xty = new Vector<T>(dim);

        for (int i = 0; i < dim; i++)
        {
            for (int j = i; j < dim; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < n; k++)
                    sum = NumOps.Add(sum, NumOps.Multiply(X[k, i], X[k, j]));
                XtX[i, j] = sum;
                XtX[j, i] = sum; // Symmetric
            }

            T sumY = NumOps.Zero;
            for (int k = 0; k < n; k++)
                sumY = NumOps.Add(sumY, NumOps.Multiply(X[k, i], y[k]));
            Xty[i] = sumY;
        }

        // Ridge regularization
        T ridge = NumOps.FromDouble(1e-4);
        for (int i = 0; i < dim; i++)
            XtX[i, i] = NumOps.Add(XtX[i, i], ridge);

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(XtX, Xty, MatrixDecompositionType.Lu);

        // Compute residual variance
        double rss = 0;
        for (int i = 0; i < n; i++)
        {
            T pred = NumOps.Zero;
            for (int j = 0; j < dim; j++)
                pred = NumOps.Add(pred, NumOps.Multiply(beta[j], X[i, j]));
            double residual = NumOps.ToDouble(NumOps.Subtract(y[i], pred));
            rss += residual * residual;
        }

        return rss / n;
    }

    /// <summary>
    /// Computes the variance of a single column.
    /// </summary>
    protected double ComputeVariance(Matrix<T> data, int column)
    {
        int n = data.Rows;
        double mean = 0;
        for (int i = 0; i < n; i++)
        {
            mean += NumOps.ToDouble(data[i, column]);
        }

        mean /= n;

        double variance = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = NumOps.ToDouble(data[i, column]) - mean;
            variance += diff * diff;
        }

        return variance / n;
    }

    /// <summary>
    /// Converts a covariance matrix to a correlation matrix.
    /// </summary>
    protected Matrix<T> CovarianceToCorrelation(Matrix<T> covariance)
    {
        int d = covariance.Rows;
        var correlation = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double covIJ = NumOps.ToDouble(covariance[i, j]);
                double varI = NumOps.ToDouble(covariance[i, i]);
                double varJ = NumOps.ToDouble(covariance[j, j]);
                double denom = Math.Sqrt(Math.Max(varI * varJ, 1e-30));
                correlation[i, j] = NumOps.FromDouble(covIJ / denom);
            }
        }

        return correlation;
    }

    /// <summary>
    /// Applies a threshold to the adjacency matrix, zeroing out entries below the threshold.
    /// </summary>
    protected Matrix<T> ThresholdMatrix(Matrix<T> matrix, double threshold)
    {
        int d = matrix.Rows;
        var result = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double val = NumOps.ToDouble(matrix[i, j]);
                result[i, j] = Math.Abs(val) >= threshold
                    ? matrix[i, j]
                    : NumOps.Zero;
            }
        }

        return result;
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
    /// Inverts a small matrix using Gauss-Jordan elimination. Returns null if singular.
    /// </summary>
    private Matrix<T>? InvertSmallMatrix(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var augmented = new Matrix<T>(n, 2 * n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = matrix[i, j];
            }

            augmented[i, i + n] = NumOps.One;
        }

        for (int col = 0; col < n; col++)
        {
            // Partial pivoting
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(NumOps.ToDouble(augmented[row, col])) >
                    Math.Abs(NumOps.ToDouble(augmented[maxRow, col])))
                {
                    maxRow = row;
                }
            }

            if (Math.Abs(NumOps.ToDouble(augmented[maxRow, col])) < 1e-12)
            {
                return null; // Singular
            }

            // Swap rows
            if (maxRow != col)
            {
                for (int j = 0; j < 2 * n; j++)
                {
                    (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
                }
            }

            // Scale pivot row
            T pivot = augmented[col, col];
            for (int j = 0; j < 2 * n; j++)
            {
                augmented[col, j] = NumOps.Divide(augmented[col, j], pivot);
            }

            // Eliminate other rows
            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    T factor = augmented[row, col];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        augmented[row, j] = NumOps.Subtract(augmented[row, j],
                            NumOps.Multiply(factor, augmented[col, j]));
                    }
                }
            }
        }

        var result = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = augmented[i, j + n];
            }
        }

        return result;
    }

    /// <summary>
    /// Converts a double[,] array to a Matrix&lt;T&gt; using NumOps.FromDouble.
    /// Used by continuous optimization algorithms that require double[,] for internal gradient computation.
    /// </summary>
    protected Matrix<T> DoubleArrayToMatrix(double[,] data)
    {
        int rows = data.GetLength(0), cols = data.GetLength(1);
        var result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = NumOps.FromDouble(data[i, j]);
        return result;
    }
}
