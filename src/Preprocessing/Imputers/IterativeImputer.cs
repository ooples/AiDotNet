using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Imputers;

/// <summary>
/// Iterative imputer using the MICE algorithm (Multiple Imputation by Chained Equations).
/// </summary>
/// <remarks>
/// <para>
/// IterativeImputer imputes missing values by modeling each feature with missing values
/// as a function of other features, iterating multiple times until convergence.
/// </para>
/// <para>
/// The algorithm:
/// 1. Initial imputation (mean/median for each feature)
/// 2. For each feature with missing values:
///    - Train a regression model using other features as predictors
///    - Predict missing values using the trained model
/// 3. Repeat step 2 for multiple iterations until convergence
/// </para>
/// <para><b>For Beginners:</b> MICE creates multiple "guesses" for missing values by
/// learning relationships between features. If taller people tend to be heavier,
/// MICE can use height to predict missing weight values more accurately than
/// simply using the average weight.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class IterativeImputer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private readonly IterativeImputerEstimator _estimator;
    private readonly IterativeImputerInitialStrategy _initialStrategy;
    private readonly int _randomState;

    // Fitted parameters
    private double[]? _initialValues;
    private double[,]? _coefficients;
    private double[]? _intercepts;
    private bool[]? _hasMissing;
    private int _nFeatures;

    /// <summary>
    /// Gets the maximum number of iterations.
    /// </summary>
    public int MaxIterations => _maxIterations;

    /// <summary>
    /// Gets the convergence tolerance.
    /// </summary>
    public double Tolerance => _tolerance;

    /// <summary>
    /// Gets the estimator type used for imputation.
    /// </summary>
    public IterativeImputerEstimator Estimator => _estimator;

    /// <summary>
    /// Gets the initial imputation strategy.
    /// </summary>
    public IterativeImputerInitialStrategy InitialStrategy => _initialStrategy;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="IterativeImputer{T}"/>.
    /// </summary>
    /// <param name="maxIterations">Maximum number of imputation iterations. Defaults to 10.</param>
    /// <param name="tolerance">Convergence tolerance. Defaults to 1e-3.</param>
    /// <param name="estimator">The estimator to use for imputation. Defaults to BayesianRidge.</param>
    /// <param name="initialStrategy">Initial imputation strategy. Defaults to Mean.</param>
    /// <param name="randomState">Random seed for reproducibility. Defaults to 0.</param>
    /// <param name="columnIndices">The column indices to impute, or null for all columns.</param>
    public IterativeImputer(
        int maxIterations = 10,
        double tolerance = 1e-3,
        IterativeImputerEstimator estimator = IterativeImputerEstimator.BayesianRidge,
        IterativeImputerInitialStrategy initialStrategy = IterativeImputerInitialStrategy.Mean,
        int randomState = 0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxIterations < 1)
        {
            throw new ArgumentException("Maximum iterations must be at least 1.", nameof(maxIterations));
        }

        if (tolerance < 0)
        {
            throw new ArgumentException("Tolerance must be non-negative.", nameof(tolerance));
        }

        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _estimator = estimator;
        _initialStrategy = initialStrategy;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the imputer by learning the relationships between features.
    /// </summary>
    /// <param name="data">The training data with missing values (NaN).</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeatures = data.Columns;
        int nSamples = data.Rows;

        // Convert to double and identify missing values
        var X = new double[nSamples, _nFeatures];
        var missingMask = new bool[nSamples, _nFeatures];
        _hasMissing = new bool[_nFeatures];

        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < _nFeatures; j++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                X[i, j] = val;
                if (double.IsNaN(val))
                {
                    missingMask[i, j] = true;
                    _hasMissing[j] = true;
                }
            }
        }

        // Initial imputation
        _initialValues = ComputeInitialValues(X, missingMask);

        // Apply initial imputation
        var Xt = (double[,])X.Clone();
        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < _nFeatures; j++)
            {
                if (missingMask[i, j])
                {
                    Xt[i, j] = _initialValues[j];
                }
            }
        }

        // Store coefficients for each feature
        _coefficients = new double[_nFeatures, _nFeatures];
        _intercepts = new double[_nFeatures];

        // Iterative imputation
        for (int iteration = 0; iteration < _maxIterations; iteration++)
        {
            var XtOld = (double[,])Xt.Clone();

            for (int targetCol = 0; targetCol < _nFeatures; targetCol++)
            {
                if (!_hasMissing[targetCol]) continue;

                // Build training set from rows without missing values in target
                var trainRows = new List<int>();
                for (int i = 0; i < nSamples; i++)
                {
                    if (!missingMask[i, targetCol])
                    {
                        trainRows.Add(i);
                    }
                }

                if (trainRows.Count < 2) continue;

                // Fit a simple linear model: y = X * w + b
                var (weights, intercept) = FitLinearModel(Xt, trainRows, targetCol);

                // Store coefficients
                for (int j = 0; j < _nFeatures; j++)
                {
                    _coefficients[targetCol, j] = weights[j];
                }
                _intercepts[targetCol] = intercept;

                // Predict missing values
                for (int i = 0; i < nSamples; i++)
                {
                    if (missingMask[i, targetCol])
                    {
                        double pred = intercept;
                        for (int j = 0; j < _nFeatures; j++)
                        {
                            if (j != targetCol)
                            {
                                pred += weights[j] * Xt[i, j];
                            }
                        }
                        Xt[i, targetCol] = pred;
                    }
                }
            }

            // Check convergence
            double maxChange = 0;
            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < _nFeatures; j++)
                {
                    if (missingMask[i, j])
                    {
                        double change = Math.Abs(Xt[i, j] - XtOld[i, j]);
                        maxChange = Math.Max(maxChange, change);
                    }
                }
            }

            if (maxChange < _tolerance)
            {
                break;
            }
        }
    }

    private double[] ComputeInitialValues(double[,] X, bool[,] missingMask)
    {
        int nSamples = X.GetLength(0);
        int nFeatures = X.GetLength(1);
        var values = new double[nFeatures];

        for (int j = 0; j < nFeatures; j++)
        {
            var nonMissing = new List<double>();
            for (int i = 0; i < nSamples; i++)
            {
                if (!missingMask[i, j])
                {
                    nonMissing.Add(X[i, j]);
                }
            }

            if (nonMissing.Count == 0)
            {
                values[j] = 0;
                continue;
            }

            switch (_initialStrategy)
            {
                case IterativeImputerInitialStrategy.Mean:
                    values[j] = nonMissing.Average();
                    break;
                case IterativeImputerInitialStrategy.Median:
                    nonMissing.Sort();
                    int mid = nonMissing.Count / 2;
                    values[j] = nonMissing.Count % 2 == 0
                        ? (nonMissing[mid - 1] + nonMissing[mid]) / 2
                        : nonMissing[mid];
                    break;
                case IterativeImputerInitialStrategy.MostFrequent:
                    values[j] = nonMissing.GroupBy(v => Math.Round(v, 6))
                        .OrderByDescending(g => g.Count())
                        .First().Key;
                    break;
                case IterativeImputerInitialStrategy.Constant:
                    values[j] = 0;
                    break;
            }
        }

        return values;
    }

    private (double[] Weights, double Intercept) FitLinearModel(
        double[,] X, List<int> trainRows, int targetCol)
    {
        int nFeatures = X.GetLength(1);
        int nTrain = trainRows.Count;

        // Extract target and features (excluding target column)
        var y = new double[nTrain];
        var Xf = new double[nTrain, nFeatures - 1];

        for (int i = 0; i < nTrain; i++)
        {
            int row = trainRows[i];
            y[i] = X[row, targetCol];

            int colIdx = 0;
            for (int j = 0; j < nFeatures; j++)
            {
                if (j != targetCol)
                {
                    Xf[i, colIdx++] = X[row, j];
                }
            }
        }

        // Simple ridge regression with regularization
        double lambda = 1e-3;
        int p = nFeatures - 1;

        // Center the data
        double yMean = y.Average();
        var xMeans = new double[p];
        for (int j = 0; j < p; j++)
        {
            double sum = 0;
            for (int i = 0; i < nTrain; i++)
            {
                sum += Xf[i, j];
            }
            xMeans[j] = sum / nTrain;
        }

        // Center
        for (int i = 0; i < nTrain; i++)
        {
            y[i] -= yMean;
            for (int j = 0; j < p; j++)
            {
                Xf[i, j] -= xMeans[j];
            }
        }

        // Compute X^T * X + lambda * I
        var XTX = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < nTrain; k++)
                {
                    sum += Xf[k, i] * Xf[k, j];
                }
                XTX[i, j] = sum;
                if (i == j)
                {
                    XTX[i, j] += lambda;
                }
            }
        }

        // Compute X^T * y
        var XTy = new double[p];
        for (int j = 0; j < p; j++)
        {
            double sum = 0;
            for (int i = 0; i < nTrain; i++)
            {
                sum += Xf[i, j] * y[i];
            }
            XTy[j] = sum;
        }

        // Solve using Cholesky or simple iteration
        var weights = SolveLinearSystem(XTX, XTy, p);

        // Expand weights back to full size
        var fullWeights = new double[nFeatures];
        int idx = 0;
        for (int j = 0; j < nFeatures; j++)
        {
            if (j != targetCol)
            {
                fullWeights[j] = weights[idx++];
            }
        }

        // Compute intercept
        double intercept = yMean;
        idx = 0;
        for (int j = 0; j < nFeatures; j++)
        {
            if (j != targetCol)
            {
                intercept -= fullWeights[j] * xMeans[idx++];
            }
        }

        return (fullWeights, intercept);
    }

    private double[] SolveLinearSystem(double[,] A, double[] b, int n)
    {
        // Simple Gaussian elimination with partial pivoting
        var augmented = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n] = b[i];
        }

        // Forward elimination
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(augmented[row, col]) > Math.Abs(augmented[maxRow, col]))
                {
                    maxRow = row;
                }
            }

            // Swap rows
            for (int j = col; j <= n; j++)
            {
                double temp = augmented[col, j];
                augmented[col, j] = augmented[maxRow, j];
                augmented[maxRow, j] = temp;
            }

            // Eliminate
            if (Math.Abs(augmented[col, col]) > 1e-10)
            {
                for (int row = col + 1; row < n; row++)
                {
                    double factor = augmented[row, col] / augmented[col, col];
                    for (int j = col; j <= n; j++)
                    {
                        augmented[row, j] -= factor * augmented[col, j];
                    }
                }
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = augmented[i, n];
            for (int j = i + 1; j < n; j++)
            {
                x[i] -= augmented[i, j] * x[j];
            }
            if (Math.Abs(augmented[i, i]) > 1e-10)
            {
                x[i] /= augmented[i, i];
            }
        }

        return x;
    }

    /// <summary>
    /// Transforms the data by imputing missing values.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with missing values imputed.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_initialValues is null || _coefficients is null || _intercepts is null)
        {
            throw new InvalidOperationException("IterativeImputer has not been fitted.");
        }

        int nSamples = data.Rows;
        int nFeatures = data.Columns;
        var result = new T[nSamples, nFeatures];

        // Convert to double
        var X = new double[nSamples, nFeatures];
        var missingMask = new bool[nSamples, nFeatures];

        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                X[i, j] = val;
                if (double.IsNaN(val))
                {
                    missingMask[i, j] = true;
                    X[i, j] = _initialValues[j];
                }
            }
        }

        // Apply learned imputations
        for (int iteration = 0; iteration < _maxIterations; iteration++)
        {
            var XOld = (double[,])X.Clone();

            for (int targetCol = 0; targetCol < nFeatures; targetCol++)
            {
                if (_hasMissing is null || !_hasMissing[targetCol]) continue;

                for (int i = 0; i < nSamples; i++)
                {
                    if (missingMask[i, targetCol])
                    {
                        double pred = _intercepts[targetCol];
                        for (int j = 0; j < nFeatures; j++)
                        {
                            if (j != targetCol)
                            {
                                pred += _coefficients[targetCol, j] * X[i, j];
                            }
                        }
                        X[i, targetCol] = pred;
                    }
                }
            }

            // Check convergence
            double maxChange = 0;
            for (int i = 0; i < nSamples; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    if (missingMask[i, j])
                    {
                        maxChange = Math.Max(maxChange, Math.Abs(X[i, j] - XOld[i, j]));
                    }
                }
            }

            if (maxChange < _tolerance) break;
        }

        // Convert back
        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                result[i, j] = NumOps.FromDouble(X[i, j]);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("IterativeImputer does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}

/// <summary>
/// Specifies the estimator to use for iterative imputation.
/// </summary>
public enum IterativeImputerEstimator
{
    /// <summary>
    /// Bayesian Ridge Regression (regularized linear regression).
    /// </summary>
    BayesianRidge,

    /// <summary>
    /// Simple linear regression (OLS).
    /// </summary>
    LinearRegression,

    /// <summary>
    /// Ridge regression with fixed regularization.
    /// </summary>
    Ridge
}

/// <summary>
/// Specifies the initial imputation strategy.
/// </summary>
public enum IterativeImputerInitialStrategy
{
    /// <summary>
    /// Use mean of non-missing values.
    /// </summary>
    Mean,

    /// <summary>
    /// Use median of non-missing values.
    /// </summary>
    Median,

    /// <summary>
    /// Use most frequent value.
    /// </summary>
    MostFrequent,

    /// <summary>
    /// Use a constant value (zero).
    /// </summary>
    Constant
}
