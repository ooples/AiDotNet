using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Recursive Feature Elimination for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// RFE performs feature selection by recursively removing features and building
/// a model on the remaining features. It ranks features by importance and removes
/// the least important ones until the desired number of features is reached.
/// </para>
/// <para>
/// The algorithm:
/// 1. Train a model on all features and compute feature importances
/// 2. Remove the least important feature(s)
/// 3. Repeat until desired number of features is reached
/// </para>
/// <para><b>For Beginners:</b> RFE is like an elimination tournament for features:
/// - Start with all features
/// - Remove the weakest performer each round
/// - Keep going until you have the desired number of features
/// - The surviving features are the most important ones
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RFE<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _step;
    private readonly RFEImportanceMethod _importanceMethod;

    // Fitted parameters
    private int[]? _ranking; // Ranking for each feature (1 = selected, 2+ = eliminated)
    private bool[]? _supportMask;
    private int[]? _selectedIndices;
    private double[]? _featureImportances;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>
    /// Gets the step size (features removed per iteration).
    /// </summary>
    public int Step => _step;

    /// <summary>
    /// Gets the feature ranking (1 = selected, 2+ = elimination order).
    /// </summary>
    public int[]? Ranking => _ranking;

    /// <summary>
    /// Gets the feature importances from the final model.
    /// </summary>
    public double[]? FeatureImportances => _featureImportances;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="RFE{T}"/>.
    /// </summary>
    /// <param name="nFeaturesToSelect">Number of features to select. Defaults to 5.</param>
    /// <param name="step">Number of features to remove at each iteration. Defaults to 1.</param>
    /// <param name="importanceMethod">Method for computing feature importance. Defaults to Correlation.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public RFE(
        int nFeaturesToSelect = 5,
        int step = 1,
        RFEImportanceMethod importanceMethod = RFEImportanceMethod.Correlation,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
        {
            throw new ArgumentException("Number of features to select must be at least 1.", nameof(nFeaturesToSelect));
        }

        if (step < 1)
        {
            throw new ArgumentException("Step must be at least 1.", nameof(step));
        }

        _nFeaturesToSelect = nFeaturesToSelect;
        _step = step;
        _importanceMethod = importanceMethod;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RFE requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits RFE by recursively eliminating features.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int nToSelect = Math.Min(_nFeaturesToSelect, p);

        // Convert to double arrays
        var X = new double[n, p];
        var y = new double[n];

        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Initialize ranking
        _ranking = new int[p];
        for (int j = 0; j < p; j++)
        {
            _ranking[j] = 0; // Not yet ranked
        }

        // Track remaining features
        var remaining = new HashSet<int>(Enumerable.Range(0, p));
        int currentRank = p;

        // Recursive elimination
        while (remaining.Count > nToSelect)
        {
            // Compute importances for remaining features
            var importances = ComputeImportances(X, y, remaining.ToArray(), n);

            // Determine how many to remove this iteration
            int nToRemove = Math.Min(_step, remaining.Count - nToSelect);

            // Find least important features
            var sortedByImportance = remaining
                .OrderBy(i => importances[i])
                .Take(nToRemove)
                .ToArray();

            // Remove and rank
            foreach (int featureIdx in sortedByImportance)
            {
                _ranking[featureIdx] = currentRank--;
                remaining.Remove(featureIdx);
            }
        }

        // Remaining features get rank 1
        foreach (int featureIdx in remaining)
        {
            _ranking[featureIdx] = 1;
        }

        // Create support mask and selected indices
        _supportMask = new bool[p];
        var selectedList = new List<int>();

        for (int j = 0; j < p; j++)
        {
            if (_ranking[j] == 1)
            {
                _supportMask[j] = true;
                selectedList.Add(j);
            }
        }

        _selectedIndices = selectedList.OrderBy(i => i).ToArray();

        // Compute final feature importances on selected features
        _featureImportances = ComputeImportances(X, y, _selectedIndices, n);

        IsFitted = true;
    }

    private double[] ComputeImportances(double[,] X, double[] y, int[] featureIndices, int n)
    {
        var importances = new double[_nInputFeatures];

        switch (_importanceMethod)
        {
            case RFEImportanceMethod.Correlation:
                // Use absolute correlation as importance
                foreach (int j in featureIndices)
                {
                    double xMean = 0, yMean = y.Average();
                    for (int i = 0; i < n; i++)
                    {
                        xMean += X[i, j];
                    }
                    xMean /= n;

                    double ssXY = 0, ssXX = 0, ssYY = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double dx = X[i, j] - xMean;
                        double dy = y[i] - yMean;
                        ssXY += dx * dy;
                        ssXX += dx * dx;
                        ssYY += dy * dy;
                    }

                    if (ssXX > 1e-10 && ssYY > 1e-10)
                    {
                        double r = ssXY / Math.Sqrt(ssXX * ssYY);
                        importances[j] = Math.Abs(r);
                    }
                }
                break;

            case RFEImportanceMethod.LinearCoefficients:
                // Use absolute value of OLS coefficients
                var coefficients = ComputeLinearCoefficients(X, y, featureIndices, n);
                foreach (var kvp in coefficients)
                {
                    importances[kvp.Key] = Math.Abs(kvp.Value);
                }
                break;

            case RFEImportanceMethod.FStatistic:
                // Use F-statistic as importance
                foreach (int j in featureIndices)
                {
                    var featureValues = new double[n];
                    for (int i = 0; i < n; i++)
                    {
                        featureValues[i] = X[i, j];
                    }
                    importances[j] = ComputeFStatistic(featureValues, y);
                }
                break;
        }

        return importances;
    }

    private Dictionary<int, double> ComputeLinearCoefficients(double[,] X, double[] y, int[] featureIndices, int n)
    {
        int p = featureIndices.Length;
        var coefficients = new Dictionary<int, double>();

        if (p == 0) return coefficients;

        // Simple OLS for selected features
        // Compute X^T X and X^T y
        var XTX = new double[p, p];
        var XTy = new double[p];

        // Center the data
        var xMeans = new double[p];
        double yMean = y.Average();

        for (int k = 0; k < p; k++)
        {
            int j = featureIndices[k];
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += X[i, j];
            }
            xMeans[k] = sum / n;
        }

        for (int k1 = 0; k1 < p; k1++)
        {
            int j1 = featureIndices[k1];
            for (int k2 = 0; k2 < p; k2++)
            {
                int j2 = featureIndices[k2];
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    sum += (X[i, j1] - xMeans[k1]) * (X[i, j2] - xMeans[k2]);
                }
                XTX[k1, k2] = sum;
            }

            // Add regularization
            XTX[k1, k1] += 1e-6;

            double ySum = 0;
            for (int i = 0; i < n; i++)
            {
                ySum += (X[i, j1] - xMeans[k1]) * (y[i] - yMean);
            }
            XTy[k1] = ySum;
        }

        // Solve using simple Gaussian elimination
        var beta = SolveLinearSystem(XTX, XTy, p);

        for (int k = 0; k < p; k++)
        {
            coefficients[featureIndices[k]] = beta[k];
        }

        return coefficients;
    }

    private double[] SolveLinearSystem(double[,] A, double[] b, int n)
    {
        var augmented = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(augmented[row, col]) > Math.Abs(augmented[maxRow, col]))
                {
                    maxRow = row;
                }
            }

            for (int j = col; j <= n; j++)
            {
                double temp = augmented[col, j];
                augmented[col, j] = augmented[maxRow, j];
                augmented[maxRow, j] = temp;
            }

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

    private double ComputeFStatistic(double[] x, double[] y)
    {
        int n = x.Length;
        double xMean = x.Average();
        double yMean = y.Average();

        double ssXY = 0, ssXX = 0, ssYY = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - xMean;
            double dy = y[i] - yMean;
            ssXY += dx * dy;
            ssXX += dx * dx;
            ssYY += dy * dy;
        }

        if (ssXX < 1e-10 || ssYY < 1e-10)
        {
            return 0;
        }

        double r2 = (ssXY * ssXY) / (ssXX * ssYY);
        int df2 = n - 2;

        if (df2 <= 0)
        {
            return 0;
        }

        return (r2) / ((1 - r2) / df2);
    }

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by selecting the most important features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("RFE has not been fitted.");
        }

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                result[i, j] = data[i, _selectedIndices[j]];
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RFE does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the support mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_supportMask is null)
        {
            throw new InvalidOperationException("RFE has not been fitted.");
        }
        return (bool[])_supportMask.Clone();
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
        {
            return Array.Empty<string>();
        }

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}

/// <summary>
/// Specifies the method for computing feature importance in RFE.
/// </summary>
public enum RFEImportanceMethod
{
    /// <summary>
    /// Use absolute correlation with target.
    /// </summary>
    Correlation,

    /// <summary>
    /// Use absolute value of linear regression coefficients.
    /// </summary>
    LinearCoefficients,

    /// <summary>
    /// Use F-statistic from univariate regression.
    /// </summary>
    FStatistic
}
