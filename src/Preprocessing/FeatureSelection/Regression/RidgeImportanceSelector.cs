using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Regression;

/// <summary>
/// Ridge Regression Importance based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their coefficient magnitudes from ridge regression,
/// which handles multicollinearity well while preserving all features.
/// </para>
/// <para><b>For Beginners:</b> Ridge regression shrinks coefficients toward zero
/// but doesn't eliminate them. Features with larger coefficients after this
/// shrinkage are more important. Unlike Lasso, Ridge handles correlated features
/// gracefully by sharing importance among them.
/// </para>
/// </remarks>
public class RidgeImportanceSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;

    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public RidgeImportanceSelector(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RidgeImportanceSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Standardize
        var means = new double[p];
        var stds = new double[p];
        double yMean = y.Average();

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += X[i, j];
            means[j] /= n;

            for (int i = 0; i < n; i++)
                stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;

            for (int i = 0; i < n; i++)
                X[i, j] = (X[i, j] - means[j]) / stds[j];
        }

        for (int i = 0; i < n; i++)
            y[i] -= yMean;

        // Compute X^T X + alpha * I
        var XtX = new double[p, p];
        var Xty = new double[p];

        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < n; k++)
                    XtX[i, j] += X[k, i] * X[k, j];
            }
            XtX[i, i] += _alpha; // Ridge penalty

            for (int k = 0; k < n; k++)
                Xty[i] += X[k, i] * y[k];
        }

        // Solve (X^T X + alpha I) beta = X^T y
        _coefficients = SolveSystem(XtX, Xty, p);

        // Select features with largest coefficients
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => Math.Abs(_coefficients[j]))
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] SolveSystem(double[,] A, double[] b, int n)
    {
        var aug = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i, j] = A[i, j];
            aug[i, n] = b[i];
        }

        // Gaussian elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;

            for (int j = 0; j <= n; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            if (Math.Abs(aug[col, col]) < 1e-10) continue;

            for (int row = col + 1; row < n; row++)
            {
                double factor = aug[row, col] / aug[col, col];
                for (int j = col; j <= n; j++)
                    aug[row, j] -= factor * aug[col, j];
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = aug[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= aug[i, j] * x[j];
            x[i] /= (Math.Abs(aug[i, i]) > 1e-10 ? aug[i, i] : 1);
        }

        return x;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RidgeImportanceSelector has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RidgeImportanceSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RidgeImportanceSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
