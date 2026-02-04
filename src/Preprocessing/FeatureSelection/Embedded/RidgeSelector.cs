using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Ridge Regression (L2) based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Ridge regression adds L2 regularization to linear regression, shrinking
/// coefficients toward zero. Features with larger absolute coefficients
/// after regularization are considered more important.
/// </para>
/// <para><b>For Beginners:</b> Unlike Lasso (L1), Ridge doesn't set coefficients
/// exactly to zero. Instead, it shrinks all coefficients proportionally.
/// Features with larger coefficients after shrinking are more important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RidgeSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
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

    public RidgeSelector(int nFeaturesToSelect = 10, double alpha = 1.0, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha < 0)
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RidgeSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert and standardize
        var X = new double[n, p];
        var y = new double[n];
        var xMeans = new double[p];
        var xStds = new double[p];

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                xMeans[j] += NumOps.ToDouble(data[i, j]);
            xMeans[j] /= n;
        }

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                xStds[j] += Math.Pow(NumOps.ToDouble(data[i, j]) - xMeans[j], 2);
            xStds[j] = Math.Sqrt(xStds[j] / n);
            if (xStds[j] < 1e-10) xStds[j] = 1;
        }

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]) - yMean;
            for (int j = 0; j < p; j++)
                X[i, j] = (NumOps.ToDouble(data[i, j]) - xMeans[j]) / xStds[j];
        }

        // Compute X^T * X + alpha * I
        var XtX = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < n; k++)
                    XtX[i, j] += X[k, i] * X[k, j];
            }
            XtX[i, i] += _alpha;  // Regularization
        }

        // Compute X^T * y
        var Xty = new double[p];
        for (int j = 0; j < p; j++)
            for (int i = 0; i < n; i++)
                Xty[j] += X[i, j] * y[i];

        // Solve (X^T * X + alpha * I) * beta = X^T * y
        _coefficients = SolveLinearSystem(XtX, Xty, p);

        // Select features by absolute coefficient magnitude
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _coefficients
            .Select((c, idx) => (AbsCoef: Math.Abs(c), Index: idx))
            .OrderByDescending(x => x.AbsCoef)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] SolveLinearSystem(double[,] A, double[] b, int n)
    {
        // Gaussian elimination with partial pivoting
        var augmented = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                augmented[i, j] = A[i, j];
            augmented[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            // Partial pivoting
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(augmented[row, col]) > Math.Abs(augmented[maxRow, col]))
                    maxRow = row;

            for (int j = 0; j <= n; j++)
            {
                double temp = augmented[col, j];
                augmented[col, j] = augmented[maxRow, j];
                augmented[maxRow, j] = temp;
            }

            if (Math.Abs(augmented[col, col]) < 1e-10)
                continue;

            // Eliminate
            for (int row = col + 1; row < n; row++)
            {
                double factor = augmented[row, col] / augmented[col, col];
                for (int j = col; j <= n; j++)
                    augmented[row, j] -= factor * augmented[col, j];
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = augmented[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= augmented[i, j] * x[j];

            if (Math.Abs(augmented[i, i]) > 1e-10)
                x[i] /= augmented[i, i];
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
            throw new InvalidOperationException("RidgeSelector has not been fitted.");

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
        throw new NotSupportedException("RidgeSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RidgeSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
