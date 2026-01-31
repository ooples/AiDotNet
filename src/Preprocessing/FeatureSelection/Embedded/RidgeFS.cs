using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Ridge Regression-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Ridge Regression uses L2 regularization which shrinks all coefficients but
/// doesn't set them to exactly zero. Feature selection is done by selecting
/// features with the largest absolute coefficients.
/// </para>
/// <para><b>For Beginners:</b> Ridge regression shrinks all feature weights
/// towards zero but never quite reaches it. Unlike LASSO, it keeps all features
/// but makes less important ones smaller. We select features by choosing those
/// with the biggest weights after shrinkage.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RidgeFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
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

    public RidgeFS(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha <= 0)
            throw new ArgumentException("Alpha must be positive.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RidgeFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Center features and target
        var xMeans = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                xMeans[j] += NumOps.ToDouble(data[i, j]);
            xMeans[j] /= n;
        }

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        // Compute X'X + alpha * I and X'y
        var xtx = new double[p, p];
        var xty = new double[p];

        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = 0; j2 < p; j2++)
            {
                for (int i = 0; i < n; i++)
                {
                    double x1 = NumOps.ToDouble(data[i, j1]) - xMeans[j1];
                    double x2 = NumOps.ToDouble(data[i, j2]) - xMeans[j2];
                    xtx[j1, j2] += x1 * x2;
                }
            }

            // Add regularization to diagonal
            xtx[j1, j1] += _alpha;

            // Compute X'y
            for (int i = 0; i < n; i++)
            {
                double x = NumOps.ToDouble(data[i, j1]) - xMeans[j1];
                double y = NumOps.ToDouble(target[i]) - yMean;
                xty[j1] += x * y;
            }
        }

        // Solve (X'X + alpha*I) * beta = X'y using Cholesky decomposition
        _coefficients = SolveLinearSystem(xtx, xty, p);

        // Select features with largest absolute coefficients
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => Math.Abs(_coefficients[j]))
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] SolveLinearSystem(double[,] a, double[] b, int n)
    {
        // Cholesky decomposition: A = L * L'
        var l = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                    sum += l[i, k] * l[j, k];

                if (i == j)
                    l[i, j] = Math.Sqrt(Math.Max(0, a[i, i] - sum));
                else if (l[j, j] > 1e-10)
                    l[i, j] = (a[i, j] - sum) / l[j, j];
            }
        }

        // Solve L * y = b
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += l[i, j] * y[j];
            y[i] = l[i, i] > 1e-10 ? (b[i] - sum) / l[i, i] : 0;
        }

        // Solve L' * x = y
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = 0;
            for (int j = i + 1; j < n; j++)
                sum += l[j, i] * x[j];
            x[i] = l[i, i] > 1e-10 ? (y[i] - sum) / l[i, i] : 0;
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
            throw new InvalidOperationException("RidgeFS has not been fitted.");

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
        throw new NotSupportedException("RidgeFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RidgeFS has not been fitted.");

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
