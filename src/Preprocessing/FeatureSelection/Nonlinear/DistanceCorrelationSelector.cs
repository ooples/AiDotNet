using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Nonlinear;

/// <summary>
/// Distance Correlation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on distance correlation with the target,
/// which detects all types of dependence including nonlinear.
/// </para>
/// <para><b>For Beginners:</b> Distance correlation measures dependence between
/// variables using pairwise distances. Unlike Pearson correlation which only
/// detects linear relationships, distance correlation equals zero if and only if
/// the variables are truly independent. It ranges from 0 to 1.
/// </para>
/// </remarks>
public class DistanceCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _dcorValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DCorValues => _dcorValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DistanceCorrelationSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DistanceCorrelationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _dcorValues = new double[p];

        // Precompute distance matrix for target
        var yDist = ComputeDistanceMatrix(y);
        var yCentered = DoubleCenterMatrix(yDist);
        double yVar = ComputeDVariance(yCentered);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            var xDist = ComputeDistanceMatrix(col);
            var xCentered = DoubleCenterMatrix(xDist);
            double xVar = ComputeDVariance(xCentered);

            if (xVar < 1e-10 || yVar < 1e-10)
            {
                _dcorValues[j] = 0;
                continue;
            }

            // Distance covariance
            double dcov = 0;
            for (int i1 = 0; i1 < n; i1++)
                for (int i2 = 0; i2 < n; i2++)
                    dcov += xCentered[i1, i2] * yCentered[i1, i2];
            dcov = Math.Sqrt(dcov / (n * n));

            // Distance correlation
            _dcorValues[j] = dcov / Math.Sqrt(Math.Sqrt(xVar * yVar));
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _dcorValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] ComputeDistanceMatrix(double[] data)
    {
        int n = data.Length;
        var dist = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                dist[i, j] = Math.Abs(data[i] - data[j]);
        return dist;
    }

    private double[,] DoubleCenterMatrix(double[,] dist)
    {
        int n = dist.GetLength(0);
        var centered = new double[n, n];

        var rowMeans = new double[n];
        var colMeans = new double[n];
        double grandMean = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                rowMeans[i] += dist[i, j];
                colMeans[j] += dist[i, j];
                grandMean += dist[i, j];
            }
            rowMeans[i] /= n;
        }
        for (int j = 0; j < n; j++)
            colMeans[j] /= n;
        grandMean /= (n * n);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                centered[i, j] = dist[i, j] - rowMeans[i] - colMeans[j] + grandMean;

        return centered;
    }

    private double ComputeDVariance(double[,] centered)
    {
        int n = centered.GetLength(0);
        double sum = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                sum += centered[i, j] * centered[i, j];
        return sum / (n * n);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DistanceCorrelationSelector has not been fitted.");

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
        throw new NotSupportedException("DistanceCorrelationSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DistanceCorrelationSelector has not been fitted.");

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
