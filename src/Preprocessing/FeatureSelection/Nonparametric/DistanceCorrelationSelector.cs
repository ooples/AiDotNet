using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Nonparametric;

/// <summary>
/// Distance Correlation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on distance correlation with the target, which can
/// detect both linear and nonlinear dependencies between variables.
/// </para>
/// <para><b>For Beginners:</b> Unlike Pearson correlation that only finds linear
/// relationships, distance correlation can find any type of relationship. If two
/// variables are truly independent, distance correlation will be zero, but it can
/// also detect complex nonlinear patterns.
/// </para>
/// </remarks>
public class DistanceCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _dcorScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DcorScores => _dcorScores;
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

        _dcorScores = new double[p];

        // Compute distance matrix for y once
        var distY = ComputeDistanceMatrix(y, n);
        var centeredY = CenterDistanceMatrix(distY, n);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            var distX = ComputeDistanceMatrix(col, n);
            var centeredX = CenterDistanceMatrix(distX, n);

            _dcorScores[j] = ComputeDistanceCorrelation(centeredX, centeredY, n);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _dcorScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] ComputeDistanceMatrix(double[] x, int n)
    {
        var dist = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                dist[i, j] = Math.Abs(x[i] - x[j]);
            }
        }
        return dist;
    }

    private double[,] CenterDistanceMatrix(double[,] dist, int n)
    {
        var centered = new double[n, n];
        var rowMeans = new double[n];
        var colMeans = new double[n];
        double grandMean = 0;

        // Compute row and column means
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                rowMeans[i] += dist[i, j];
                colMeans[j] += dist[i, j];
                grandMean += dist[i, j];
            }
        }

        for (int i = 0; i < n; i++)
        {
            rowMeans[i] /= n;
            colMeans[i] /= n;
        }
        grandMean /= (n * n);

        // Double centering
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                centered[i, j] = dist[i, j] - rowMeans[i] - colMeans[j] + grandMean;
            }
        }

        return centered;
    }

    private double ComputeDistanceCorrelation(double[,] A, double[,] B, int n)
    {
        double dcovXY = 0, dcovXX = 0, dcovYY = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                dcovXY += A[i, j] * B[i, j];
                dcovXX += A[i, j] * A[i, j];
                dcovYY += B[i, j] * B[i, j];
            }
        }

        dcovXY /= (n * n);
        dcovXX /= (n * n);
        dcovYY /= (n * n);

        double denom = Math.Sqrt(dcovXX) * Math.Sqrt(dcovYY);
        return denom > 1e-10 ? Math.Sqrt(Math.Max(0, dcovXY)) / Math.Sqrt(denom) : 0;
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
