using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Distance Correlation-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Distance correlation measures the dependence between random variables, detecting
/// both linear and nonlinear relationships. Unlike Pearson correlation, it equals
/// zero if and only if the variables are independent.
/// </para>
/// <para><b>For Beginners:</b> Regular correlation only catches straight-line
/// relationships. Distance correlation can detect curved, U-shaped, or any type
/// of pattern between variables. A distance correlation of 0 means truly no
/// relationship at all.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DistanceCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _distCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DistCorrelations => _distCorrelations;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DistanceCorrelation(
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
            "DistanceCorrelation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Precompute target distances
        var targetDist = ComputeDistanceMatrix(target, n);
        var targetCentered = CenterDistanceMatrix(targetDist, n);

        _distCorrelations = new double[p];

        for (int j = 0; j < p; j++)
        {
            var featureDist = ComputeFeatureDistanceMatrix(data, j, n);
            var featureCentered = CenterDistanceMatrix(featureDist, n);

            double dcov2XY = ComputeDCov2(featureCentered, targetCentered, n);
            double dcov2X = ComputeDCov2(featureCentered, featureCentered, n);
            double dcov2Y = ComputeDCov2(targetCentered, targetCentered, n);

            if (dcov2X > 1e-10 && dcov2Y > 1e-10)
                _distCorrelations[j] = Math.Sqrt(dcov2XY / Math.Sqrt(dcov2X * dcov2Y));
            else
                _distCorrelations[j] = 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _distCorrelations[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] ComputeDistanceMatrix(Vector<T> vec, int n)
    {
        var dist = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double d = Math.Abs(NumOps.ToDouble(vec[i]) - NumOps.ToDouble(vec[j]));
                dist[i, j] = d;
                dist[j, i] = d;
            }
        }
        return dist;
    }

    private double[,] ComputeFeatureDistanceMatrix(Matrix<T> data, int col, int n)
    {
        var dist = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double d = Math.Abs(NumOps.ToDouble(data[i, col]) - NumOps.ToDouble(data[j, col]));
                dist[i, j] = d;
                dist[j, i] = d;
            }
        }
        return dist;
    }

    private double[,] CenterDistanceMatrix(double[,] dist, int n)
    {
        var rowMeans = new double[n];
        var colMeans = new double[n];
        double grandMean = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                rowMeans[i] += dist[i, j];
                colMeans[j] += dist[i, j];
            }
            rowMeans[i] /= n;
        }

        for (int j = 0; j < n; j++)
        {
            colMeans[j] /= n;
            grandMean += colMeans[j];
        }
        grandMean /= n;

        var centered = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                centered[i, j] = dist[i, j] - rowMeans[i] - colMeans[j] + grandMean;

        return centered;
    }

    private double ComputeDCov2(double[,] a, double[,] b, int n)
    {
        double sum = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                sum += a[i, j] * b[i, j];
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
            throw new InvalidOperationException("DistanceCorrelation has not been fitted.");

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
        throw new NotSupportedException("DistanceCorrelation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DistanceCorrelation has not been fitted.");

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
