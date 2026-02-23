using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Distance Correlation Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses distance correlation to detect both linear and non-linear dependencies
/// between features and the target. Distance correlation is zero if and only
/// if the variables are independent.
/// </para>
/// <para><b>For Beginners:</b> Distance correlation is like regular correlation
/// but can detect any type of relationship, not just linear ones. Unlike regular
/// correlation, a distance correlation of 0 truly means no relationship exists.
/// It computes distances between all pairs of points and correlates those distances.
/// </para>
/// </remarks>
public class DistanceCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _distanceCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DistanceCorrelations => _distanceCorrelations;
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

        // Compute centered distance matrix for y
        var Dy = ComputeCenteredDistanceMatrix(y, n);

        _distanceCorrelations = new double[p];
        for (int j = 0; j < p; j++)
        {
            var x = new double[n];
            for (int i = 0; i < n; i++) x[i] = X[i, j];
            var Dx = ComputeCenteredDistanceMatrix(x, n);

            _distanceCorrelations[j] = ComputeDistanceCorrelation(Dx, Dy, n);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _distanceCorrelations[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] ComputeCenteredDistanceMatrix(double[] x, int n)
    {
        // Compute pairwise distances
        var D = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                D[i, j] = Math.Abs(x[i] - x[j]);

        // Double-center the distance matrix
        var rowMeans = new double[n];
        var colMeans = new double[n];
        double grandMean = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                rowMeans[i] += D[i, j];
                colMeans[j] += D[i, j];
            }
            rowMeans[i] /= n;
        }

        for (int j = 0; j < n; j++)
        {
            colMeans[j] /= n;
            grandMean += colMeans[j];
        }
        grandMean /= n;

        // Center
        var A = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i, j] = D[i, j] - rowMeans[i] - colMeans[j] + grandMean;

        return A;
    }

    private double ComputeDistanceCorrelation(double[,] A, double[,] B, int n)
    {
        // dCov^2 = average of A[i,j] * B[i,j]
        double dCovSq = 0;
        double dVarASq = 0;
        double dVarBSq = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                dCovSq += A[i, j] * B[i, j];
                dVarASq += A[i, j] * A[i, j];
                dVarBSq += B[i, j] * B[i, j];
            }
        }

        dCovSq /= n * n;
        dVarASq /= n * n;
        dVarBSq /= n * n;

        if (dVarASq < 1e-10 || dVarBSq < 1e-10) return 0;

        return Math.Sqrt(dCovSq / Math.Sqrt(dVarASq * dVarBSq));
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
