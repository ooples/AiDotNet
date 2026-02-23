using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Distance Correlation for detecting nonlinear relationships between features and target.
/// </summary>
/// <remarks>
/// <para>
/// Distance correlation measures both linear and nonlinear dependencies between variables.
/// Unlike Pearson correlation which only detects linear relationships, distance correlation
/// equals zero if and only if the variables are statistically independent.
/// </para>
/// <para><b>For Beginners:</b> Pearson correlation can miss relationships where one variable
/// depends on another in a curved or complex way. Distance correlation catches all types
/// of relationships. If it's zero, the variables are truly independent; if it's high,
/// there's some kind of relationship (linear or not).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DistanceCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minCorrelation;

    private double[]? _distanceCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DistanceCorrelations => _distanceCorrelations;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DistanceCorrelation(
        int nFeaturesToSelect = 10,
        double minCorrelation = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minCorrelation = minCorrelation;
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

        // Compute distance matrix for target
        var targetValues = new double[n];
        for (int i = 0; i < n; i++)
            targetValues[i] = NumOps.ToDouble(target[i]);

        var targetDistMatrix = ComputeDistanceMatrix(targetValues);
        var targetCentered = CenterDistanceMatrix(targetDistMatrix, n);

        _distanceCorrelations = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Extract feature column
            var featureValues = new double[n];
            for (int i = 0; i < n; i++)
                featureValues[i] = NumOps.ToDouble(data[i, j]);

            var featureDistMatrix = ComputeDistanceMatrix(featureValues);
            var featureCentered = CenterDistanceMatrix(featureDistMatrix, n);

            // Compute distance covariance and variances
            double dCov = ComputeDistanceCovariance(featureCentered, targetCentered, n);
            double dVarX = ComputeDistanceCovariance(featureCentered, featureCentered, n);
            double dVarY = ComputeDistanceCovariance(targetCentered, targetCentered, n);

            // Distance correlation
            double denominator = Math.Sqrt(dVarX * dVarY);
            _distanceCorrelations[j] = denominator > 1e-10 ? dCov / denominator : 0;
        }

        // Select features above threshold or top by distance correlation
        var candidates = new List<int>();
        for (int j = 0; j < p; j++)
            if (_distanceCorrelations[j] >= _minCorrelation)
                candidates.Add(j);

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = candidates
                .OrderByDescending(j => _distanceCorrelations[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = _distanceCorrelations
                .Select((dc, idx) => (DC: dc, Index: idx))
                .OrderByDescending(x => x.DC)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double[,] ComputeDistanceMatrix(double[] values)
    {
        int n = values.Length;
        var dist = new double[n, n];

        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                dist[i, k] = Math.Abs(values[i] - values[k]);

        return dist;
    }

    private double[,] CenterDistanceMatrix(double[,] dist, int n)
    {
        var centered = new double[n, n];

        // Compute row means, column means, and grand mean
        var rowMeans = new double[n];
        var colMeans = new double[n];
        double grandMean = 0;

        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < n; k++)
            {
                rowMeans[i] += dist[i, k];
                colMeans[k] += dist[i, k];
                grandMean += dist[i, k];
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
            for (int k = 0; k < n; k++)
                centered[i, k] = dist[i, k] - rowMeans[i] - colMeans[k] + grandMean;

        return centered;
    }

    private double ComputeDistanceCovariance(double[,] a, double[,] b, int n)
    {
        double sum = 0;
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                sum += a[i, k] * b[i, k];

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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
