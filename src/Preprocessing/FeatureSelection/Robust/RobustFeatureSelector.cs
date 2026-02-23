using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Robust;

/// <summary>
/// Robust Feature Selection resistant to outliers and noise.
/// </summary>
/// <remarks>
/// <para>
/// Uses robust statistics (median, MAD, rank-based measures) to identify
/// important features while being resistant to outliers and noise in the data.
/// </para>
/// <para><b>For Beginners:</b> Regular feature selection can be fooled by extreme
/// values (outliers) in the data. This method uses statistics that are "robust"
/// - they don't change much when a few data points are extreme. This gives
/// more reliable feature selection for messy real-world data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RobustFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _outlierThreshold;

    private double[]? _robustScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? RobustScores => _robustScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public RobustFeatureSelector(
        int nFeaturesToSelect = 10,
        double outlierThreshold = 3.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _outlierThreshold = outlierThreshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RobustFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to arrays
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Identify non-outlier indices based on target
        var nonOutlierIndices = IdentifyNonOutliers(y, n);

        // Compute robust correlation scores
        _robustScores = new double[p];
        for (int j = 0; j < p; j++)
            _robustScores[j] = ComputeRobustCorrelation(X, y, j, nonOutlierIndices, n);

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => Math.Abs(_robustScores[j]))
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private List<int> IdentifyNonOutliers(double[] values, int n)
    {
        var sorted = values.OrderBy(v => v).ToList();
        double median = sorted[n / 2];

        var absoluteDeviations = values.Select(v => Math.Abs(v - median)).OrderBy(d => d).ToList();
        double mad = absoluteDeviations[n / 2] + 1e-10;

        return Enumerable.Range(0, n)
            .Where(i => Math.Abs(values[i] - median) / mad <= _outlierThreshold)
            .ToList();
    }

    private double ComputeRobustCorrelation(double[,] X, double[] y, int featureIdx, List<int> indices, int totalN)
    {
        int n = indices.Count;
        if (n < 3) return 0;

        // Use Spearman correlation (rank-based, robust to outliers)
        var xRanks = new double[n];
        var yRanks = new double[n];

        var xSorted = indices.OrderBy(i => X[i, featureIdx]).ToList();
        var ySorted = indices.OrderBy(i => y[i]).ToList();

        for (int i = 0; i < n; i++)
        {
            xRanks[indices.IndexOf(xSorted[i])] = i + 1;
            yRanks[indices.IndexOf(ySorted[i])] = i + 1;
        }

        // Pearson correlation on ranks
        double xMean = xRanks.Average();
        double yMean = yRanks.Average();

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = xRanks[i] - xMean;
            double yDiff = yRanks[i] - yMean;
            sxy += xDiff * yDiff;
            sxx += xDiff * xDiff;
            syy += yDiff * yDiff;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RobustFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("RobustFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RobustFeatureSelector has not been fitted.");

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
