using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Robustness;

/// <summary>
/// Outlier Robust Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that maintain their predictive relationship with the target
/// even when outliers are present, using robust statistics.
/// </para>
/// <para><b>For Beginners:</b> Some features are heavily influenced by extreme values
/// (outliers). This selector finds features whose relationship with the target is
/// consistent whether outliers are present or removed.
/// </para>
/// </remarks>
public class OutlierRobustSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _outlierThreshold;

    private double[]? _robustnessScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double OutlierThreshold => _outlierThreshold;
    public double[]? RobustnessScores => _robustnessScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public OutlierRobustSelector(
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
            "OutlierRobustSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _robustnessScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Compute robust statistics using median and MAD
            double median = GetMedian(col);
            double mad = GetMAD(col, median);

            // Identify outliers
            var isOutlier = new bool[n];
            for (int i = 0; i < n; i++)
                isOutlier[i] = mad > 1e-10 && Math.Abs(col[i] - median) / mad > _outlierThreshold;

            // Compute correlation with all data
            double corrAll = ComputeCorrelation(col, y);

            // Compute correlation without outliers
            var colFiltered = new List<double>();
            var yFiltered = new List<double>();
            for (int i = 0; i < n; i++)
            {
                if (!isOutlier[i])
                {
                    colFiltered.Add(col[i]);
                    yFiltered.Add(y[i]);
                }
            }

            double corrFiltered = colFiltered.Count > 2
                ? ComputeCorrelation(colFiltered.ToArray(), yFiltered.ToArray())
                : 0;

            // Robustness score: features where correlation is stable with/without outliers
            // We want high correlation AND stability
            double stability = 1 - Math.Abs(corrAll - corrFiltered);
            _robustnessScores[j] = Math.Abs(corrAll) * stability;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _robustnessScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double GetMedian(double[] values)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        int n = sorted.Length;
        if (n == 0) return 0;
        return n % 2 == 0
            ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
            : sorted[n / 2];
    }

    private double GetMAD(double[] values, double median)
    {
        var deviations = values.Select(v => Math.Abs(v - median)).ToArray();
        return GetMedian(deviations) * 1.4826; // Scale factor for normal distribution
    }

    private double ComputeCorrelation(double[] x, double[] y)
    {
        int n = x.Length;
        if (n < 2) return 0;

        double xMean = x.Average();
        double yMean = y.Average();

        double numerator = 0, xSumSq = 0, ySumSq = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = x[i] - xMean;
            double yDiff = y[i] - yMean;
            numerator += xDiff * yDiff;
            xSumSq += xDiff * xDiff;
            ySumSq += yDiff * yDiff;
        }

        double denominator = Math.Sqrt(xSumSq * ySumSq);
        return denominator > 1e-10 ? numerator / denominator : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("OutlierRobustSelector has not been fitted.");

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
        throw new NotSupportedException("OutlierRobustSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("OutlierRobustSelector has not been fitted.");

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
