using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Two-Sample T-Test based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their t-statistic from two-sample t-tests between
/// classes, identifying features with significantly different means.
/// </para>
/// <para><b>For Beginners:</b> The t-test checks if two groups have different
/// average values. This selector finds features where classes have significantly
/// different means, which helps in classification tasks.
/// </para>
/// </remarks>
public class TTestSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _tStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? TStatistics => _tStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TTestSelector(
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
            "TTestSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        var classes = y.Distinct().OrderBy(c => c).ToList();
        if (classes.Count < 2)
        {
            // Can't do t-test with less than 2 classes
            _tStatistics = new double[p];
            _pValues = Enumerable.Repeat(1.0, p).ToArray();
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        var classIndices = new Dictionary<int, List<int>>();
        foreach (var c in classes)
            classIndices[c] = Enumerable.Range(0, n).Where(i => y[i] == c).ToList();

        _tStatistics = new double[p];
        _pValues = new double[p];

        // For each feature, compute t-statistic between first two classes
        // (For multiclass, we use max t-stat across all pairs)
        for (int j = 0; j < p; j++)
        {
            double maxTStat = 0;
            double minPValue = 1.0;

            for (int c1Idx = 0; c1Idx < classes.Count; c1Idx++)
            {
                for (int c2Idx = c1Idx + 1; c2Idx < classes.Count; c2Idx++)
                {
                    int c1 = classes[c1Idx];
                    int c2 = classes[c2Idx];

                    var group1 = classIndices[c1].Select(i => X[i, j]).ToArray();
                    var group2 = classIndices[c2].Select(i => X[i, j]).ToArray();

                    var (tStat, pValue) = ComputeTTest(group1, group2);

                    if (Math.Abs(tStat) > Math.Abs(maxTStat))
                    {
                        maxTStat = tStat;
                        minPValue = pValue;
                    }
                }
            }

            _tStatistics[j] = Math.Abs(maxTStat);
            _pValues[j] = minPValue;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _tStatistics[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private (double tStat, double pValue) ComputeTTest(double[] group1, double[] group2)
    {
        int n1 = group1.Length;
        int n2 = group2.Length;

        if (n1 < 2 || n2 < 2) return (0, 1);

        double mean1 = group1.Average();
        double mean2 = group2.Average();

        double var1 = group1.Sum(x => (x - mean1) * (x - mean1)) / (n1 - 1);
        double var2 = group2.Sum(x => (x - mean2) * (x - mean2)) / (n2 - 1);

        double pooledSE = Math.Sqrt(var1 / n1 + var2 / n2);

        if (pooledSE < 1e-10) return (0, 1);

        double tStat = (mean1 - mean2) / pooledSE;

        // Approximate p-value using normal distribution (for large samples)
        double pValue = 2 * (1 - NormalCDF(Math.Abs(tStat)));

        return (tStat, pValue);
    }

    private double NormalCDF(double x)
    {
        // Approximation of standard normal CDF
        double t = 1.0 / (1.0 + 0.2316419 * x);
        double d = 0.3989423 * Math.Exp(-x * x / 2);
        double prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return x > 0 ? 1 - prob : prob;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TTestSelector has not been fitted.");

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
        throw new NotSupportedException("TTestSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TTestSelector has not been fitted.");

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
