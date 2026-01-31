using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Supervised;

/// <summary>
/// T-Test Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Student's t-test to select features that show statistically
/// significant differences between two classes.
/// </para>
/// <para><b>For Beginners:</b> The t-test checks whether two groups have
/// significantly different means. Features with large t-statistics have
/// means that are very different between classes, making them useful for
/// classification.
/// </para>
/// </remarks>
public class TTestSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _equalVariance;

    private double[]? _tStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? TStatistics => _tStatistics;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TTestSelector(
        int nFeaturesToSelect = 10,
        bool equalVariance = false,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _equalVariance = equalVariance;
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
            double yVal = NumOps.ToDouble(target[i]);
            y[i] = yVal > 0.5 ? 1 : 0;
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        int n1 = y.Count(yi => yi == 1);
        int n0 = n - n1;

        if (n0 < 2 || n1 < 2)
            throw new InvalidOperationException("Each class must have at least 2 samples.");

        _tStatistics = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute means and variances for each class
            double sum0 = 0, sum1 = 0;
            for (int i = 0; i < n; i++)
            {
                if (y[i] == 0)
                    sum0 += X[i, j];
                else
                    sum1 += X[i, j];
            }
            double mean0 = sum0 / n0;
            double mean1 = sum1 / n1;

            double var0 = 0, var1 = 0;
            for (int i = 0; i < n; i++)
            {
                if (y[i] == 0)
                    var0 += (X[i, j] - mean0) * (X[i, j] - mean0);
                else
                    var1 += (X[i, j] - mean1) * (X[i, j] - mean1);
            }
            var0 /= (n0 - 1);
            var1 /= (n1 - 1);

            double tStat;
            if (_equalVariance)
            {
                // Pooled variance t-test
                double pooledVar = ((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2);
                double se = Math.Sqrt(pooledVar * (1.0 / n0 + 1.0 / n1));
                tStat = (mean1 - mean0) / (se + 1e-10);
            }
            else
            {
                // Welch's t-test (unequal variances)
                double se = Math.Sqrt(var0 / n0 + var1 / n1);
                tStat = (mean1 - mean0) / (se + 1e-10);
            }

            _tStatistics[j] = Math.Abs(tStat);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _tStatistics[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
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
