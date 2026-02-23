using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// T-Test based feature selection for binary classification.
/// </summary>
/// <remarks>
/// <para>
/// Uses independent samples t-test to measure how well each feature separates
/// two classes. Features with higher t-statistics have significantly different
/// means between classes.
/// </para>
/// <para><b>For Beginners:</b> The t-test asks "are the average values of this
/// feature significantly different between the two groups?" A high t-statistic
/// means the feature's values are very different for each class, making it useful
/// for telling them apart.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
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

        var class0 = new List<int>();
        var class1 = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                class0.Add(i);
            else
                class1.Add(i);
        }

        int n0 = class0.Count;
        int n1 = class1.Count;

        _tStatistics = new double[p];
        _pValues = new double[p];

        if (n0 < 2 || n1 < 2)
        {
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        for (int j = 0; j < p; j++)
        {
            double mean0 = class0.Sum(i => NumOps.ToDouble(data[i, j])) / n0;
            double mean1 = class1.Sum(i => NumOps.ToDouble(data[i, j])) / n1;

            double var0 = class0.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean0, 2)) / (n0 - 1);
            double var1 = class1.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean1, 2)) / (n1 - 1);

            double pooledSE = Math.Sqrt(var0 / n0 + var1 / n1);
            _tStatistics[j] = pooledSE > 1e-10 ? Math.Abs(mean0 - mean1) / pooledSE : 0;

            // Approximate p-value using normal distribution for large samples
            double z = _tStatistics[j];
            _pValues[j] = 2 * (1 - NormalCDF(z));
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _tStatistics[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double NormalCDF(double x)
    {
        double t = 1.0 / (1.0 + 0.2316419 * Math.Abs(x));
        double d = 0.3989423 * Math.Exp(-x * x / 2);
        double p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return x > 0 ? 1 - p : p;
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
