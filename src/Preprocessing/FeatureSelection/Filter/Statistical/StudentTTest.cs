using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// Student's t-test for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Student's t-test compares the means of two groups to determine if they are
/// statistically different. For feature selection, it tests whether each feature
/// has significantly different values between positive and negative classes.
/// </para>
/// <para><b>For Beginners:</b> This test asks: "Is the average value of this feature
/// different enough between the two classes that it's probably not due to random
/// chance?" Features with large differences are more likely to be useful for
/// distinguishing between classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StudentTTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly bool _equalVariances;

    private double[]? _tStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public bool EqualVariances => _equalVariances;
    public double[]? TStatistics => _tStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public StudentTTest(
        int nFeaturesToSelect = 10,
        double alpha = 0.05,
        bool equalVariances = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _equalVariances = equalVariances;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StudentTTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Split samples by class (binary classification)
        var group0 = new List<int>();
        var group1 = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                group0.Add(i);
            else
                group1.Add(i);
        }

        int n0 = group0.Count;
        int n1 = group1.Count;

        if (n0 < 2 || n1 < 2)
        {
            _tStatistics = new double[p];
            _pValues = Enumerable.Repeat(1.0, p).ToArray();
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        _tStatistics = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute means
            double mean0 = group0.Sum(i => NumOps.ToDouble(data[i, j])) / n0;
            double mean1 = group1.Sum(i => NumOps.ToDouble(data[i, j])) / n1;

            // Compute variances
            double var0 = group0.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean0, 2)) / (n0 - 1);
            double var1 = group1.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean1, 2)) / (n1 - 1);

            double t, df;

            if (_equalVariances)
            {
                // Pooled variance (Student's t-test)
                double sp2 = ((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2);
                double se = Math.Sqrt(sp2 * (1.0 / n0 + 1.0 / n1));

                t = se > 1e-10 ? (mean0 - mean1) / se : 0;
                df = n0 + n1 - 2;
            }
            else
            {
                // Welch's t-test (unequal variances)
                double se = Math.Sqrt(var0 / n0 + var1 / n1);
                t = se > 1e-10 ? (mean0 - mean1) / se : 0;

                // Welch-Satterthwaite degrees of freedom
                double num = Math.Pow(var0 / n0 + var1 / n1, 2);
                double den = Math.Pow(var0 / n0, 2) / (n0 - 1) + Math.Pow(var1 / n1, 2) / (n1 - 1);
                df = den > 0 ? num / den : n0 + n1 - 2;
            }

            _tStatistics[j] = t;

            // Approximate p-value using normal distribution for large df
            _pValues[j] = 2 * (1 - NormalCDF(Math.Abs(t)));
        }

        // Select features with smallest p-values
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _pValues
            .Select((pval, idx) => (PValue: pval, Index: idx))
            .OrderBy(x => x.PValue)
            .Take(numToSelect)
            .Select(x => x.Index)
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
            throw new InvalidOperationException("StudentTTest has not been fitted.");

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
        throw new NotSupportedException("StudentTTest does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StudentTTest has not been fitted.");

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
