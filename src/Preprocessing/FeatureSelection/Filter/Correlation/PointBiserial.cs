using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Point-Biserial Correlation for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Point-biserial correlation measures the relationship between a continuous
/// variable and a dichotomous (binary) variable. It's the appropriate correlation
/// measure when the target is binary.
/// </para>
/// <para><b>For Beginners:</b> When you're trying to predict something that has
/// only two outcomes (like yes/no, pass/fail), this measures how well a continuous
/// feature separates the two groups. Higher absolute values mean the feature is
/// better at distinguishing between the classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PointBiserial<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _correlations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Correlations => _correlations;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PointBiserial(
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
            "PointBiserial requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Split samples by class
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

        _correlations = new double[p];

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

            double sTotal = 0;
            double meanTotal = 0;
            for (int i = 0; i < n; i++)
                meanTotal += NumOps.ToDouble(data[i, j]);
            meanTotal /= n;

            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - meanTotal;
                sTotal += diff * diff;
            }
            double stdTotal = Math.Sqrt(sTotal / n);

            if (stdTotal < 1e-10)
            {
                _correlations[j] = 0;
                continue;
            }

            double pq = (double)(n0 * n1) / (n * n);
            _correlations[j] = Math.Abs((mean1 - mean0) / stdTotal * Math.Sqrt(pq));
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _correlations[j])
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
            throw new InvalidOperationException("PointBiserial has not been fitted.");

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
        throw new NotSupportedException("PointBiserial does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PointBiserial has not been fitted.");

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
