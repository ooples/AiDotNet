using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Hotelling's T² based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on Hotelling's T² statistic, the multivariate
/// generalization of the t-test for comparing two group means.
/// </para>
/// <para><b>For Beginners:</b> While a t-test compares means of one variable,
/// Hotelling's T² compares mean vectors of multiple variables simultaneously.
/// It accounts for correlations between features when testing if two groups differ.
/// Features contributing most to group separation are selected.
/// </para>
/// </remarks>
public class HotellingT2Selector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _t2Contributions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? T2Contributions => _t2Contributions;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public HotellingT2Selector(
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
            "HotellingT2Selector requires binary target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        if (classes.Count != 2)
            throw new ArgumentException("Hotelling's T² requires exactly 2 classes.");

        var group1 = Enumerable.Range(0, n).Where(i => y[i] == classes[0]).ToList();
        var group2 = Enumerable.Range(0, n).Where(i => y[i] == classes[1]).ToList();
        int n1 = group1.Count;
        int n2 = group2.Count;

        _t2Contributions = new double[p];

        // Compute mean difference and pooled variance for each feature
        var meanDiffs = new double[p];
        var pooledVars = new double[p];

        for (int j = 0; j < p; j++)
        {
            double mean1 = group1.Select(i => X[i, j]).Average();
            double mean2 = group2.Select(i => X[i, j]).Average();
            meanDiffs[j] = mean1 - mean2;

            double var1 = group1.Select(i => (X[i, j] - mean1) * (X[i, j] - mean1)).Sum();
            double var2 = group2.Select(i => (X[i, j] - mean2) * (X[i, j] - mean2)).Sum();
            pooledVars[j] = (var1 + var2) / (n1 + n2 - 2);
        }

        // Univariate T² contribution for each feature
        // T² = (n1*n2)/(n1+n2) * (mean1-mean2)² / pooledVar
        for (int j = 0; j < p; j++)
        {
            if (pooledVars[j] < 1e-10)
            {
                _t2Contributions[j] = 0;
                continue;
            }

            double t2 = ((double)n1 * n2 / (n1 + n2)) * meanDiffs[j] * meanDiffs[j] / pooledVars[j];
            _t2Contributions[j] = t2;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _t2Contributions[j])
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
            throw new InvalidOperationException("HotellingT2Selector has not been fitted.");

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
        throw new NotSupportedException("HotellingT2Selector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HotellingT2Selector has not been fitted.");

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
