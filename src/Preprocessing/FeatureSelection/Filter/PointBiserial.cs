using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Point-Biserial Correlation for feature selection with binary target.
/// </summary>
/// <remarks>
/// <para>
/// Point-biserial correlation measures the relationship between a continuous feature
/// and a binary (dichotomous) target. It's mathematically equivalent to Pearson
/// correlation when one variable is binary.
/// </para>
/// <para><b>For Beginners:</b> When your target is binary (like pass/fail, yes/no),
/// point-biserial correlation tells you how much a continuous feature differs between
/// the two groups. If the feature values are very different for class 0 vs class 1,
/// the correlation is high and the feature is useful for classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PointBiserial<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minCorrelation;

    private double[]? _correlations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Correlations => _correlations;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PointBiserial(
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
            "PointBiserial requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Separate samples by class (binary)
        var class0 = new List<int>();
        var class1 = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                class0.Add(i);
            else
                class1.Add(i);
        }

        if (class0.Count == 0 || class1.Count == 0)
            throw new ArgumentException("Both classes must have at least one sample.");

        int n0 = class0.Count;
        int n1 = class1.Count;
        double pProp = (double)n1 / n;
        double qProp = (double)n0 / n;

        _correlations = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute overall mean and standard deviation
            double overallMean = 0;
            for (int i = 0; i < n; i++)
                overallMean += NumOps.ToDouble(data[i, j]);
            overallMean /= n;

            double overallVar = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - overallMean;
                overallVar += diff * diff;
            }
            double overallStd = Math.Sqrt(overallVar / n);

            if (overallStd < 1e-10)
            {
                _correlations[j] = 0;
                continue;
            }

            // Compute means for each class
            double mean0 = 0, mean1 = 0;
            foreach (int i in class0)
                mean0 += NumOps.ToDouble(data[i, j]);
            mean0 /= n0;

            foreach (int i in class1)
                mean1 += NumOps.ToDouble(data[i, j]);
            mean1 /= n1;

            // Point-biserial correlation
            _correlations[j] = ((mean1 - mean0) / overallStd) * Math.Sqrt(pProp * qProp);
        }

        // Select features above threshold or top by absolute correlation
        var candidates = new List<int>();
        for (int j = 0; j < p; j++)
            if (Math.Abs(_correlations[j]) >= _minCorrelation)
                candidates.Add(j);

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = candidates
                .OrderByDescending(j => Math.Abs(_correlations[j]))
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = _correlations
                .Select((c, idx) => (Corr: Math.Abs(c), Index: idx))
                .OrderByDescending(x => x.Corr)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
