using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Interquartile Range (IQR) for robust unsupervised feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The interquartile range is the difference between the 75th and 25th percentiles.
/// It measures the spread of the middle 50% of data and is extremely robust to outliers.
/// </para>
/// <para><b>For Beginners:</b> IQR focuses on the "typical" range of values, ignoring
/// the extreme high and low values. A feature with small IQR has most values bunched
/// together and may not be discriminative. IQR is great for data with outliers because
/// it completely ignores them.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class InterquartileRange<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minIQR;

    private double[]? _iqrValues;
    private double[]? _q1Values;
    private double[]? _q3Values;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? IQRValues => _iqrValues;
    public double[]? Q1Values => _q1Values;
    public double[]? Q3Values => _q3Values;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public InterquartileRange(
        int nFeaturesToSelect = 10,
        double minIQR = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minIQR = minIQR;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _iqrValues = new double[p];
        _q1Values = new double[p];
        _q3Values = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Extract and sort column values
            var values = new double[n];
            for (int i = 0; i < n; i++)
                values[i] = NumOps.ToDouble(data[i, j]);
            Array.Sort(values);

            // Compute Q1 and Q3 using linear interpolation
            _q1Values[j] = Percentile(values, 0.25);
            _q3Values[j] = Percentile(values, 0.75);
            _iqrValues[j] = _q3Values[j] - _q1Values[j];
        }

        // Select features above threshold or top by IQR
        var candidates = new List<int>();
        for (int j = 0; j < p; j++)
            if (_iqrValues[j] >= _minIQR)
                candidates.Add(j);

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = candidates
                .OrderByDescending(j => _iqrValues[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = _iqrValues
                .Select((iqr, idx) => (IQR: iqr, Index: idx))
                .OrderByDescending(x => x.IQR)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double Percentile(double[] sortedValues, double percentile)
    {
        int n = sortedValues.Length;
        double index = percentile * (n - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (lower == upper)
            return sortedValues[lower];

        double fraction = index - lower;
        return sortedValues[lower] * (1 - fraction) + sortedValues[upper] * fraction;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InterquartileRange has not been fitted.");

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
        throw new NotSupportedException("InterquartileRange does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InterquartileRange has not been fitted.");

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
