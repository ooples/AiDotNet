using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Mean Absolute Deviation (MAD) for robust unsupervised feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Mean Absolute Deviation measures dispersion as the average distance from the mean.
/// Unlike variance/standard deviation, MAD is more robust to outliers because it
/// doesn't square the deviations.
/// </para>
/// <para><b>For Beginners:</b> MAD tells you on average how far each value is from
/// the mean. Features with low MAD are nearly constant and probably not useful.
/// Unlike standard deviation, MAD isn't overly influenced by extreme outliers,
/// making it a safer choice for messy data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MeanAbsoluteDeviation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minMAD;
    private readonly bool _useMedian;

    private double[]? _madValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? MADValues => _madValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MeanAbsoluteDeviation(
        int nFeaturesToSelect = 10,
        double minMAD = 0.0,
        bool useMedian = false,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minMAD = minMAD;
        _useMedian = useMedian;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _madValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Extract column values
            var values = new double[n];
            for (int i = 0; i < n; i++)
                values[i] = NumOps.ToDouble(data[i, j]);

            double center;
            if (_useMedian)
            {
                // Use median as center (more robust)
                Array.Sort(values);
                center = n % 2 == 0
                    ? (values[n / 2 - 1] + values[n / 2]) / 2
                    : values[n / 2];
            }
            else
            {
                // Use mean as center
                center = values.Average();
            }

            // Compute mean absolute deviation
            double sumAbsDev = 0;
            for (int i = 0; i < n; i++)
                sumAbsDev += Math.Abs(values[i] - center);

            _madValues[j] = sumAbsDev / n;
        }

        // Select features above threshold or top by MAD
        var candidates = new List<int>();
        for (int j = 0; j < p; j++)
            if (_madValues[j] >= _minMAD)
                candidates.Add(j);

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = candidates
                .OrderByDescending(j => _madValues[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = _madValues
                .Select((mad, idx) => (MAD: mad, Index: idx))
                .OrderByDescending(x => x.MAD)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MeanAbsoluteDeviation has not been fitted.");

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
        throw new NotSupportedException("MeanAbsoluteDeviation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MeanAbsoluteDeviation has not been fitted.");

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
