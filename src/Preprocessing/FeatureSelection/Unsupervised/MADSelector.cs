using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Median Absolute Deviation (MAD) based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses MAD as a robust measure of variability. Unlike variance, MAD is
/// resistant to outliers, making it better for selecting features in
/// datasets with extreme values.
/// </para>
/// <para><b>For Beginners:</b> Standard variance can be thrown off by outliers.
/// MAD uses the median instead of mean, making it more robust. Features with
/// higher MAD have more consistent spread in their values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MADSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _threshold;

    private double[]? _madScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Threshold => _threshold;
    public double[]? MADScores => _madScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MADSelector(
        int nFeaturesToSelect = -1,
        double threshold = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _nFeaturesToSelect = nFeaturesToSelect;
        _threshold = threshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _madScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Extract column values
            var values = new double[n];
            for (int i = 0; i < n; i++)
                values[i] = NumOps.ToDouble(data[i, j]);

            // Compute median
            Array.Sort(values);
            double median = n % 2 == 1
                ? values[n / 2]
                : (values[n / 2 - 1] + values[n / 2]) / 2;

            // Compute absolute deviations
            var absDeviations = new double[n];
            for (int i = 0; i < n; i++)
                absDeviations[i] = Math.Abs(values[i] - median);

            // Compute MAD (median of absolute deviations)
            Array.Sort(absDeviations);
            _madScores[j] = n % 2 == 1
                ? absDeviations[n / 2]
                : (absDeviations[n / 2 - 1] + absDeviations[n / 2]) / 2;
        }

        // Select features
        if (_nFeaturesToSelect > 0)
        {
            int nToSelect = Math.Min(_nFeaturesToSelect, p);
            _selectedIndices = _madScores
                .Select((mad, idx) => (MAD: mad, Index: idx))
                .OrderByDescending(x => x.MAD)
                .Take(nToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            // Use threshold
            _selectedIndices = _madScores
                .Select((mad, idx) => (MAD: mad, Index: idx))
                .Where(x => x.MAD > _threshold)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();

            if (_selectedIndices.Length == 0)
            {
                // Keep highest MAD
                int maxIdx = 0;
                for (int j = 1; j < p; j++)
                    if (_madScores[j] > _madScores[maxIdx])
                        maxIdx = j;
                _selectedIndices = new[] { maxIdx };
            }
        }

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        FitCore(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MADSelector has not been fitted.");

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
        throw new NotSupportedException("MADSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MADSelector has not been fitted.");

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
