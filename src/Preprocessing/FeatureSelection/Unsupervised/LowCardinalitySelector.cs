using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Low Cardinality Feature Removal.
/// </summary>
/// <remarks>
/// <para>
/// Removes features with very low cardinality (few unique values) as they
/// may not provide enough discriminative power.
/// </para>
/// <para><b>For Beginners:</b> Cardinality is the number of unique values a
/// feature has. A feature with only 1 or 2 unique values doesn't vary much
/// and may not be useful for distinguishing between samples. This selector
/// removes such low-variety features.
/// </para>
/// </remarks>
public class LowCardinalitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _minCardinality;

    private int[]? _cardinalities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MinCardinality => _minCardinality;
    public int[]? Cardinalities => _cardinalities;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LowCardinalitySelector(
        int minCardinality = 2,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (minCardinality < 1)
            throw new ArgumentException("Minimum cardinality must be at least 1.", nameof(minCardinality));

        _minCardinality = minCardinality;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _cardinalities = new int[p];
        var selected = new List<int>();

        for (int j = 0; j < p; j++)
        {
            var uniqueValues = new HashSet<double>();
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (!double.IsNaN(val))
                    uniqueValues.Add(Math.Round(val, 10)); // Round to handle floating point
            }

            _cardinalities[j] = uniqueValues.Count;

            if (_cardinalities[j] >= _minCardinality)
                selected.Add(j);
        }

        _selectedIndices = selected.ToArray();
        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LowCardinalitySelector has not been fitted.");

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
        throw new NotSupportedException("LowCardinalitySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LowCardinalitySelector has not been fitted.");

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
