using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Morphological;

/// <summary>
/// Erosion/Dilation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on morphological properties, measuring how feature values
/// respond to erosion and dilation operations (local minimum/maximum filtering).
/// </para>
/// <para><b>For Beginners:</b> Morphological operations like erosion (shrinking)
/// and dilation (expanding) reveal structural properties of data. Features that
/// change significantly under these operations may capture important local patterns.
/// </para>
/// </remarks>
public class ErosionDilationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _windowSize;

    private double[]? _morphologicalScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int WindowSize => _windowSize;
    public double[]? MorphologicalScores => _morphologicalScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ErosionDilationSelector(
        int nFeaturesToSelect = 10,
        int windowSize = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _windowSize = windowSize;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        _morphologicalScores = new double[p];
        int halfWindow = _windowSize / 2;

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Apply erosion (local minimum)
            var eroded = new double[n];
            for (int i = 0; i < n; i++)
            {
                double minVal = col[i];
                for (int k = Math.Max(0, i - halfWindow); k <= Math.Min(n - 1, i + halfWindow); k++)
                    minVal = Math.Min(minVal, col[k]);
                eroded[i] = minVal;
            }

            // Apply dilation (local maximum)
            var dilated = new double[n];
            for (int i = 0; i < n; i++)
            {
                double maxVal = col[i];
                for (int k = Math.Max(0, i - halfWindow); k <= Math.Min(n - 1, i + halfWindow); k++)
                    maxVal = Math.Max(maxVal, col[k]);
                dilated[i] = maxVal;
            }

            // Morphological gradient (dilation - erosion)
            double gradientMean = 0;
            for (int i = 0; i < n; i++)
                gradientMean += dilated[i] - eroded[i];
            gradientMean /= n;

            // Features with larger morphological gradients have more local variation
            _morphologicalScores[j] = gradientMean;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _morphologicalScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ErosionDilationSelector has not been fitted.");

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
        throw new NotSupportedException("ErosionDilationSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ErosionDilationSelector has not been fitted.");

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
