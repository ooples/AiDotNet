using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Distribution;

/// <summary>
/// Modality based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the estimated number of modes (peaks) in their
/// distributions, using a histogram-based peak detection approach.
/// </para>
/// <para><b>For Beginners:</b> Modality refers to how many peaks a distribution has.
/// Unimodal = 1 peak, bimodal = 2 peaks, multimodal = many peaks. Features with
/// multiple modes often indicate natural groupings or distinct populations in the data.
/// </para>
/// </remarks>
public class ModalitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly int _targetModality;

    private int[]? _modalityCounts;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public int TargetModality => _targetModality;
    public int[]? ModalityCounts => _modalityCounts;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ModalitySelector(
        int nFeaturesToSelect = 10,
        int nBins = 30,
        int targetModality = 2,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 3)
            throw new ArgumentException("Number of bins must be at least 3.", nameof(nBins));
        if (targetModality < 1)
            throw new ArgumentException("Target modality must be at least 1.", nameof(targetModality));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _targetModality = targetModality;
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

        _modalityCounts = new int[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double min = col.Min();
            double max = col.Max();
            double range = max - min;

            if (range < 1e-10)
            {
                _modalityCounts[j] = 1; // Constant = unimodal
                continue;
            }

            // Create histogram
            var binCounts = new double[_nBins];
            foreach (var val in col)
            {
                int bin = (int)((val - min) / range * (_nBins - 1));
                bin = Math.Min(bin, _nBins - 1);
                binCounts[bin]++;
            }

            // Smooth histogram with simple moving average
            var smoothed = new double[_nBins];
            for (int b = 0; b < _nBins; b++)
            {
                int count = 0;
                double sum = 0;
                for (int k = Math.Max(0, b - 1); k <= Math.Min(_nBins - 1, b + 1); k++)
                {
                    sum += binCounts[k];
                    count++;
                }
                smoothed[b] = sum / count;
            }

            // Count peaks (local maxima)
            int peaks = 0;
            for (int b = 1; b < _nBins - 1; b++)
            {
                if (smoothed[b] > smoothed[b - 1] && smoothed[b] > smoothed[b + 1] && smoothed[b] > 0)
                {
                    peaks++;
                }
            }

            // Handle edge cases
            if (peaks == 0 && smoothed.Max() > 0)
                peaks = 1;

            _modalityCounts[j] = peaks;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // Select features closest to target modality
        _selectedIndices = Enumerable.Range(0, p)
            .OrderBy(j => Math.Abs(_modalityCounts[j] - _targetModality))
            .ThenByDescending(j => _modalityCounts[j]) // Tie-break: prefer higher modality
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
            throw new InvalidOperationException("ModalitySelector has not been fitted.");

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
        throw new NotSupportedException("ModalitySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ModalitySelector has not been fitted.");

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
