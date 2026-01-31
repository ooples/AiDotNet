using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Morphological;

/// <summary>
/// Granulometry based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on granulometric analysis, measuring how features respond
/// to morphological openings of increasing size.
/// </para>
/// <para><b>For Beginners:</b> Granulometry measures the size distribution of structures
/// in data by applying progressively larger opening operations. Features with interesting
/// size distributions (neither too smooth nor too noisy) are selected.
/// </para>
/// </remarks>
public class GranulometrySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxWindowSize;

    private double[]? _granulometryScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxWindowSize => _maxWindowSize;
    public double[]? GranulometryScores => _granulometryScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GranulometrySelector(
        int nFeaturesToSelect = 10,
        int maxWindowSize = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxWindowSize = maxWindowSize;
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

        _granulometryScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Normalize to [0, 1]
            double minVal = col.Min();
            double maxVal = col.Max();
            double range = maxVal - minVal;
            if (range > 1e-10)
                for (int i = 0; i < n; i++)
                    col[i] = (col[i] - minVal) / range;

            // Compute pattern spectrum (granulometry curve)
            var patternSpectrum = new double[_maxWindowSize];
            double prevSum = col.Sum();

            for (int ws = 1; ws <= _maxWindowSize; ws++)
            {
                // Opening: erosion followed by dilation
                var opened = ApplyOpening(col, ws);
                double currSum = opened.Sum();
                patternSpectrum[ws - 1] = prevSum - currSum;
                prevSum = currSum;
            }

            // Score: entropy of pattern spectrum (diverse size distribution is interesting)
            double total = patternSpectrum.Sum();
            if (total > 1e-10)
            {
                double entropy = 0;
                for (int ws = 0; ws < _maxWindowSize; ws++)
                {
                    double prob = patternSpectrum[ws] / total;
                    if (prob > 1e-10)
                        entropy -= prob * Math.Log(prob) / Math.Log(2);
                }
                _granulometryScores[j] = entropy;
            }
            else
            {
                _granulometryScores[j] = 0;
            }
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _granulometryScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ApplyOpening(double[] signal, int windowSize)
    {
        int n = signal.Length;
        int halfWindow = windowSize / 2;

        // Erosion (local minimum)
        var eroded = new double[n];
        for (int i = 0; i < n; i++)
        {
            double minVal = signal[i];
            for (int k = Math.Max(0, i - halfWindow); k <= Math.Min(n - 1, i + halfWindow); k++)
                minVal = Math.Min(minVal, signal[k]);
            eroded[i] = minVal;
        }

        // Dilation (local maximum)
        var opened = new double[n];
        for (int i = 0; i < n; i++)
        {
            double maxVal = eroded[i];
            for (int k = Math.Max(0, i - halfWindow); k <= Math.Min(n - 1, i + halfWindow); k++)
                maxVal = Math.Max(maxVal, eroded[k]);
            opened[i] = maxVal;
        }

        return opened;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GranulometrySelector has not been fitted.");

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
        throw new NotSupportedException("GranulometrySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GranulometrySelector has not been fitted.");

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
