using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.NLP;

/// <summary>
/// Document Frequency based feature selection for text data.
/// </summary>
/// <remarks>
/// <para>
/// Document Frequency (DF) selection filters terms based on how many documents
/// they appear in. Terms appearing in too few or too many documents are removed,
/// as they are either too rare to be useful or too common to be discriminative.
/// </para>
/// <para><b>For Beginners:</b> Some words appear in almost every document (like "is" or "the")
/// and don't help distinguish documents. Other words appear only once or twice in the
/// entire corpus and might be typos or too specific. DF filtering keeps words that appear
/// in a reasonable number of documents - common enough to be meaningful but not so common
/// that everyone uses them.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DocumentFrequencyFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _minDf;
    private readonly int _maxDf;

    private int[]? _documentFrequencies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MinDf => _minDf;
    public int MaxDf => _maxDf;
    public int[]? DocumentFrequencies => _documentFrequencies;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DocumentFrequencyFS(
        int nFeaturesToSelect = 10,
        int minDf = 1,
        int maxDf = int.MaxValue,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minDf < 0)
            throw new ArgumentException("Min document frequency must be non-negative.", nameof(minDf));
        if (maxDf < minDf)
            throw new ArgumentException("Max document frequency cannot be less than min.", nameof(maxDf));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minDf = minDf;
        _maxDf = maxDf;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute document frequency for each term
        _documentFrequencies = new int[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
            {
                if (NumOps.ToDouble(data[i, j]) > 0)
                    _documentFrequencies[j]++;
            }
        }

        // Filter by document frequency and select top features
        var candidates = Enumerable.Range(0, p)
            .Where(j => _documentFrequencies[j] >= _minDf && _documentFrequencies[j] <= _maxDf)
            .OrderByDescending(j => _documentFrequencies[j])
            .ToList();

        int numToSelect = Math.Min(_nFeaturesToSelect, candidates.Count);
        if (numToSelect == 0 && p > 0)
        {
            // No features pass filter, select features closest to criteria
            _selectedIndices = Enumerable.Range(0, p)
                .Where(j => _documentFrequencies[j] >= _minDf)
                .OrderByDescending(j => _documentFrequencies[j])
                .Take(Math.Min(_nFeaturesToSelect, p))
                .OrderBy(x => x)
                .ToArray();

            if (_selectedIndices.Length == 0)
            {
                _selectedIndices = Enumerable.Range(0, p)
                    .OrderByDescending(j => _documentFrequencies[j])
                    .Take(Math.Min(_nFeaturesToSelect, p))
                    .OrderBy(x => x)
                    .ToArray();
            }
        }
        else
        {
            _selectedIndices = candidates
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DocumentFrequencyFS has not been fitted.");

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
        throw new NotSupportedException("DocumentFrequencyFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DocumentFrequencyFS has not been fitted.");

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
