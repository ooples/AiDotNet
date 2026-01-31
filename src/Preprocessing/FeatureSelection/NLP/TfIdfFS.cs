using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.NLP;

/// <summary>
/// TF-IDF based feature selection for text data.
/// </summary>
/// <remarks>
/// <para>
/// TF-IDF (Term Frequency-Inverse Document Frequency) based selection identifies
/// features (terms) that are both frequent within documents and discriminative
/// across the corpus. Features with high TF-IDF scores are informative for
/// distinguishing documents.
/// </para>
/// <para><b>For Beginners:</b> In text data, some words appear often but are
/// meaningless (like "the"), while others are rare but important. TF-IDF balances
/// term frequency (how often a word appears in a document) with inverse document
/// frequency (how rare it is across all documents). Words that are common in some
/// documents but rare overall get high scores.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TfIdfFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minDf;
    private readonly double _maxDf;

    private double[]? _tfidfScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double MinDf => _minDf;
    public double MaxDf => _maxDf;
    public double[]? TfIdfScores => _tfidfScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TfIdfFS(
        int nFeaturesToSelect = 10,
        double minDf = 0.0,
        double maxDf = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minDf < 0 || minDf > 1)
            throw new ArgumentException("Min document frequency must be between 0 and 1.", nameof(minDf));
        if (maxDf < 0 || maxDf > 1)
            throw new ArgumentException("Max document frequency must be between 0 and 1.", nameof(maxDf));
        if (minDf > maxDf)
            throw new ArgumentException("Min document frequency cannot exceed max document frequency.");

        _nFeaturesToSelect = nFeaturesToSelect;
        _minDf = minDf;
        _maxDf = maxDf;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows; // Number of documents
        int p = data.Columns; // Number of terms

        // Compute document frequency for each term
        var docFreq = new int[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
            {
                if (NumOps.ToDouble(data[i, j]) > 0)
                    docFreq[j]++;
            }
        }

        // Compute IDF
        var idf = new double[p];
        for (int j = 0; j < p; j++)
        {
            if (docFreq[j] > 0)
                idf[j] = Math.Log((double)n / docFreq[j]) + 1; // Smoothed IDF
            else
                idf[j] = 0;
        }

        // Compute average TF-IDF score for each feature
        _tfidfScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            double docFreqRatio = (double)docFreq[j] / n;

            // Filter by document frequency
            if (docFreqRatio < _minDf || docFreqRatio > _maxDf)
            {
                _tfidfScores[j] = 0;
                continue;
            }

            double totalTfidf = 0;
            for (int i = 0; i < n; i++)
            {
                double tf = NumOps.ToDouble(data[i, j]);
                totalTfidf += tf * idf[j];
            }
            _tfidfScores[j] = totalTfidf / n;
        }

        // Select top features by TF-IDF score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _tfidfScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TfIdfFS has not been fitted.");

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
        throw new NotSupportedException("TfIdfFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TfIdfFS has not been fitted.");

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
