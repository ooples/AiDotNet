using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Text;

/// <summary>
/// TF-IDF based feature selection for text/document data.
/// </summary>
/// <remarks>
/// <para>
/// Uses Term Frequency-Inverse Document Frequency to select terms that are
/// important in some documents but not ubiquitous across all documents.
/// </para>
/// <para><b>For Beginners:</b> TF-IDF finds words that are frequent in specific
/// documents but rare overall. These terms help distinguish documents and
/// are better features than common words appearing everywhere.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TFIDFSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minDF;
    private readonly double _maxDF;

    private double[]? _idfScores;
    private double[]? _maxTFIDF;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? IDFScores => _idfScores;
    public double[]? MaxTFIDF => _maxTFIDF;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TFIDFSelector(
        int nFeaturesToSelect = 100,
        double minDF = 0.01,
        double maxDF = 0.95,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minDF < 0 || minDF > 1)
            throw new ArgumentException("minDF must be between 0 and 1.", nameof(minDF));
        if (maxDF < minDF || maxDF > 1)
            throw new ArgumentException("maxDF must be between minDF and 1.", nameof(maxDF));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minDF = minDF;
        _maxDF = maxDF;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;  // Documents
        int p = data.Columns;  // Terms

        // Compute document frequency for each term
        var documentFrequency = new int[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
            {
                if (NumOps.ToDouble(data[i, j]) > 0)
                    documentFrequency[j]++;
            }
        }

        // Compute IDF
        _idfScores = new double[p];
        _maxTFIDF = new double[p];

        for (int j = 0; j < p; j++)
        {
            double df = (double)documentFrequency[j] / n;

            // Filter by document frequency
            if (df < _minDF || df > _maxDF)
            {
                _idfScores[j] = 0;
                _maxTFIDF[j] = 0;
                continue;
            }

            // IDF with smoothing
            _idfScores[j] = Math.Log((n + 1.0) / (documentFrequency[j] + 1.0)) + 1;

            // Compute max TF-IDF across all documents
            for (int i = 0; i < n; i++)
            {
                double tf = NumOps.ToDouble(data[i, j]);
                double tfidf = tf * _idfScores[j];
                if (tfidf > _maxTFIDF[j])
                    _maxTFIDF[j] = tfidf;
            }
        }

        // Select features by max TF-IDF score
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _maxTFIDF
            .Select((s, idx) => (Score: s, Index: idx))
            .Where(x => x.Score > 0)
            .OrderByDescending(x => x.Score)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        if (_selectedIndices.Length == 0)
        {
            // Fallback: select highest IDF terms
            _selectedIndices = _idfScores
                .Select((s, idx) => (Score: s, Index: idx))
                .OrderByDescending(x => x.Score)
                .Take(nToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
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
            throw new InvalidOperationException("TFIDFSelector has not been fitted.");

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
        throw new NotSupportedException("TFIDFSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TFIDFSelector has not been fitted.");

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
