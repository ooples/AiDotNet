using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.DomainSpecific;

/// <summary>
/// Feature selection for text/NLP data.
/// </summary>
/// <remarks>
/// <para>
/// TextFeatureSelector is designed for selecting features from text data,
/// such as word frequencies, TF-IDF scores, or word embeddings. It considers
/// text-specific properties like term frequency and document frequency.
/// </para>
/// <para><b>For Beginners:</b> Text data has unique properties - some words are
/// common but uninformative (like "the"), while rare words might be very
/// distinctive. This selector picks words/features that best distinguish
/// between different text categories.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TextFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minDocumentFrequency;
    private readonly double _maxDocumentFrequency;
    private readonly bool _useChiSquare;

    private double[]? _featureScores;
    private double[]? _documentFrequencies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double MinDocumentFrequency => _minDocumentFrequency;
    public double MaxDocumentFrequency => _maxDocumentFrequency;
    public bool UseChiSquare => _useChiSquare;
    public double[]? FeatureScores => _featureScores;
    public double[]? DocumentFrequencies => _documentFrequencies;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TextFeatureSelector(
        int nFeaturesToSelect = 10,
        double minDocumentFrequency = 0.01,
        double maxDocumentFrequency = 0.95,
        bool useChiSquare = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minDocumentFrequency < 0 || minDocumentFrequency > 1)
            throw new ArgumentException("Min document frequency must be between 0 and 1.", nameof(minDocumentFrequency));
        if (maxDocumentFrequency < 0 || maxDocumentFrequency > 1)
            throw new ArgumentException("Max document frequency must be between 0 and 1.", nameof(maxDocumentFrequency));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minDocumentFrequency = minDocumentFrequency;
        _maxDocumentFrequency = maxDocumentFrequency;
        _useChiSquare = useChiSquare;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TextFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute document frequencies
        _documentFrequencies = new double[p];
        for (int j = 0; j < p; j++)
        {
            int docCount = 0;
            for (int i = 0; i < n; i++)
            {
                if (NumOps.ToDouble(data[i, j]) > 0)
                    docCount++;
            }
            _documentFrequencies[j] = (double)docCount / n;
        }

        // Filter by document frequency
        var candidates = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (_documentFrequencies[j] >= _minDocumentFrequency &&
                _documentFrequencies[j] <= _maxDocumentFrequency)
            {
                candidates.Add(j);
            }
        }

        // Compute feature scores
        _featureScores = new double[p];

        if (_useChiSquare)
        {
            ComputeChiSquareScores(data, target, candidates);
        }
        else
        {
            ComputeCorrelationScores(data, target, candidates);
        }

        // Select top features from candidates
        int numToSelect = Math.Min(_nFeaturesToSelect, candidates.Count);
        _selectedIndices = candidates
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        // If not enough candidates, add from remaining features
        if (_selectedIndices.Length < _nFeaturesToSelect)
        {
            var remaining = Enumerable.Range(0, p)
                .Where(j => !candidates.Contains(j))
                .OrderByDescending(j => _featureScores[j])
                .Take(_nFeaturesToSelect - _selectedIndices.Length);

            _selectedIndices = _selectedIndices.Concat(remaining).OrderBy(x => x).ToArray();
        }

        IsFitted = true;
    }

    private void ComputeChiSquareScores(Matrix<T> data, Vector<T> target, List<int> candidates)
    {
        int n = data.Rows;

        // Get class counts
        var classCounts = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int classLabel = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classCounts.ContainsKey(classLabel))
                classCounts[classLabel] = 0;
            classCounts[classLabel]++;
        }

        foreach (int j in candidates)
        {
            // Build contingency table
            var featureClassCounts = new Dictionary<(int Feature, int Class), int>();

            for (int i = 0; i < n; i++)
            {
                int classLabel = (int)Math.Round(NumOps.ToDouble(target[i]));
                int featurePresent = NumOps.ToDouble(data[i, j]) > 0 ? 1 : 0;

                var key = (featurePresent, classLabel);
                if (!featureClassCounts.ContainsKey(key))
                    featureClassCounts[key] = 0;
                featureClassCounts[key]++;
            }

            // Compute chi-square statistic
            double chiSq = 0;
            int featureCount = 0;
            for (int i = 0; i < n; i++)
            {
                if (NumOps.ToDouble(data[i, j]) > 0)
                    featureCount++;
            }

            foreach (var classCount in classCounts)
            {
                int classLabel = classCount.Key;
                int classTotal = classCount.Value;

                for (int f = 0; f <= 1; f++)
                {
                    int fTotal = f == 1 ? featureCount : (n - featureCount);
                    double expected = (double)fTotal * classTotal / n;

                    if (expected > 0)
                    {
                        var key = (f, classLabel);
                        int observed = featureClassCounts.GetValueOrDefault(key, 0);
                        chiSq += (observed - expected) * (observed - expected) / expected;
                    }
                }
            }

            _featureScores![j] = chiSq;
        }
    }

    private void ComputeCorrelationScores(Matrix<T> data, Vector<T> target, List<int> candidates)
    {
        int n = data.Rows;

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        foreach (int j in candidates)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            _featureScores![j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TextFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("TextFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TextFeatureSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Term{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
