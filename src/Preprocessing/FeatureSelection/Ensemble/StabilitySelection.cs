using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Stability Selection for robust feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Stability Selection runs feature selection multiple times on random subsamples
/// of the data and selects features that are consistently chosen across iterations.
/// This reduces sensitivity to noise and produces more reliable feature sets.
/// </para>
/// <para><b>For Beginners:</b> A single run of feature selection might pick some features
/// by chance. Stability Selection runs many rounds on different parts of the data and
/// counts how often each feature is selected. Features that are picked consistently
/// (say, 70% of the time) are more trustworthy than those picked rarely.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StabilitySelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly double _sampleFraction;
    private readonly double _threshold;
    private readonly Func<Matrix<T>, Vector<T>, int, int[]>? _baseSelector;
    private readonly int? _randomState;

    private double[]? _selectionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NIterations => _nIterations;
    public double SampleFraction => _sampleFraction;
    public double Threshold => _threshold;
    public double[]? SelectionProbabilities => _selectionProbabilities;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public StabilitySelection(
        int nFeaturesToSelect = 10,
        int nIterations = 100,
        double sampleFraction = 0.5,
        double threshold = 0.6,
        Func<Matrix<T>, Vector<T>, int, int[]>? baseSelector = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (sampleFraction <= 0 || sampleFraction > 1)
            throw new ArgumentException("Sample fraction must be between 0 and 1.", nameof(sampleFraction));
        if (threshold <= 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _sampleFraction = sampleFraction;
        _threshold = threshold;
        _baseSelector = baseSelector;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StabilitySelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var baseSelector = _baseSelector ?? DefaultSelector;

        // Count selections across iterations
        var selectionCounts = new int[p];
        int sampleSize = Math.Max(1, (int)(n * _sampleFraction));

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Random subsample
            var indices = Enumerable.Range(0, n)
                .OrderBy(_ => random.Next())
                .Take(sampleSize)
                .ToArray();

            // Extract subsample
            var subData = new T[sampleSize, p];
            var subTarget = new T[sampleSize];
            for (int i = 0; i < sampleSize; i++)
            {
                for (int j = 0; j < p; j++)
                    subData[i, j] = data[indices[i], j];
                subTarget[i] = target[indices[i]];
            }

            var subDataMatrix = new Matrix<T>(subData);
            var subTargetVector = new Vector<T>(subTarget);

            // Run base selector
            int numBase = Math.Min(_nFeaturesToSelect, p);
            var selected = baseSelector(subDataMatrix, subTargetVector, numBase);

            foreach (int idx in selected)
                selectionCounts[idx]++;
        }

        // Compute selection probabilities
        _selectionProbabilities = new double[p];
        for (int j = 0; j < p; j++)
            _selectionProbabilities[j] = (double)selectionCounts[j] / _nIterations;

        // Select features above threshold
        var candidates = Enumerable.Range(0, p)
            .Where(j => _selectionProbabilities[j] >= _threshold)
            .OrderByDescending(j => _selectionProbabilities[j])
            .ToList();

        int numToSelect = Math.Min(_nFeaturesToSelect, candidates.Count);
        if (numToSelect == 0)
        {
            // No features pass threshold, select top by probability
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _selectionProbabilities[j])
                .Take(Math.Min(_nFeaturesToSelect, p))
                .OrderBy(x => x)
                .ToArray();
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

    private int[] DefaultSelector(Matrix<T> data, Vector<T> target, int k)
    {
        // Simple correlation-based selection
        int n = data.Rows;
        int p = data.Columns;

        var scores = new double[p];

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < p; j++)
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

            double corr = (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
            scores[j] = Math.Abs(corr);
        }

        return scores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(k)
            .Select(x => x.Index)
            .ToArray();
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StabilitySelection has not been fitted.");

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
        throw new NotSupportedException("StabilitySelection does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StabilitySelection has not been fitted.");

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
