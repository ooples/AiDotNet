using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Bootstrap Feature Selection for robust feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Bootstrap Feature Selection uses bootstrap sampling (sampling with replacement)
/// to estimate the stability of feature selection. Features that are consistently
/// selected across many bootstrap samples are more reliable.
/// </para>
/// <para><b>For Beginners:</b> Bootstrap is like asking the same question many
/// times with slightly different data each time. If a feature is selected most
/// of the time, it's probably genuinely important. If it only gets selected
/// sometimes, it might just be noise.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BootstrapFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBootstrapSamples;
    private readonly double _confidenceLevel;
    private readonly int? _randomState;

    private double[]? _selectionProbabilities;
    private double[]? _confidenceIntervals;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBootstrapSamples => _nBootstrapSamples;
    public double ConfidenceLevel => _confidenceLevel;
    public double[]? SelectionProbabilities => _selectionProbabilities;
    public double[]? ConfidenceIntervals => _confidenceIntervals;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BootstrapFS(
        int nFeaturesToSelect = 10,
        int nBootstrapSamples = 100,
        double confidenceLevel = 0.95,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBootstrapSamples < 1)
            throw new ArgumentException("Number of bootstrap samples must be at least 1.", nameof(nBootstrapSamples));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBootstrapSamples = nBootstrapSamples;
        _confidenceLevel = confidenceLevel;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BootstrapFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var allScores = new List<double[]>();
        var selectionCounts = new int[p];

        for (int b = 0; b < _nBootstrapSamples; b++)
        {
            // Create bootstrap sample (with replacement)
            var sampleIndices = new int[n];
            for (int i = 0; i < n; i++)
                sampleIndices[i] = random.Next(n);

            // Compute feature scores on bootstrap sample
            var scores = ComputeScores(data, target, sampleIndices, p);
            allScores.Add(scores);

            // Track which features would be selected
            var topFeatures = scores
                .Select((s, idx) => (Score: s, Index: idx))
                .OrderByDescending(x => x.Score)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index);

            foreach (int j in topFeatures)
                selectionCounts[j]++;
        }

        // Compute selection probabilities
        _selectionProbabilities = selectionCounts.Select(c => (double)c / _nBootstrapSamples).ToArray();

        // Compute confidence intervals for feature importance
        _confidenceIntervals = new double[p];
        double alpha = 1 - _confidenceLevel;
        int lowerIdx = (int)(alpha / 2 * _nBootstrapSamples);
        int upperIdx = (int)((1 - alpha / 2) * _nBootstrapSamples);

        for (int j = 0; j < p; j++)
        {
            var sortedScores = allScores.Select(s => s[j]).OrderBy(x => x).ToArray();
            double lower = sortedScores[Math.Max(0, lowerIdx)];
            double upper = sortedScores[Math.Min(sortedScores.Length - 1, upperIdx)];
            _confidenceIntervals[j] = upper - lower;
        }

        // Select features by probability (most consistently selected)
        _selectedIndices = _selectionProbabilities
            .Select((prob, idx) => (Prob: prob, Index: idx))
            .OrderByDescending(x => x.Prob)
            .Take(_nFeaturesToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeScores(Matrix<T> data, Vector<T> target, int[] sampleIndices, int p)
    {
        int sampleSize = sampleIndices.Length;
        var scores = new double[p];

        double yMean = sampleIndices.Sum(i => NumOps.ToDouble(target[i])) / sampleSize;

        for (int j = 0; j < p; j++)
        {
            double xMean = sampleIndices.Sum(i => NumOps.ToDouble(data[i, j])) / sampleSize;

            double sxy = 0, sxx = 0, syy = 0;
            foreach (int i in sampleIndices)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            scores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        return scores;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapFS has not been fitted.");

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
        throw new NotSupportedException("BootstrapFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapFS has not been fitted.");

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
