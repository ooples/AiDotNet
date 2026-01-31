using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Incremental;

/// <summary>
/// Adaptive Feature Selector that adjusts selection based on performance.
/// </summary>
/// <remarks>
/// <para>
/// This selector adapts its feature selection over time based on observed
/// performance. Features that consistently perform well are retained, while
/// poor performers are replaced with alternatives.
/// </para>
/// <para><b>For Beginners:</b> Think of this like a sports team manager who
/// keeps track of how well each player performs. Players who consistently
/// do well stay on the team, while underperformers get replaced. Over time,
/// this builds an optimal team (set of features).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AdaptiveFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _adaptationRate;
    private readonly int _minSamplesForAdaptation;

    private double[]? _performanceScores;
    private int _sampleCount;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double AdaptationRate => _adaptationRate;
    public double[]? PerformanceScores => _performanceScores;
    public int SampleCount => _sampleCount;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AdaptiveFeatureSelector(
        int nFeaturesToSelect = 10,
        double adaptationRate = 0.1,
        int minSamplesForAdaptation = 50,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (adaptationRate <= 0 || adaptationRate > 1)
            throw new ArgumentException("Adaptation rate must be between 0 and 1.", nameof(adaptationRate));

        _nFeaturesToSelect = nFeaturesToSelect;
        _adaptationRate = adaptationRate;
        _minSamplesForAdaptation = minSamplesForAdaptation;
        _sampleCount = 0;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AdaptiveFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        Reset();
        PartialFit(data, target);
    }

    public void PartialFit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        int n = data.Rows;
        int p = data.Columns;

        // Initialize on first call
        if (_performanceScores is null)
        {
            _nInputFeatures = p;
            _performanceScores = new double[p];
        }

        // Compute performance scores for this batch
        var batchScores = ComputeBatchScores(data, target, n, p);

        // Update performance scores with exponential moving average
        for (int j = 0; j < p; j++)
        {
            if (_sampleCount == 0)
                _performanceScores[j] = batchScores[j];
            else
                _performanceScores[j] = (1 - _adaptationRate) * _performanceScores[j] + _adaptationRate * batchScores[j];
        }

        _sampleCount += n;

        // Update selection
        UpdateSelection(p);
    }

    private double[] ComputeBatchScores(Matrix<T> data, Vector<T> target, int n, int p)
    {
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

            scores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        return scores;
    }

    private void UpdateSelection(int p)
    {
        if (_performanceScores is null)
            return;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _performanceScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public void Reset()
    {
        _performanceScores = null;
        _sampleCount = 0;
        _selectedIndices = null;
        IsFitted = false;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AdaptiveFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("AdaptiveFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AdaptiveFeatureSelector has not been fitted.");

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
