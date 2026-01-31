using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Incremental;

/// <summary>
/// Streaming Feature Selector for online/incremental learning.
/// </summary>
/// <remarks>
/// <para>
/// Maintains running statistics to update feature scores as new data arrives.
/// This enables feature selection on streaming data without storing all
/// historical samples.
/// </para>
/// <para><b>For Beginners:</b> When you can't store all your data (it's too big
/// or comes continuously), you need to update feature scores incrementally.
/// This method keeps running averages that update with each new sample,
/// allowing feature selection on endless data streams.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StreamingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _decayFactor;

    private double[]? _runningMeanX;
    private double[]? _runningMeanY;
    private double[]? _runningMeanXY;
    private double[]? _runningMeanX2;
    private double _runningMeanY2;
    private int _sampleCount;
    private double[]? _correlationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double DecayFactor => _decayFactor;
    public double[]? CorrelationScores => _correlationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public int SampleCount => _sampleCount;
    public override bool SupportsInverseTransform => false;

    public StreamingFeatureSelector(
        int nFeaturesToSelect = 10,
        double decayFactor = 0.99,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (decayFactor <= 0 || decayFactor > 1)
            throw new ArgumentException("Decay factor must be between 0 and 1.", nameof(decayFactor));

        _nFeaturesToSelect = nFeaturesToSelect;
        _decayFactor = decayFactor;
        _sampleCount = 0;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StreamingFeatureSelector requires target values. Use PartialFit or Fit with target.");
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
        if (_runningMeanX is null)
        {
            _nInputFeatures = p;
            _runningMeanX = new double[p];
            _runningMeanY = new double[p];
            _runningMeanXY = new double[p];
            _runningMeanX2 = new double[p];
            _runningMeanY2 = 0;
            _correlationScores = new double[p];
        }

        // Update running statistics with exponential decay
        // Local references to avoid null checks in tight loop
        var runningMeanX = _runningMeanX;
        var runningMeanY = _runningMeanY;
        var runningMeanXY = _runningMeanXY;
        var runningMeanX2 = _runningMeanX2;

        if (runningMeanX is null || runningMeanY is null || runningMeanXY is null || runningMeanX2 is null)
            throw new InvalidOperationException("Running statistics arrays not initialized.");

        for (int i = 0; i < n; i++)
        {
            double y = NumOps.ToDouble(target[i]);
            _runningMeanY2 = _decayFactor * _runningMeanY2 + (1 - _decayFactor) * y * y;

            for (int j = 0; j < p; j++)
            {
                double x = NumOps.ToDouble(data[i, j]);

                runningMeanX[j] = _decayFactor * runningMeanX[j] + (1 - _decayFactor) * x;
                runningMeanY[j] = _decayFactor * runningMeanY[j] + (1 - _decayFactor) * y;
                runningMeanXY[j] = _decayFactor * runningMeanXY[j] + (1 - _decayFactor) * x * y;
                runningMeanX2[j] = _decayFactor * runningMeanX2[j] + (1 - _decayFactor) * x * x;
            }

            _sampleCount++;
        }

        UpdateScores(p);
    }

    private void UpdateScores(int p)
    {
        if (_runningMeanX is null || _runningMeanY is null ||
            _runningMeanXY is null || _runningMeanX2 is null ||
            _correlationScores is null)
            return;

        for (int j = 0; j < p; j++)
        {
            double varX = _runningMeanX2[j] - _runningMeanX[j] * _runningMeanX[j];
            double varY = _runningMeanY2 - _runningMeanY[j] * _runningMeanY[j];
            double covXY = _runningMeanXY[j] - _runningMeanX[j] * _runningMeanY[j];

            if (varX > 1e-10 && varY > 1e-10)
                _correlationScores[j] = Math.Abs(covXY / Math.Sqrt(varX * varY));
            else
                _correlationScores[j] = 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _correlationScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public void Reset()
    {
        _runningMeanX = null;
        _runningMeanY = null;
        _runningMeanXY = null;
        _runningMeanX2 = null;
        _runningMeanY2 = 0;
        _sampleCount = 0;
        _correlationScores = null;
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
            throw new InvalidOperationException("StreamingFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("StreamingFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StreamingFeatureSelector has not been fitted.");

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
