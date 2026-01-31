using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Streaming;

/// <summary>
/// Online (streaming) feature selection for incremental learning scenarios.
/// </summary>
/// <remarks>
/// <para>
/// Maintains feature statistics incrementally, allowing feature selection decisions
/// to be updated as new data arrives without storing or reprocessing all data.
/// </para>
/// <para><b>For Beginners:</b> In streaming scenarios, data arrives continuously and
/// you can't store everything. Online feature selection keeps running statistics
/// about each feature and makes selection decisions on-the-fly, updating as new
/// data comes in.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class OnlineFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _decayFactor;

    private int _sampleCount;
    private double[]? _runningMeans;
    private double[]? _runningVariances;
    private double[]? _runningCorrelations;
    private double _targetMean;
    private double _targetVariance;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? RunningCorrelations => _runningCorrelations;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public OnlineFeatureSelector(
        int nFeaturesToSelect = 10,
        double decayFactor = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (decayFactor <= 0 || decayFactor > 1)
            throw new ArgumentException("Decay factor must be between 0 and 1.", nameof(decayFactor));

        _nFeaturesToSelect = nFeaturesToSelect;
        _decayFactor = decayFactor;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "OnlineFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Initialize statistics
        _runningMeans = new double[p];
        _runningVariances = new double[p];
        _runningCorrelations = new double[p];
        _targetMean = 0;
        _targetVariance = 0;
        _sampleCount = 0;

        // Process data incrementally
        for (int i = 0; i < n; i++)
        {
            double y = NumOps.ToDouble(target[i]);
            PartialFit(data, i, y);
        }

        // Select top features
        UpdateSelection(p);
        IsFitted = true;
    }

    /// <summary>
    /// Update statistics with a single new sample (streaming update).
    /// </summary>
    public void PartialFit(Matrix<T> data, int rowIndex, double target)
    {
        if (_runningMeans is null || _runningVariances is null || _runningCorrelations is null)
        {
            _nInputFeatures = data.Columns;
            _runningMeans = new double[data.Columns];
            _runningVariances = new double[data.Columns];
            _runningCorrelations = new double[data.Columns];
        }

        _sampleCount++;
        int p = _runningMeans.Length;

        // Update target statistics (Welford's algorithm)
        double deltaY = target - _targetMean;
        _targetMean += deltaY / _sampleCount;
        _targetVariance += deltaY * (target - _targetMean);

        // Update feature statistics and correlations
        for (int j = 0; j < p; j++)
        {
            double x = NumOps.ToDouble(data[rowIndex, j]);

            // Welford's algorithm for mean and variance
            double deltaX = x - _runningMeans[j];
            _runningMeans[j] += deltaX / _sampleCount;
            _runningVariances[j] += deltaX * (x - _runningMeans[j]);

            // Exponential moving correlation estimate
            double stdX = Math.Sqrt(_runningVariances[j] / (_sampleCount + 1e-10));
            double stdY = Math.Sqrt(_targetVariance / (_sampleCount + 1e-10));

            if (stdX > 1e-10 && stdY > 1e-10)
            {
                double z_x = (x - _runningMeans[j]) / stdX;
                double z_y = (target - _targetMean) / stdY;
                double instantCorr = z_x * z_y;

                // Exponential moving average of correlation
                _runningCorrelations[j] = (1 - _decayFactor) * _runningCorrelations[j]
                                        + _decayFactor * instantCorr;
            }
        }
    }

    private void UpdateSelection(int p)
    {
        if (_runningCorrelations is null) return;

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _runningCorrelations
            .Select((c, idx) => (Corr: Math.Abs(c), Index: idx))
            .OrderByDescending(x => x.Corr)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
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
            throw new InvalidOperationException("OnlineFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("OnlineFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("OnlineFeatureSelector has not been fitted.");

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
