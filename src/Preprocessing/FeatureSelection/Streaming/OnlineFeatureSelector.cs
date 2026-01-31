using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Streaming;

/// <summary>
/// Online Feature Selection for Streaming Data.
/// </summary>
/// <remarks>
/// <para>
/// Performs incremental feature selection that can update as new data arrives,
/// suitable for streaming or online learning scenarios.
/// </para>
/// <para><b>For Beginners:</b> Traditional feature selection processes all data
/// at once. Online selection updates incrementally as new samples arrive, which
/// is useful when you can't store all data in memory or when data comes in
/// continuously.
/// </para>
/// </remarks>
public class OnlineFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _learningRate;

    private double[]? _runningScores;
    private double[]? _runningMeans;
    private double[]? _runningVars;
    private int _sampleCount;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double LearningRate => _learningRate;
    public double[]? RunningScores => _runningScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public OnlineFeatureSelector(
        int nFeaturesToSelect = 10,
        double learningRate = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _learningRate = learningRate;
        _sampleCount = 0;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "OnlineFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        // Initialize or update statistics
        int n = data.Rows;
        int p = data.Columns;
        _nInputFeatures = p;

        if (_runningScores is null)
        {
            _runningScores = new double[p];
            _runningMeans = new double[p];
            _runningVars = new double[p];
        }

        // Process each sample incrementally
        for (int i = 0; i < n; i++)
        {
            double yi = NumOps.ToDouble(target[i]);
            _sampleCount++;

            for (int j = 0; j < p; j++)
            {
                double xi = NumOps.ToDouble(data[i, j]);

                // Welford's online algorithm for mean and variance
                double delta = xi - _runningMeans![j];
                _runningMeans[j] += delta / _sampleCount;
                double delta2 = xi - _runningMeans[j];
                _runningVars![j] += delta * delta2;

                // Update correlation estimate incrementally
                double correlation = ComputeOnlineCorrelation(xi, yi, j);
                _runningScores![j] = (1 - _learningRate) * _runningScores[j] + _learningRate * correlation;
            }
        }

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => Math.Abs(_runningScores![j]))
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double _targetMean = 0;
    private double _targetVar = 0;
    private double[]? _covariances;

    private double ComputeOnlineCorrelation(double x, double y, int featureIdx)
    {
        if (_covariances is null)
            _covariances = new double[_nInputFeatures];

        // Update target statistics
        double oldTargetMean = _targetMean;
        _targetMean += (y - _targetMean) / _sampleCount;
        _targetVar += (y - oldTargetMean) * (y - _targetMean);

        // Update covariance
        _covariances[featureIdx] += (x - _runningMeans![featureIdx]) * (y - _targetMean);

        // Compute correlation
        double xVar = _runningVars![featureIdx];
        double yVar = _targetVar;

        if (xVar > 1e-10 && yVar > 1e-10)
            return _covariances[featureIdx] / Math.Sqrt(xVar * yVar);
        return 0;
    }

    public void PartialFit(Matrix<T> data, Vector<T> target)
    {
        // Same as Fit but designed for incremental updates
        Fit(data, target);
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
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
