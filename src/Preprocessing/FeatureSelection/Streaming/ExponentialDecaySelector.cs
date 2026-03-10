using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Streaming;

/// <summary>
/// Exponential Decay Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses exponentially weighted statistics for feature selection, giving more
/// importance to recent samples while gradually forgetting older ones.
/// </para>
/// <para><b>For Beginners:</b> Unlike a sharp sliding window, exponential decay
/// smoothly reduces the importance of older samples. Recent data has the most
/// influence, but older data still contributes a little. This creates a smooth
/// transition as the data distribution changes.
/// </para>
/// </remarks>
public class ExponentialDecaySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _decayFactor;

    private double[]? _ewmaMeans;
    private double[]? _ewmaVars;
    private double[]? _ewmaCovariances;
    private double _ewmaTargetMean;
    private double _ewmaTargetVar;
    private double _effectiveN;
    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double DecayFactor => _decayFactor;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ExponentialDecaySelector(
        int nFeaturesToSelect = 10,
        double decayFactor = 0.99,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (decayFactor <= 0 || decayFactor >= 1)
            throw new ArgumentException("Decay factor must be between 0 and 1.", nameof(decayFactor));

        _nFeaturesToSelect = nFeaturesToSelect;
        _decayFactor = decayFactor;
        _effectiveN = 0;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ExponentialDecaySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        int n = data.Rows;
        int p = data.Columns;
        _nInputFeatures = p;

        // Initialize if needed
        if (_ewmaMeans is null)
        {
            _ewmaMeans = new double[p];
            _ewmaVars = new double[p];
            _ewmaCovariances = new double[p];
        }

        // Process samples with exponential decay
        var ewmaMeans = _ewmaMeans ?? throw new InvalidOperationException("EWMA means have not been initialized.");
        var ewmaVars = _ewmaVars ?? throw new InvalidOperationException("EWMA variances have not been initialized.");
        var ewmaCovariances = _ewmaCovariances ?? throw new InvalidOperationException("EWMA covariances have not been initialized.");

        for (int i = 0; i < n; i++)
        {
            double yi = NumOps.ToDouble(target[i]);

            // Update effective sample count
            _effectiveN = _decayFactor * _effectiveN + 1;

            // Update target EWMA
            double oldTargetMean = _ewmaTargetMean;
            _ewmaTargetMean = _decayFactor * _ewmaTargetMean + (1 - _decayFactor) * yi;
            _ewmaTargetVar = _decayFactor * _ewmaTargetVar +
                (1 - _decayFactor) * (yi - oldTargetMean) * (yi - _ewmaTargetMean);

            for (int j = 0; j < p; j++)
            {
                double xij = NumOps.ToDouble(data[i, j]);

                double oldMean = ewmaMeans[j];
                ewmaMeans[j] = _decayFactor * ewmaMeans[j] + (1 - _decayFactor) * xij;
                ewmaVars[j] = _decayFactor * ewmaVars[j] +
                    (1 - _decayFactor) * (xij - oldMean) * (xij - ewmaMeans[j]);

                ewmaCovariances[j] = _decayFactor * ewmaCovariances[j] +
                    (1 - _decayFactor) * (xij - ewmaMeans[j]) * (yi - _ewmaTargetMean);
            }
        }

        // Compute correlation scores
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            double xVar = ewmaVars[j];
            double yVar = _ewmaTargetVar;
            if (xVar > 1e-10 && yVar > 1e-10)
                _featureScores[j] = Math.Abs(ewmaCovariances[j] / Math.Sqrt(xVar * yVar));
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public void PartialFit(Matrix<T> data, Vector<T> target)
    {
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
            throw new InvalidOperationException("ExponentialDecaySelector has not been fitted.");

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
        throw new NotSupportedException("ExponentialDecaySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ExponentialDecaySelector has not been fitted.");

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
