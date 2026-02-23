using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Mutual Information and Correlation Hybrid Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines mutual information (captures non-linear relationships) with
/// Pearson correlation (captures linear relationships) to identify features
/// that are important under both perspectives.
/// </para>
/// <para><b>For Beginners:</b> Some features have linear relationships
/// (more X means more Y), while others have non-linear patterns (X affects
/// Y in complex ways). This method uses correlation to find linear patterns
/// and mutual information to find any pattern, then combines both scores.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MICorrelationHybrid<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _miWeight;
    private readonly int _nBins;

    private double[]? _miScores;
    private double[]? _correlationScores;
    private double[]? _hybridScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double MIWeight => _miWeight;
    public double[]? MIScores => _miScores;
    public double[]? CorrelationScores => _correlationScores;
    public double[]? HybridScores => _hybridScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MICorrelationHybrid(
        int nFeaturesToSelect = 10,
        double miWeight = 0.5,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (miWeight < 0 || miWeight > 1)
            throw new ArgumentException("MI weight must be between 0 and 1.", nameof(miWeight));

        _nFeaturesToSelect = nFeaturesToSelect;
        _miWeight = miWeight;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MICorrelationHybrid requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _miScores = new double[p];
        _correlationScores = new double[p];
        _hybridScores = new double[p];

        // Compute target statistics
        double yMean = 0;
        var yValues = new double[n];
        for (int i = 0; i < n; i++)
        {
            yValues[i] = NumOps.ToDouble(target[i]);
            yMean += yValues[i];
        }
        yMean /= n;

        double yMin = yValues.Min();
        double yMax = yValues.Max();
        double yRange = yMax - yMin;

        var yBins = new int[n];
        for (int i = 0; i < n; i++)
            yBins[i] = yRange > 1e-10 ? Math.Min((int)((yValues[i] - yMin) / yRange * (_nBins - 1)), _nBins - 1) : 0;

        var yCounts = new int[_nBins];
        foreach (int b in yBins)
            yCounts[b]++;

        for (int j = 0; j < p; j++)
        {
            // Get feature values
            var xValues = new double[n];
            double xMean = 0;
            double xMin = double.MaxValue, xMax = double.MinValue;

            for (int i = 0; i < n; i++)
            {
                xValues[i] = NumOps.ToDouble(data[i, j]);
                xMean += xValues[i];
                xMin = Math.Min(xMin, xValues[i]);
                xMax = Math.Max(xMax, xValues[i]);
            }
            xMean /= n;

            // Compute Pearson correlation
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = xValues[i] - xMean;
                double yDiff = yValues[i] - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            _correlationScores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;

            // Compute Mutual Information
            double xRange = xMax - xMin;
            var xBins = new int[n];
            for (int i = 0; i < n; i++)
                xBins[i] = xRange > 1e-10 ? Math.Min((int)((xValues[i] - xMin) / xRange * (_nBins - 1)), _nBins - 1) : 0;

            var xCounts = new int[_nBins];
            foreach (int b in xBins)
                xCounts[b]++;

            var jointCounts = new Dictionary<(int, int), int>();
            for (int i = 0; i < n; i++)
            {
                var key = (xBins[i], yBins[i]);
                jointCounts[key] = jointCounts.GetValueOrDefault(key) + 1;
            }

            double mi = 0;
            foreach (var kvp in jointCounts)
            {
                double pxy = (double)kvp.Value / n;
                double px = (double)xCounts[kvp.Key.Item1] / n;
                double py = (double)yCounts[kvp.Key.Item2] / n;

                if (pxy > 0 && px > 0 && py > 0)
                    mi += pxy * Math.Log(pxy / (px * py));
            }

            _miScores[j] = mi;
        }

        // Normalize scores
        double maxMI = _miScores.Max();
        double maxCorr = _correlationScores.Max();

        for (int j = 0; j < p; j++)
        {
            double normalizedMI = maxMI > 1e-10 ? _miScores[j] / maxMI : 0;
            double normalizedCorr = maxCorr > 1e-10 ? _correlationScores[j] / maxCorr : 0;
            _hybridScores[j] = _miWeight * normalizedMI + (1 - _miWeight) * normalizedCorr;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _hybridScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MICorrelationHybrid has not been fitted.");

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
        throw new NotSupportedException("MICorrelationHybrid does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MICorrelationHybrid has not been fitted.");

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
