using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Correlation-Mutual Information Hybrid feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines linear correlation (Pearson) with mutual information (non-linear)
/// to capture both types of feature-target relationships.
/// </para>
/// <para><b>For Beginners:</b> Correlation finds straight-line relationships
/// while mutual information finds any type of pattern. A feature might be
/// strongly related to the target in a curved way that correlation misses.
/// This hybrid catches both types of useful features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CorrelationMIHybrid<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _correlationWeight;
    private readonly int _nBins;

    private double[]? _correlationScores;
    private double[]? _miScores;
    private double[]? _hybridScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double CorrelationWeight => _correlationWeight;
    public double[]? CorrelationScores => _correlationScores;
    public double[]? MIScores => _miScores;
    public double[]? HybridScores => _hybridScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CorrelationMIHybrid(
        int nFeaturesToSelect = 10,
        double correlationWeight = 0.5,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (correlationWeight < 0 || correlationWeight > 1)
            throw new ArgumentException("Correlation weight must be between 0 and 1.", nameof(correlationWeight));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _correlationWeight = correlationWeight;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CorrelationMIHybrid requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _correlationScores = ComputeCorrelationScores(data, target, n, p);
        _miScores = ComputeMutualInfoScores(data, target, n, p);

        var normalizedCorr = Normalize(_correlationScores);
        var normalizedMI = Normalize(_miScores);

        _hybridScores = new double[p];
        for (int j = 0; j < p; j++)
            _hybridScores[j] = _correlationWeight * normalizedCorr[j] + (1 - _correlationWeight) * normalizedMI[j];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _hybridScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeCorrelationScores(Matrix<T> data, Vector<T> target, int n, int p)
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

    private double[] ComputeMutualInfoScores(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];

        for (int j = 0; j < p; j++)
        {
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }

            double range = maxVal - minVal;
            var jointCounts = new int[_nBins, 2];
            var featureCounts = new int[_nBins];
            var targetCounts = new int[2];

            for (int i = 0; i < n; i++)
            {
                int fBin = range > 1e-10
                    ? Math.Min(_nBins - 1, (int)((NumOps.ToDouble(data[i, j]) - minVal) / range * (_nBins - 1)))
                    : 0;
                int tBin = NumOps.ToDouble(target[i]) >= 0.5 ? 1 : 0;

                jointCounts[fBin, tBin]++;
                featureCounts[fBin]++;
                targetCounts[tBin]++;
            }

            double mi = 0;
            for (int f = 0; f < _nBins; f++)
            {
                for (int t = 0; t < 2; t++)
                {
                    if (jointCounts[f, t] > 0 && featureCounts[f] > 0 && targetCounts[t] > 0)
                    {
                        double pJoint = (double)jointCounts[f, t] / n;
                        double pFeature = (double)featureCounts[f] / n;
                        double pTarget = (double)targetCounts[t] / n;
                        mi += pJoint * Math.Log(pJoint / (pFeature * pTarget) + 1e-10);
                    }
                }
            }

            scores[j] = mi;
        }

        return scores;
    }

    private double[] Normalize(double[] scores)
    {
        double min = scores.Min();
        double max = scores.Max();
        double range = max - min;

        if (range < 1e-10)
            return scores.Select(_ => 0.5).ToArray();

        return scores.Select(s => (s - min) / range).ToArray();
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationMIHybrid has not been fitted.");

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
        throw new NotSupportedException("CorrelationMIHybrid does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationMIHybrid has not been fitted.");

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
