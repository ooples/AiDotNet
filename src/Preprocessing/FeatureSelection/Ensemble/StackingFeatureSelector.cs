using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Stacking-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses a stacking ensemble approach where multiple base feature scorers are
/// combined using a meta-learner to produce final feature rankings.
/// </para>
/// <para><b>For Beginners:</b> Stacking combines multiple ways of measuring
/// feature importance (like correlation, variance, and mutual information).
/// A second layer then learns how to best combine these measures, giving you
/// more reliable feature selection than any single method alone.
/// </para>
/// </remarks>
public class StackingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _stackedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? StackedScores => _stackedScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public StackingFeatureSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StackingFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Base scorers: correlation, variance, mutual information approximation
        var correlationScores = ComputeCorrelationScores(X, y, n, p);
        var varianceScores = ComputeVarianceScores(X, n, p);
        var miScores = ComputeMIScores(X, y, n, p);

        // Normalize scores to [0, 1]
        NormalizeScores(correlationScores);
        NormalizeScores(varianceScores);
        NormalizeScores(miScores);

        // Meta-learner: weighted combination (learn weights from data)
        // Use correlation between base scores and target variance explained
        double wCorr = ComputeMetaWeight(X, y, correlationScores, n, p);
        double wVar = ComputeMetaWeight(X, y, varianceScores, n, p);
        double wMI = ComputeMetaWeight(X, y, miScores, n, p);

        double totalWeight = wCorr + wVar + wMI + 1e-10;
        wCorr /= totalWeight;
        wVar /= totalWeight;
        wMI /= totalWeight;

        // Combine scores
        _stackedScores = new double[p];
        for (int j = 0; j < p; j++)
            _stackedScores[j] = wCorr * correlationScores[j] + wVar * varianceScores[j] + wMI * miScores[j];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _stackedScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeCorrelationScores(double[,] X, double[] y, int n, int p)
    {
        var scores = new double[p];
        double yMean = y.Average();

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++) xMean += X[i, j];
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xd = X[i, j] - xMean;
                double yd = y[i] - yMean;
                sxy += xd * yd;
                sxx += xd * xd;
                syy += yd * yd;
            }

            scores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        return scores;
    }

    private double[] ComputeVarianceScores(double[,] X, int n, int p)
    {
        var scores = new double[p];

        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += X[i, j];
            mean /= n;

            double variance = 0;
            for (int i = 0; i < n; i++)
                variance += (X[i, j] - mean) * (X[i, j] - mean);
            scores[j] = variance / (n - 1);
        }

        return scores;
    }

    private double[] ComputeMIScores(double[,] X, double[] y, int n, int p)
    {
        var scores = new double[p];
        int nBins = (int)Math.Sqrt(n) + 1;

        double yMin = y.Min(), yMax = y.Max();
        double yBinWidth = (yMax - yMin) / nBins + 1e-10;

        for (int j = 0; j < p; j++)
        {
            double xMin = double.MaxValue, xMax = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                xMin = Math.Min(xMin, X[i, j]);
                xMax = Math.Max(xMax, X[i, j]);
            }
            double xBinWidth = (xMax - xMin) / nBins + 1e-10;

            var jointCounts = new int[nBins, nBins];
            var xCounts = new int[nBins];
            var yCounts = new int[nBins];

            for (int i = 0; i < n; i++)
            {
                int xBin = Math.Min((int)((X[i, j] - xMin) / xBinWidth), nBins - 1);
                int yBin = Math.Min((int)((y[i] - yMin) / yBinWidth), nBins - 1);
                jointCounts[xBin, yBin]++;
                xCounts[xBin]++;
                yCounts[yBin]++;
            }

            double mi = 0;
            for (int xb = 0; xb < nBins; xb++)
            {
                for (int yb = 0; yb < nBins; yb++)
                {
                    if (jointCounts[xb, yb] > 0 && xCounts[xb] > 0 && yCounts[yb] > 0)
                    {
                        double pxy = (double)jointCounts[xb, yb] / n;
                        double px = (double)xCounts[xb] / n;
                        double py = (double)yCounts[yb] / n;
                        mi += pxy * Math.Log(pxy / (px * py) + 1e-10) / Math.Log(2);
                    }
                }
            }
            scores[j] = Math.Max(0, mi);
        }

        return scores;
    }

    private void NormalizeScores(double[] scores)
    {
        double minVal = scores.Min();
        double maxVal = scores.Max();
        double range = maxVal - minVal + 1e-10;
        for (int j = 0; j < scores.Length; j++)
            scores[j] = (scores[j] - minVal) / range;
    }

    private double ComputeMetaWeight(double[,] X, double[] y, double[] scores, int n, int p)
    {
        // Weight based on how well the top-scored features explain target variance
        var topFeatures = Enumerable.Range(0, p)
            .OrderByDescending(j => scores[j])
            .Take(Math.Min(5, p))
            .ToList();

        double yMean = y.Average();
        double ssTot = y.Sum(yi => (yi - yMean) * (yi - yMean));

        // Simple weighted average prediction
        double ssRes = 0;
        for (int i = 0; i < n; i++)
        {
            double pred = 0;
            double weightSum = 0;
            foreach (int j in topFeatures)
            {
                pred += scores[j] * X[i, j];
                weightSum += scores[j];
            }
            if (weightSum > 1e-10) pred /= weightSum;
            double err = y[i] - pred;
            ssRes += err * err;
        }

        return ssTot > 1e-10 ? Math.Max(0, 1 - ssRes / ssTot) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StackingFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("StackingFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StackingFeatureSelector has not been fitted.");

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
