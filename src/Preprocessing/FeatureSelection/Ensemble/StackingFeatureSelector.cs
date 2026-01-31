using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Stacking-based Ensemble Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Stacking Feature Selection uses multiple base selectors and combines their
/// rankings using a meta-level aggregation. Each base selector contributes a
/// score, and features are selected based on the aggregated meta-score.
/// </para>
/// <para><b>For Beginners:</b> Like stacking in machine learning, this method
/// uses the outputs of several feature selection methods as inputs to make a
/// final decision. It's more sophisticated than simple voting because it learns
/// how to weight each method's opinion.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StackingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly string _aggregation;

    private double[][]? _baseScores;
    private double[]? _metaScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public string Aggregation => _aggregation;
    public double[][]? BaseScores => _baseScores;
    public double[]? MetaScores => _metaScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public StackingFeatureSelector(
        int nFeaturesToSelect = 10,
        string aggregation = "mean",
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        var validAggregations = new[] { "mean", "median", "max", "harmonic" };
        if (!validAggregations.Contains(aggregation.ToLower()))
            throw new ArgumentException("Aggregation must be 'mean', 'median', 'max', or 'harmonic'.", nameof(aggregation));

        _nFeaturesToSelect = nFeaturesToSelect;
        _aggregation = aggregation.ToLower();
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

        // Compute scores from multiple base methods
        _baseScores = new double[5][];
        _baseScores[0] = NormalizeScores(ComputeCorrelation(data, target, n, p));
        _baseScores[1] = NormalizeScores(ComputeFisherScore(data, target, n, p));
        _baseScores[2] = NormalizeScores(ComputeVariance(data, n, p));
        _baseScores[3] = NormalizeScores(ComputeReliefF(data, target, n, p));
        _baseScores[4] = NormalizeScores(ComputeMutualInfo(data, target, n, p));

        // Aggregate scores using meta-level function
        _metaScores = AggregateScores(_baseScores, p);

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _metaScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] NormalizeScores(double[] scores)
    {
        double min = scores.Min();
        double max = scores.Max();
        double range = max - min;

        if (range < 1e-10)
            return scores.Select(_ => 0.5).ToArray();

        return scores.Select(s => (s - min) / range).ToArray();
    }

    private double[] AggregateScores(double[][] baseScores, int p)
    {
        var meta = new double[p];
        int nMethods = baseScores.Length;

        for (int j = 0; j < p; j++)
        {
            var featureScores = new double[nMethods];
            for (int m = 0; m < nMethods; m++)
                featureScores[m] = baseScores[m][j];

            switch (_aggregation)
            {
                case "mean":
                    meta[j] = featureScores.Average();
                    break;
                case "median":
                    Array.Sort(featureScores);
                    meta[j] = featureScores[nMethods / 2];
                    break;
                case "max":
                    meta[j] = featureScores.Max();
                    break;
                case "harmonic":
                    double sum = 0;
                    foreach (double s in featureScores)
                        sum += 1.0 / (s + 1e-10);
                    meta[j] = nMethods / sum;
                    break;
            }
        }

        return meta;
    }

    private double[] ComputeCorrelation(Matrix<T> data, Vector<T> target, int n, int p)
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

    private double[] ComputeFisherScore(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];

        var class0 = new List<int>();
        var class1 = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                class0.Add(i);
            else
                class1.Add(i);
        }

        if (class0.Count < 2 || class1.Count < 2)
            return scores;

        for (int j = 0; j < p; j++)
        {
            double mean0 = class0.Sum(i => NumOps.ToDouble(data[i, j])) / class0.Count;
            double mean1 = class1.Sum(i => NumOps.ToDouble(data[i, j])) / class1.Count;

            double var0 = class0.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean0, 2)) / class0.Count;
            double var1 = class1.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean1, 2)) / class1.Count;

            double denom = var0 + var1;
            scores[j] = denom > 1e-10 ? Math.Pow(mean0 - mean1, 2) / denom : 0;
        }

        return scores;
    }

    private double[] ComputeVariance(Matrix<T> data, int n, int p)
    {
        var scores = new double[p];

        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
                mean += NumOps.ToDouble(data[i, j]);
            mean /= n;

            double variance = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean;
                variance += diff * diff;
            }
            scores[j] = variance / n;
        }

        return scores;
    }

    private double[] ComputeReliefF(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        int nSamples = Math.Min(100, n);

        for (int s = 0; s < nSamples; s++)
        {
            int idx = s * n / nSamples;
            double targetVal = NumOps.ToDouble(target[idx]);

            // Find nearest hit and miss
            double nearestHitDist = double.MaxValue;
            double nearestMissDist = double.MaxValue;
            int nearestHit = -1, nearestMiss = -1;

            for (int i = 0; i < n; i++)
            {
                if (i == idx) continue;

                double dist = 0;
                for (int j = 0; j < p; j++)
                {
                    double diff = NumOps.ToDouble(data[idx, j]) - NumOps.ToDouble(data[i, j]);
                    dist += diff * diff;
                }

                bool sameClass = Math.Abs(NumOps.ToDouble(target[i]) - targetVal) < 0.5;

                if (sameClass && dist < nearestHitDist)
                {
                    nearestHitDist = dist;
                    nearestHit = i;
                }
                else if (!sameClass && dist < nearestMissDist)
                {
                    nearestMissDist = dist;
                    nearestMiss = i;
                }
            }

            if (nearestHit >= 0 && nearestMiss >= 0)
            {
                for (int j = 0; j < p; j++)
                {
                    double hitDiff = Math.Abs(NumOps.ToDouble(data[idx, j]) - NumOps.ToDouble(data[nearestHit, j]));
                    double missDiff = Math.Abs(NumOps.ToDouble(data[idx, j]) - NumOps.ToDouble(data[nearestMiss, j]));
                    scores[j] += missDiff - hitDiff;
                }
            }
        }

        for (int j = 0; j < p; j++)
            scores[j] /= nSamples;

        return scores;
    }

    private double[] ComputeMutualInfo(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        int nBins = 10;

        for (int j = 0; j < p; j++)
        {
            // Discretize feature
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }

            double range = maxVal - minVal;
            var jointCounts = new int[nBins, 2];
            var featureCounts = new int[nBins];
            var targetCounts = new int[2];

            for (int i = 0; i < n; i++)
            {
                int fBin = range > 1e-10
                    ? Math.Min(nBins - 1, (int)((NumOps.ToDouble(data[i, j]) - minVal) / range * (nBins - 1)))
                    : 0;
                int tBin = NumOps.ToDouble(target[i]) >= 0.5 ? 1 : 0;

                jointCounts[fBin, tBin]++;
                featureCounts[fBin]++;
                targetCounts[tBin]++;
            }

            double mi = 0;
            for (int f = 0; f < nBins; f++)
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
