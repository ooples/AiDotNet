using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Double Input Symmetric Relevance (DISR) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// DISR extends mutual information by considering symmetric relevance between
/// pairs of features and the target. It uses the concept of joint symmetric
/// uncertainty to measure feature relevance.
/// </para>
/// <para><b>For Beginners:</b> DISR looks at how pairs of features together
/// relate to the target, using a normalized measure that accounts for both
/// directions of information flow. This helps find features that complement
/// each other in predicting the target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DISR<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _relevanceScores;
    private double[]? _disrScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? RelevanceScores => _relevanceScores;
    public double[]? DISRScores => _disrScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DISR(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DISR requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var discretized = DiscretizeData(data, n, p);
        var discretizedTarget = DiscretizeTarget(target, n);

        // Compute individual relevance scores
        _relevanceScores = new double[p];
        var entropies = new double[p];
        double targetEntropy = ComputeEntropy(discretizedTarget, n, 2);

        for (int j = 0; j < p; j++)
        {
            entropies[j] = ComputeFeatureEntropy(discretized, j, n);
            double mi = ComputeMI(discretized, j, discretizedTarget, n);
            // Symmetric uncertainty
            _relevanceScores[j] = (entropies[j] + targetEntropy) > 0
                ? 2 * mi / (entropies[j] + targetEntropy)
                : 0;
        }

        // Greedy selection using DISR criterion
        var selected = new List<int>();
        _disrScores = new double[p];

        // Select first feature with highest relevance
        int bestFirst = _relevanceScores
            .Select((r, idx) => (Relevance: r, Index: idx))
            .OrderByDescending(x => x.Relevance)
            .First().Index;

        selected.Add(bestFirst);
        _disrScores[bestFirst] = _relevanceScores[bestFirst];

        while (selected.Count < _nFeaturesToSelect && selected.Count < p)
        {
            int bestFeature = -1;
            double bestScore = double.MinValue;

            for (int j = 0; j < p; j++)
            {
                if (selected.Contains(j)) continue;

                // DISR: sum of double input symmetric relevance
                double disr = 0;
                foreach (int s in selected)
                {
                    double jointMI = ComputeJointMI(discretized, j, s, discretizedTarget, n);
                    double jointEntropy = ComputeJointFeatureEntropy(discretized, j, s, n);

                    // Double input symmetric relevance
                    double denom = jointEntropy + targetEntropy;
                    disr += denom > 0 ? 2 * jointMI / denom : 0;
                }

                double score = disr / selected.Count;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                _disrScores[bestFeature] = bestScore;
            }
            else
            {
                break;
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private int[,] DiscretizeData(Matrix<T> data, int n, int p)
    {
        var discretized = new int[n, p];

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
            for (int i = 0; i < n; i++)
            {
                if (range < 1e-10)
                {
                    discretized[i, j] = 0;
                }
                else
                {
                    int bin = (int)((NumOps.ToDouble(data[i, j]) - minVal) / range * (_nBins - 1));
                    discretized[i, j] = Math.Min(_nBins - 1, Math.Max(0, bin));
                }
            }
        }

        return discretized;
    }

    private int[] DiscretizeTarget(Vector<T> target, int n)
    {
        var discretized = new int[n];
        for (int i = 0; i < n; i++)
            discretized[i] = NumOps.ToDouble(target[i]) > 0.5 ? 1 : 0;
        return discretized;
    }

    private double ComputeEntropy(int[] values, int n, int numClasses)
    {
        var counts = new int[numClasses];
        for (int i = 0; i < n; i++)
            counts[values[i]]++;

        double entropy = 0;
        for (int c = 0; c < numClasses; c++)
        {
            if (counts[c] > 0)
            {
                double p = (double)counts[c] / n;
                entropy -= p * Math.Log(p);
            }
        }

        return entropy;
    }

    private double ComputeFeatureEntropy(int[,] discretized, int featureIdx, int n)
    {
        var counts = new int[_nBins];
        for (int i = 0; i < n; i++)
            counts[discretized[i, featureIdx]]++;

        double entropy = 0;
        for (int b = 0; b < _nBins; b++)
        {
            if (counts[b] > 0)
            {
                double p = (double)counts[b] / n;
                entropy -= p * Math.Log(p);
            }
        }

        return entropy;
    }

    private double ComputeJointFeatureEntropy(int[,] discretized, int f1, int f2, int n)
    {
        var counts = new Dictionary<(int, int), int>();
        for (int i = 0; i < n; i++)
        {
            var key = (discretized[i, f1], discretized[i, f2]);
            counts[key] = counts.GetValueOrDefault(key, 0) + 1;
        }

        double entropy = 0;
        foreach (var count in counts.Values)
        {
            if (count > 0)
            {
                double p = (double)count / n;
                entropy -= p * Math.Log(p);
            }
        }

        return entropy;
    }

    private double ComputeMI(int[,] discretized, int featureIdx, int[] target, int n)
    {
        var jointCounts = new int[_nBins, 2];
        var featureCounts = new int[_nBins];
        var targetCounts = new int[2];

        for (int i = 0; i < n; i++)
        {
            int f = discretized[i, featureIdx];
            int t = target[i];
            jointCounts[f, t]++;
            featureCounts[f]++;
            targetCounts[t]++;
        }

        double mi = 0;
        for (int f = 0; f < _nBins; f++)
        {
            for (int t = 0; t < 2; t++)
            {
                if (jointCounts[f, t] > 0)
                {
                    double pJoint = (double)jointCounts[f, t] / n;
                    double pFeature = (double)featureCounts[f] / n;
                    double pTarget = (double)targetCounts[t] / n;

                    if (pFeature > 0 && pTarget > 0)
                        mi += pJoint * Math.Log(pJoint / (pFeature * pTarget) + 1e-10);
                }
            }
        }

        return mi;
    }

    private double ComputeJointMI(int[,] discretized, int f1, int f2, int[] target, int n)
    {
        var counts = new Dictionary<(int, int, int), int>();
        var jointFeatureCounts = new Dictionary<(int, int), int>();
        var targetCounts = new int[2];

        for (int i = 0; i < n; i++)
        {
            int v1 = discretized[i, f1];
            int v2 = discretized[i, f2];
            int t = target[i];

            var tripleKey = (v1, v2, t);
            var pairKey = (v1, v2);

            counts[tripleKey] = counts.GetValueOrDefault(tripleKey, 0) + 1;
            jointFeatureCounts[pairKey] = jointFeatureCounts.GetValueOrDefault(pairKey, 0) + 1;
            targetCounts[t]++;
        }

        double mi = 0;
        foreach (var kvp in counts)
        {
            var (v1, v2, t) = kvp.Key;
            int jointCount = kvp.Value;

            double pJoint = (double)jointCount / n;
            double pFeatures = (double)jointFeatureCounts[(v1, v2)] / n;
            double pTarget = (double)targetCounts[t] / n;

            if (pFeatures > 0 && pTarget > 0)
                mi += pJoint * Math.Log(pJoint / (pFeatures * pTarget) + 1e-10);
        }

        return mi;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DISR has not been fitted.");

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
        throw new NotSupportedException("DISR does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DISR has not been fitted.");

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
