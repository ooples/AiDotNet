using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Double Input Symmetrical Relevance (DISR) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// DISR uses a symmetrical relevance measure that considers both feature-target and
/// feature-feature relationships. It maximizes a normalized joint relevance score
/// that accounts for redundancy in a principled way.
/// </para>
/// <para><b>For Beginners:</b> DISR balances relevance and redundancy by looking at
/// how much two features together tell you about the target, normalized by their
/// total information content. It's like asking: "Of all the information in this
/// feature pair, what fraction is actually useful for prediction?"
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
    public double[]? RelevanceScores => _relevanceScores;
    public double[]? DisrScores => _disrScores;
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

        // Discretize features
        var discretized = new int[n, p];
        for (int j = 0; j < p; j++)
        {
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, val);
                maxVal = Math.Max(maxVal, val);
            }

            double range = maxVal - minVal;
            if (range < 1e-10) range = 1;

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                discretized[i, j] = Math.Min((int)(((val - minVal) / range) * _nBins), _nBins - 1);
            }
        }

        // Discretize target
        var targetDiscrete = new int[n];
        for (int i = 0; i < n; i++)
            targetDiscrete[i] = (int)Math.Round(NumOps.ToDouble(target[i]));

        // Compute individual relevance scores
        _relevanceScores = new double[p];
        for (int j = 0; j < p; j++)
            _relevanceScores[j] = ComputeMI(discretized, j, targetDiscrete, n);

        // Compute entropies
        var featureEntropies = new double[p];
        for (int j = 0; j < p; j++)
            featureEntropies[j] = ComputeEntropy(discretized, j, n);

        double targetEntropy = ComputeTargetEntropy(targetDiscrete, n);

        _disrScores = new double[p];

        // Greedy forward selection with DISR criterion
        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToHashSet();

        // Select first feature with highest symmetrical uncertainty
        double[] initialScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            double sumEntropy = featureEntropies[j] + targetEntropy;
            initialScores[j] = sumEntropy > 1e-10 ? 2 * _relevanceScores[j] / sumEntropy : 0;
        }

        int bestFirst = initialScores.Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .First().Index;
        selected.Add(bestFirst);
        remaining.Remove(bestFirst);
        _disrScores[bestFirst] = initialScores[bestFirst];

        // Iteratively select remaining features
        while (selected.Count < _nFeaturesToSelect && remaining.Count > 0)
        {
            double bestScore = double.NegativeInfinity;
            int bestFeature = -1;

            foreach (int j in remaining)
            {
                // DISR: sum of I(Xj, Xs; Y) / H(Xj, Xs, Y) for selected features
                double disrSum = 0;
                foreach (int s in selected)
                {
                    double jointMI = ComputeJointMI(discretized, j, s, targetDiscrete, n);
                    double jointEntropy = ComputeJointEntropy(discretized, j, s, targetDiscrete, n);
                    disrSum += jointEntropy > 1e-10 ? jointMI / jointEntropy : 0;
                }
                disrSum /= selected.Count;

                if (disrSum > bestScore)
                {
                    bestScore = disrSum;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                remaining.Remove(bestFeature);
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

    private double ComputeMI(int[,] features, int featureIdx, int[] target, int n)
    {
        var jointCounts = new Dictionary<(int, int), int>();
        var featureCounts = new int[_nBins];
        var targetCounts = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            int f = features[i, featureIdx];
            int t = target[i];

            featureCounts[f]++;
            if (!targetCounts.ContainsKey(t))
                targetCounts[t] = 0;
            targetCounts[t]++;

            var key = (f, t);
            if (!jointCounts.ContainsKey(key))
                jointCounts[key] = 0;
            jointCounts[key]++;
        }

        double mi = 0;
        foreach (var kvp in jointCounts)
        {
            int f = kvp.Key.Item1;
            int t = kvp.Key.Item2;
            int joint = kvp.Value;

            double pJoint = (double)joint / n;
            double pF = (double)featureCounts[f] / n;
            double pT = (double)targetCounts[t] / n;

            if (pJoint > 0 && pF > 0 && pT > 0)
                mi += pJoint * Math.Log(pJoint / (pF * pT));
        }

        return mi;
    }

    private double ComputeEntropy(int[,] features, int featureIdx, int n)
    {
        var counts = new int[_nBins];
        for (int i = 0; i < n; i++)
            counts[features[i, featureIdx]]++;

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

    private double ComputeTargetEntropy(int[] target, int n)
    {
        var counts = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            if (!counts.ContainsKey(target[i]))
                counts[target[i]] = 0;
            counts[target[i]]++;
        }

        double entropy = 0;
        foreach (var c in counts.Values)
        {
            double p = (double)c / n;
            entropy -= p * Math.Log(p);
        }

        return entropy;
    }

    private double ComputeJointMI(int[,] features, int f1, int f2, int[] target, int n)
    {
        var jointXXY = new Dictionary<(int, int, int), int>();
        var jointXX = new Dictionary<(int, int), int>();
        var targetCounts = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            int v1 = features[i, f1];
            int v2 = features[i, f2];
            int t = target[i];

            var keyXXY = (v1, v2, t);
            if (!jointXXY.ContainsKey(keyXXY))
                jointXXY[keyXXY] = 0;
            jointXXY[keyXXY]++;

            var keyXX = (v1, v2);
            if (!jointXX.ContainsKey(keyXX))
                jointXX[keyXX] = 0;
            jointXX[keyXX]++;

            if (!targetCounts.ContainsKey(t))
                targetCounts[t] = 0;
            targetCounts[t]++;
        }

        double mi = 0;
        foreach (var kvp in jointXXY)
        {
            int v1 = kvp.Key.Item1;
            int v2 = kvp.Key.Item2;
            int t = kvp.Key.Item3;
            int countXXY = kvp.Value;

            var keyXX = (v1, v2);
            int countXX = jointXX[keyXX];
            int countY = targetCounts[t];

            double pXXY = (double)countXXY / n;
            double pXX = (double)countXX / n;
            double pY = (double)countY / n;

            if (pXXY > 0 && pXX > 0 && pY > 0)
                mi += pXXY * Math.Log(pXXY / (pXX * pY));
        }

        return mi;
    }

    private double ComputeJointEntropy(int[,] features, int f1, int f2, int[] target, int n)
    {
        var jointCounts = new Dictionary<(int, int, int), int>();

        for (int i = 0; i < n; i++)
        {
            var key = (features[i, f1], features[i, f2], target[i]);
            if (!jointCounts.ContainsKey(key))
                jointCounts[key] = 0;
            jointCounts[key]++;
        }

        double entropy = 0;
        foreach (var count in jointCounts.Values)
        {
            double p = (double)count / n;
            entropy -= p * Math.Log(p);
        }

        return entropy;
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
