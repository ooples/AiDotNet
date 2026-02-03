using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Mutual Information Feature Selection (MIFS) with redundancy penalty.
/// </summary>
/// <remarks>
/// <para>
/// MIFS iteratively selects features that have high mutual information with the target
/// while penalizing redundancy with already-selected features. The beta parameter
/// controls the strength of the redundancy penalty.
/// </para>
/// <para><b>For Beginners:</b> MIFS greedily picks features that tell you a lot about
/// the target. But it also subtracts a penalty if the new feature overlaps with what
/// you already know from selected features. The beta parameter controls how harsh
/// this penalty is: 0 = no penalty (just pick high MI), 1 = full penalty.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MIFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly double _beta;

    private double[]? _relevanceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? RelevanceScores => _relevanceScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MIFS(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        double beta = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _beta = beta;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MIFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Compute relevance (MI with target) for each feature
        _relevanceScores = new double[p];
        for (int j = 0; j < p; j++)
            _relevanceScores[j] = ComputeMI(discretized, j, targetDiscrete, n);

        // Greedy forward selection with MIFS criterion
        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToHashSet();

        // Select first feature with highest relevance
        int bestFirst = remaining.OrderByDescending(j => _relevanceScores[j]).First();
        selected.Add(bestFirst);
        remaining.Remove(bestFirst);

        // Iteratively select remaining features
        while (selected.Count < _nFeaturesToSelect && remaining.Count > 0)
        {
            double bestScore = double.MinValue;
            int bestFeature = -1;

            foreach (int j in remaining)
            {
                // Compute redundancy with selected features
                double redundancy = 0;
                foreach (int s in selected)
                    redundancy += ComputeFeaturesMI(discretized, j, s, n);

                // MIFS criterion: I(f;Y) - beta * sum(I(f;fs))
                double score = _relevanceScores[j] - _beta * redundancy;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                remaining.Remove(bestFeature);
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

    private double ComputeFeaturesMI(int[,] features, int f1, int f2, int n)
    {
        var jointCounts = new Dictionary<(int, int), int>();
        var counts1 = new int[_nBins];
        var counts2 = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            int v1 = features[i, f1];
            int v2 = features[i, f2];

            counts1[v1]++;
            counts2[v2]++;

            var key = (v1, v2);
            if (!jointCounts.ContainsKey(key))
                jointCounts[key] = 0;
            jointCounts[key]++;
        }

        double mi = 0;
        foreach (var kvp in jointCounts)
        {
            int v1 = kvp.Key.Item1;
            int v2 = kvp.Key.Item2;
            int joint = kvp.Value;

            double pJoint = (double)joint / n;
            double p1 = (double)counts1[v1] / n;
            double p2 = (double)counts2[v2] / n;

            if (pJoint > 0 && p1 > 0 && p2 > 0)
                mi += pJoint * Math.Log(pJoint / (p1 * p2));
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
            throw new InvalidOperationException("MIFS has not been fitted.");

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
        throw new NotSupportedException("MIFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MIFS has not been fitted.");

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
