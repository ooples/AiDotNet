using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Joint Mutual Information (JMI) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// JMI selects features by maximizing the joint mutual information between candidate
/// features and already-selected features with respect to the target. It considers
/// second-order feature interactions for improved selection.
/// </para>
/// <para><b>For Beginners:</b> JMI doesn't just look at each feature individually; it
/// considers how pairs of features work together. When selecting a new feature, it asks:
/// "How much new information does this feature provide about the target when combined
/// with each already-selected feature?" This helps find features that complement each other.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class JMI<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _relevanceScores;
    private double[]? _jmiScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? RelevanceScores => _relevanceScores;
    public double[]? JmiScores => _jmiScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public JMI(
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
            "JMI requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Compute relevance for each feature
        _relevanceScores = new double[p];
        for (int j = 0; j < p; j++)
            _relevanceScores[j] = ComputeMI(discretized, j, targetDiscrete, n);

        _jmiScores = new double[p];

        // Greedy forward selection with JMI criterion
        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToHashSet();

        // Select first feature with highest relevance
        int bestFirst = remaining.OrderByDescending(j => _relevanceScores[j]).First();
        selected.Add(bestFirst);
        remaining.Remove(bestFirst);
        _jmiScores[bestFirst] = _relevanceScores[bestFirst];

        // Iteratively select remaining features
        while (selected.Count < _nFeaturesToSelect && remaining.Count > 0)
        {
            double bestScore = double.MinValue;
            int bestFeature = -1;

            foreach (int j in remaining)
            {
                // JMI criterion: sum of I(Xj, Xs; Y) for all selected features
                double jmiSum = 0;
                foreach (int s in selected)
                {
                    // I(Xj, Xs; Y) = I(Xj; Y) + I(Xs; Y | Xj)
                    double jointMI = ComputeJointMI(discretized, j, s, targetDiscrete, n);
                    jmiSum += jointMI;
                }
                jmiSum /= selected.Count; // Average over selected features

                if (jmiSum > bestScore)
                {
                    bestScore = jmiSum;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                remaining.Remove(bestFeature);
                _jmiScores[bestFeature] = bestScore;
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

    private double ComputeJointMI(int[,] features, int f1, int f2, int[] target, int n)
    {
        // I(X1, X2; Y) = sum over x1,x2,y of p(x1,x2,y) * log(p(x1,x2,y) / (p(x1,x2) * p(y)))
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

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("JMI has not been fitted.");

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
        throw new NotSupportedException("JMI does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("JMI has not been fitted.");

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
