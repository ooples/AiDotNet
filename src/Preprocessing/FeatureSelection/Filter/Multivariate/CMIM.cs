using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Conditional Mutual Information Maximization (CMIM) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// CMIM uses a max-min criterion to select features. For each candidate, it computes
/// the minimum conditional mutual information with respect to all already-selected
/// features, then selects the candidate with the maximum of these minima.
/// </para>
/// <para><b>For Beginners:</b> CMIM is cautious about redundancy. When considering a
/// new feature, it looks at how much information it provides about the target given
/// each already-selected feature (conditioning). It takes the worst case (minimum)
/// and picks the feature whose worst case is best. This ensures every selected feature
/// adds value no matter which other feature you consider.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CMIM<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _relevanceScores;
    private double[]? _cmimScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? RelevanceScores => _relevanceScores;
    public double[]? CmimScores => _cmimScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CMIM(
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
            "CMIM requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Compute relevance for each feature (initial CMIM scores)
        _relevanceScores = new double[p];
        _cmimScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            _relevanceScores[j] = ComputeMI(discretized, j, targetDiscrete, n);
            _cmimScores[j] = _relevanceScores[j];
        }

        // Greedy forward selection with CMIM criterion
        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToHashSet();

        // Select first feature with highest relevance
        int bestFirst = remaining.OrderByDescending(j => _relevanceScores[j]).First();
        selected.Add(bestFirst);
        remaining.Remove(bestFirst);

        // Iteratively select remaining features
        while (selected.Count < _nFeaturesToSelect && remaining.Count > 0)
        {
            int lastSelected = selected[^1];

            // Update CMIM scores based on last selected feature
            foreach (int j in remaining)
            {
                // I(Xj; Y | Xs) = I(Xj; Y, Xs) - I(Xj; Xs)
                double condMI = ComputeConditionalMI(discretized, j, lastSelected, targetDiscrete, n);
                _cmimScores[j] = Math.Min(_cmimScores[j], condMI);
            }

            // Select feature with maximum CMIM score
            int bestFeature = -1;
            double bestScore = double.NegativeInfinity;

            foreach (int j in remaining)
            {
                if (_cmimScores[j] > bestScore)
                {
                    bestScore = _cmimScores[j];
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

    private double ComputeConditionalMI(int[,] features, int f1, int f2, int[] target, int n)
    {
        // I(X1; Y | X2) = sum over x2 of P(x2) * I(X1; Y | X2=x2)
        var f2Counts = new int[_nBins];
        for (int i = 0; i < n; i++)
            f2Counts[features[i, f2]]++;

        double cmi = 0;

        for (int x2 = 0; x2 < _nBins; x2++)
        {
            if (f2Counts[x2] == 0) continue;

            double pX2 = (double)f2Counts[x2] / n;

            // Compute I(X1; Y | X2 = x2)
            var jointCounts = new Dictionary<(int, int), int>();
            var f1Counts = new int[_nBins];
            var targetCounts = new Dictionary<int, int>();
            int nConditional = 0;

            for (int i = 0; i < n; i++)
            {
                if (features[i, f2] != x2) continue;

                nConditional++;
                int v1 = features[i, f1];
                int t = target[i];

                f1Counts[v1]++;

                if (!targetCounts.ContainsKey(t))
                    targetCounts[t] = 0;
                targetCounts[t]++;

                var key = (v1, t);
                if (!jointCounts.ContainsKey(key))
                    jointCounts[key] = 0;
                jointCounts[key]++;
            }

            if (nConditional == 0) continue;

            double miGivenX2 = 0;
            foreach (var kvp in jointCounts)
            {
                int v1 = kvp.Key.Item1;
                int t = kvp.Key.Item2;
                int joint = kvp.Value;

                double pJoint = (double)joint / nConditional;
                double pF1 = (double)f1Counts[v1] / nConditional;
                double pT = (double)targetCounts[t] / nConditional;

                if (pJoint > 0 && pF1 > 0 && pT > 0)
                    miGivenX2 += pJoint * Math.Log(pJoint / (pF1 * pT));
            }

            cmi += pX2 * miGivenX2;
        }

        return cmi;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CMIM has not been fitted.");

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
        throw new NotSupportedException("CMIM does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CMIM has not been fitted.");

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
