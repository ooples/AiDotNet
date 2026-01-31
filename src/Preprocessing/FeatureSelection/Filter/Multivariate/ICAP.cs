using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Interaction Capping (ICAP) feature selection.
/// </summary>
/// <remarks>
/// <para>
/// ICAP extends mRMR by adding interaction information. It considers not just pairwise
/// redundancy but also synergistic effects where combining features provides more
/// information than the sum of individual contributions.
/// </para>
/// <para><b>For Beginners:</b> ICAP looks for features that work well together.
/// Sometimes two features alone aren't very informative, but combined they reveal
/// patterns. ICAP tries to find these synergistic combinations while still avoiding
/// redundancy. It's like finding teammates who complement each other's abilities.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ICAP<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _icapScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? ICAPScores => _icapScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ICAP(
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
            "ICAP requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Discretize all features
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
        var targetSet = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            targetDiscrete[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            targetSet.Add(targetDiscrete[i]);
        }
        int nClasses = targetSet.Count;

        // Compute relevance I(X_j; Y)
        var relevance = new double[p];
        for (int j = 0; j < p; j++)
            relevance[j] = ComputeMI(discretized, j, targetDiscrete, n, _nBins, nClasses);

        // Greedy ICAP selection
        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToList();
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        _icapScores = new double[p];

        // First feature: maximum relevance
        int bestFirst = 0;
        double bestRelevance = relevance[0];
        for (int j = 1; j < p; j++)
        {
            if (relevance[j] > bestRelevance)
            {
                bestRelevance = relevance[j];
                bestFirst = j;
            }
        }

        selected.Add(bestFirst);
        remaining.Remove(bestFirst);
        _icapScores[bestFirst] = bestRelevance;

        // Subsequent features using ICAP criterion
        while (selected.Count < numToSelect && remaining.Count > 0)
        {
            double bestScore = double.NegativeInfinity;
            int bestIdx = -1;
            int bestPos = -1;

            for (int r = 0; r < remaining.Count; r++)
            {
                int candidate = remaining[r];
                double score = relevance[candidate];

                // Compute ICAP penalty
                foreach (int s in selected)
                {
                    // I(X_c; X_s) - I(X_c; X_s | Y)
                    double featureMI = ComputeFeatureMI(discretized, candidate, s, n, _nBins);
                    double conditionalMI = ComputeConditionalMI(discretized, candidate, s, targetDiscrete, n, _nBins, nClasses);

                    // Capping: only penalize if redundancy exceeds class-conditional info
                    double penalty = Math.Max(0, featureMI - conditionalMI);
                    score -= penalty;
                }

                if (score > bestScore)
                {
                    bestScore = score;
                    bestIdx = candidate;
                    bestPos = r;
                }
            }

            if (bestIdx >= 0)
            {
                selected.Add(bestIdx);
                remaining.RemoveAt(bestPos);
                _icapScores[bestIdx] = bestScore;
            }
            else
            {
                break;
            }
        }

        _selectedIndices = [.. selected.OrderBy(x => x)];

        IsFitted = true;
    }

    private double ComputeMI(int[,] X, int j, int[] Y, int n, int nBinsX, int nBinsY)
    {
        var jointCounts = new Dictionary<(int, int), int>();
        var xCounts = new int[nBinsX];
        var yCounts = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            int x = X[i, j];
            int y = Y[i];

            xCounts[x]++;
            if (!yCounts.ContainsKey(y)) yCounts[y] = 0;
            yCounts[y]++;

            var key = (x, y);
            if (!jointCounts.ContainsKey(key)) jointCounts[key] = 0;
            jointCounts[key]++;
        }

        double mi = 0;
        foreach (var kvp in jointCounts)
        {
            int x = kvp.Key.Item1;
            int y = kvp.Key.Item2;
            int joint = kvp.Value;

            double pxy = (double)joint / n;
            double px = (double)xCounts[x] / n;
            double py = (double)yCounts[y] / n;

            if (pxy > 0 && px > 0 && py > 0)
                mi += pxy * Math.Log(pxy / (px * py));
        }

        return mi;
    }

    private double ComputeFeatureMI(int[,] X, int j1, int j2, int n, int nBins)
    {
        var jointCounts = new int[nBins, nBins];
        var counts1 = new int[nBins];
        var counts2 = new int[nBins];

        for (int i = 0; i < n; i++)
        {
            int x1 = X[i, j1];
            int x2 = X[i, j2];
            jointCounts[x1, x2]++;
            counts1[x1]++;
            counts2[x2]++;
        }

        double mi = 0;
        for (int b1 = 0; b1 < nBins; b1++)
        {
            for (int b2 = 0; b2 < nBins; b2++)
            {
                if (jointCounts[b1, b2] > 0)
                {
                    double pxy = (double)jointCounts[b1, b2] / n;
                    double px = (double)counts1[b1] / n;
                    double py = (double)counts2[b2] / n;
                    if (px > 0 && py > 0)
                        mi += pxy * Math.Log(pxy / (px * py));
                }
            }
        }

        return mi;
    }

    private double ComputeConditionalMI(int[,] X, int j1, int j2, int[] Y, int n, int nBins, int nClasses)
    {
        // I(X_j1; X_j2 | Y) = sum_y P(Y=y) * I(X_j1; X_j2 | Y=y)
        var classCounts = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int y = Y[i];
            if (!classCounts.ContainsKey(y)) classCounts[y] = 0;
            classCounts[y]++;
        }

        double totalCMI = 0;
        foreach (var kvp in classCounts)
        {
            int y = kvp.Key;
            int classCount = kvp.Value;
            if (classCount < 2) continue;

            double py = (double)classCount / n;

            // Compute I(X_j1; X_j2) conditioned on Y=y
            var jointCounts = new int[nBins, nBins];
            var counts1 = new int[nBins];
            var counts2 = new int[nBins];

            for (int i = 0; i < n; i++)
            {
                if (Y[i] != y) continue;
                int x1 = X[i, j1];
                int x2 = X[i, j2];
                jointCounts[x1, x2]++;
                counts1[x1]++;
                counts2[x2]++;
            }

            double classCondMI = 0;
            for (int b1 = 0; b1 < nBins; b1++)
            {
                for (int b2 = 0; b2 < nBins; b2++)
                {
                    if (jointCounts[b1, b2] > 0)
                    {
                        double pxy = (double)jointCounts[b1, b2] / classCount;
                        double px = (double)counts1[b1] / classCount;
                        double ppy = (double)counts2[b2] / classCount;
                        if (px > 0 && ppy > 0)
                            classCondMI += pxy * Math.Log(pxy / (px * ppy));
                    }
                }
            }

            totalCMI += py * classCondMI;
        }

        return totalCMI;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ICAP has not been fitted.");

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
        throw new NotSupportedException("ICAP does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ICAP has not been fitted.");

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
