using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Minimum Redundancy Maximum Relevance (mRMR) feature selection.
/// </summary>
/// <remarks>
/// <para>
/// mRMR selects features that have maximum relevance to the target while having
/// minimum redundancy among themselves. It balances finding informative features
/// with avoiding duplicated information.
/// </para>
/// <para><b>For Beginners:</b> mRMR tries to pick features that are both useful
/// (related to what you're predicting) and diverse (not repeating the same information).
/// It's like assembling a team - you want skilled people, but also people with
/// different skills rather than everyone being good at the same thing.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MRMR<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _mrmrScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? MRMRScores => _mrmrScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MRMR(
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
            "MRMR requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Compute mutual information I(X_j; Y) for all features
        var relevance = new double[p];
        for (int j = 0; j < p; j++)
            relevance[j] = ComputeMutualInformation(discretized, j, targetDiscrete, n, _nBins, targetSet.Count);

        // Compute pairwise mutual information I(X_i; X_j)
        var redundancy = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                double mi = ComputeFeatureMI(discretized, j1, j2, n, _nBins);
                redundancy[j1, j2] = mi;
                redundancy[j2, j1] = mi;
            }
        }

        // Greedy mRMR selection
        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToList();
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        _mrmrScores = new double[p];

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
        _mrmrScores[bestFirst] = bestRelevance;

        // Subsequent features: maximize (relevance - average redundancy)
        while (selected.Count < numToSelect && remaining.Count > 0)
        {
            double bestScore = double.NegativeInfinity;
            int bestIdx = -1;
            int bestPos = -1;

            for (int r = 0; r < remaining.Count; r++)
            {
                int candidate = remaining[r];
                double rel = relevance[candidate];

                // Compute average redundancy with selected features
                double avgRed = 0;
                foreach (int s in selected)
                    avgRed += redundancy[candidate, s];
                avgRed /= selected.Count;

                double score = rel - avgRed;

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
                _mrmrScores[bestIdx] = bestScore;
            }
            else
            {
                break;
            }
        }

        _selectedIndices = [.. selected.OrderBy(x => x)];

        IsFitted = true;
    }

    private double ComputeMutualInformation(int[,] X, int featureIdx, int[] Y, int n, int nBinsX, int nBinsY)
    {
        var jointCounts = new Dictionary<(int, int), int>();
        var xCounts = new int[nBinsX];
        var yCounts = new int[nBinsY];

        for (int i = 0; i < n; i++)
        {
            int x = X[i, featureIdx];
            int y = Y[i];

            xCounts[x]++;
            if (y >= 0 && y < nBinsY)
                yCounts[y]++;

            var key = (x, y);
            if (!jointCounts.ContainsKey(key))
                jointCounts[key] = 0;
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
            double py = y >= 0 && y < nBinsY ? (double)yCounts[y] / n : 1.0 / n;

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

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MRMR has not been fitted.");

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
        throw new NotSupportedException("MRMR does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MRMR has not been fitted.");

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
