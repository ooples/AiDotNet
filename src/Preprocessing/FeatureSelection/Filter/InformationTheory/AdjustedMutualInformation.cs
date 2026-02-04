using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Adjusted Mutual Information correcting for chance agreement.
/// </summary>
/// <remarks>
/// <para>
/// Adjusted Mutual Information (AMI) corrects for the fact that random
/// labelings have non-zero expected MI. It accounts for the number of
/// clusters to give a more accurate measure of agreement.
/// </para>
/// <para><b>For Beginners:</b> Just by chance, two random groupings will
/// share some information. AMI subtracts this expected chance agreement
/// to give a cleaner measure of true dependency.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AdjustedMutualInformation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _amiScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? AMIScores => _amiScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AdjustedMutualInformation(
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
            "AdjustedMutualInformation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Discretize target
        var yDiscrete = DiscretizeTarget(target, n);
        var yCounts = GetCounts(yDiscrete);

        _amiScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var xDiscrete = DiscretizeFeature(data, j, n);
            var xCounts = GetCounts(xDiscrete);

            double mi = ComputeMI(xDiscrete, yDiscrete, n);
            double emi = ComputeExpectedMI(xCounts, yCounts, n);
            double hX = ComputeEntropy(xDiscrete, n);
            double hY = ComputeEntropy(yDiscrete, n);

            double maxH = Math.Max(hX, hY);
            double denominator = maxH - emi;

            if (Math.Abs(denominator) > 1e-10)
                _amiScores[j] = (mi - emi) / denominator;
            else
                _amiScores[j] = 0;
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _amiScores
            .Select((ami, idx) => (AMI: ami, Index: idx))
            .OrderByDescending(x => x.AMI)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] DiscretizeTarget(Vector<T> target, int n)
    {
        var values = new double[n];
        for (int i = 0; i < n; i++)
            values[i] = NumOps.ToDouble(target[i]);

        var unique = values.Distinct().OrderBy(x => x).ToList();
        var result = new int[n];

        for (int i = 0; i < n; i++)
            result[i] = unique.IndexOf(values[i]);

        return result;
    }

    private int[] DiscretizeFeature(Matrix<T> data, int j, int n)
    {
        var values = new double[n];
        for (int i = 0; i < n; i++)
            values[i] = NumOps.ToDouble(data[i, j]);

        double min = values.Min();
        double max = values.Max();
        double range = max - min;
        if (range < 1e-10) range = 1;

        var result = new int[n];
        for (int i = 0; i < n; i++)
        {
            int bin = (int)((values[i] - min) / range * (_nBins - 1));
            result[i] = Math.Max(0, Math.Min(bin, _nBins - 1));
        }

        return result;
    }

    private Dictionary<int, int> GetCounts(int[] labels)
    {
        var counts = new Dictionary<int, int>();
        foreach (int v in labels)
        {
            if (!counts.ContainsKey(v)) counts[v] = 0;
            counts[v]++;
        }
        return counts;
    }

    private double ComputeMI(int[] x, int[] y, int n)
    {
        var jointCounts = new Dictionary<(int, int), int>();
        var xCounts = new Dictionary<int, int>();
        var yCounts = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            var key = (x[i], y[i]);
            if (!jointCounts.ContainsKey(key)) jointCounts[key] = 0;
            jointCounts[key]++;

            if (!xCounts.ContainsKey(x[i])) xCounts[x[i]] = 0;
            xCounts[x[i]]++;

            if (!yCounts.ContainsKey(y[i])) yCounts[y[i]] = 0;
            yCounts[y[i]]++;
        }

        double mi = 0;
        foreach (var kvp in jointCounts)
        {
            int xi = kvp.Key.Item1;
            int yi = kvp.Key.Item2;
            int nij = kvp.Value;

            double pij = (double)nij / n;
            double pi = (double)xCounts[xi] / n;
            double pj = (double)yCounts[yi] / n;

            if (pij > 0 && pi > 0 && pj > 0)
                mi += pij * Math.Log(pij / (pi * pj));
        }

        return mi;
    }

    private double ComputeExpectedMI(Dictionary<int, int> xCounts, Dictionary<int, int> yCounts, int n)
    {
        // Compute expected mutual information under random permutation
        double emi = 0;

        foreach (var ai in xCounts.Values)
        {
            foreach (var bj in yCounts.Values)
            {
                int minNij = Math.Max(1, ai + bj - n);
                int maxNij = Math.Min(ai, bj);

                for (int nij = minNij; nij <= maxNij; nij++)
                {
                    double term = (double)nij / n * Math.Log((double)n * nij / (ai * bj));
                    double hyperProb = HypergeometricPMF(nij, ai, bj, n);
                    emi += term * hyperProb;
                }
            }
        }

        return emi;
    }

    private double HypergeometricPMF(int k, int K, int n1, int N)
    {
        return Math.Exp(LogBinomial(K, k) + LogBinomial(N - K, n1 - k) - LogBinomial(N, n1));
    }

    private double LogBinomial(int n, int k)
    {
        if (k < 0 || k > n) return double.NegativeInfinity;
        if (k == 0 || k == n) return 0;
        return LogFactorial(n) - LogFactorial(k) - LogFactorial(n - k);
    }

    private double LogFactorial(int n)
    {
        if (n <= 1) return 0;
        if (n > 20)
            return n * Math.Log(n) - n + 0.5 * Math.Log(2 * Math.PI * n);

        double result = 0;
        for (int i = 2; i <= n; i++)
            result += Math.Log(i);
        return result;
    }

    private double ComputeEntropy(int[] x, int n)
    {
        var counts = new Dictionary<int, int>();
        foreach (int v in x)
        {
            if (!counts.ContainsKey(v)) counts[v] = 0;
            counts[v]++;
        }

        double entropy = 0;
        foreach (var count in counts.Values)
        {
            double p = (double)count / n;
            if (p > 0) entropy -= p * Math.Log(p);
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
            throw new InvalidOperationException("AdjustedMutualInformation has not been fitted.");

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
        throw new NotSupportedException("AdjustedMutualInformation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AdjustedMutualInformation has not been fitted.");

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
