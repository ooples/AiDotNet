using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Conditional Mutual Information Maximization (CMIM) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// CMIM selects features that maximize mutual information with the target while
/// conditioning on already selected features. It addresses redundancy by considering
/// conditional dependencies.
/// </para>
/// <para><b>For Beginners:</b> CMIM picks features that provide new information about
/// the target that isn't already captured by previously selected features. It's like
/// building a team where each new member brings truly unique knowledge.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CMIM<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _miScores;
    private double[]? _cmimScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? MIScores => _miScores;
    public double[]? CMIMScores => _cmimScores;
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
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

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

        var discretized = DiscretizeData(data, n, p);
        var discretizedTarget = DiscretizeTarget(target, n);

        // Compute MI(X_i; Y) for all features
        _miScores = new double[p];
        for (int j = 0; j < p; j++)
            _miScores[j] = ComputeMI(discretized, j, discretizedTarget, n);

        // Greedy selection using CMIM criterion
        var selected = new List<int>();
        _cmimScores = new double[p];

        // Select first feature with highest MI
        int bestFirst = _miScores
            .Select((mi, idx) => (MI: mi, Index: idx))
            .OrderByDescending(x => x.MI)
            .First().Index;

        selected.Add(bestFirst);
        _cmimScores[bestFirst] = _miScores[bestFirst];

        // Initialize minimum conditional MI for each feature
        var minCondMI = new double[p];
        for (int j = 0; j < p; j++)
            minCondMI[j] = double.MaxValue;

        while (selected.Count < _nFeaturesToSelect && selected.Count < p)
        {
            int lastSelected = selected[^1];
            int bestFeature = -1;
            double bestScore = double.MinValue;

            for (int j = 0; j < p; j++)
            {
                if (selected.Contains(j)) continue;

                // Update minimum conditional MI with last selected feature
                double condMI = ComputeConditionalMI(discretized, j, lastSelected, discretizedTarget, n);
                minCondMI[j] = Math.Min(minCondMI[j], condMI);

                // CMIM criterion: max over features of min conditional MI
                if (minCondMI[j] > bestScore)
                {
                    bestScore = minCondMI[j];
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                _cmimScores[bestFeature] = bestScore;
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

    private double ComputeConditionalMI(int[,] discretized, int f1, int f2, int[] target, int n)
    {
        // I(X_f1; Y | X_f2) = H(X_f1 | X_f2) + H(Y | X_f2) - H(X_f1, Y | X_f2)
        // Simplified computation using joint probabilities
        var counts = new Dictionary<(int, int, int), int>();
        var f2Counts = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            int v1 = discretized[i, f1];
            int v2 = discretized[i, f2];
            int t = target[i];

            var key = (v1, v2, t);
            counts[key] = counts.GetValueOrDefault(key, 0) + 1;
            f2Counts[v2]++;
        }

        double condMI = 0;
        foreach (var kvp in counts)
        {
            var (v1, v2, t) = kvp.Key;
            int jointCount = kvp.Value;

            if (jointCount > 0 && f2Counts[v2] > 0)
            {
                // Count marginals conditioned on f2
                int f1GivenF2 = 0;
                int tGivenF2 = 0;
                for (int ti = 0; ti < 2; ti++)
                    f1GivenF2 += counts.GetValueOrDefault((v1, v2, ti), 0);
                for (int vi = 0; vi < _nBins; vi++)
                    tGivenF2 += counts.GetValueOrDefault((vi, v2, t), 0);

                double pJointGivenF2 = (double)jointCount / f2Counts[v2];
                double pF1GivenF2 = (double)f1GivenF2 / f2Counts[v2];
                double pTGivenF2 = (double)tGivenF2 / f2Counts[v2];
                double pF2 = (double)f2Counts[v2] / n;

                if (pF1GivenF2 > 0 && pTGivenF2 > 0)
                    condMI += pF2 * pJointGivenF2 * Math.Log(pJointGivenF2 / (pF1GivenF2 * pTGivenF2) + 1e-10);
            }
        }

        return condMI;
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
