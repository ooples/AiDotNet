using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Interaction Information for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Interaction Information measures the amount of information bound up in a set of
/// variables that is not present in any subset. It can be positive (synergy) or
/// negative (redundancy), revealing higher-order dependencies.
/// </para>
/// <para><b>For Beginners:</b> This method finds features that work together in
/// unexpected ways. Sometimes two features alone are useless, but together they're
/// very predictive (synergy). Other times, features duplicate each other's information
/// (redundancy). Interaction Information captures both.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class InteractionInformation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly bool _preferSynergy;

    private double[]? _miScores;
    private double[]? _interactionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public bool PreferSynergy => _preferSynergy;
    public double[]? MIScores => _miScores;
    public double[]? InteractionScores => _interactionScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public InteractionInformation(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        bool preferSynergy = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _preferSynergy = preferSynergy;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "InteractionInformation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Greedy selection considering interaction information
        var selected = new List<int>();
        _interactionScores = new double[p];

        // Select first feature with highest MI
        int bestFirst = _miScores
            .Select((mi, idx) => (MI: mi, Index: idx))
            .OrderByDescending(x => x.MI)
            .First().Index;

        selected.Add(bestFirst);
        _interactionScores[bestFirst] = _miScores[bestFirst];

        while (selected.Count < _nFeaturesToSelect && selected.Count < p)
        {
            int bestFeature = -1;
            double bestScore = double.MinValue;

            for (int j = 0; j < p; j++)
            {
                if (selected.Contains(j)) continue;

                // Compute interaction information with selected features
                double score = _miScores[j];
                double totalInteraction = 0;

                foreach (int s in selected)
                {
                    // I(X_j; X_s; Y) = I(X_j; Y) + I(X_s; Y) - I(X_j, X_s; Y)
                    double jointMI = ComputeJointMI(discretized, j, s, discretizedTarget, n);
                    double interaction = _miScores[j] + _miScores[s] - jointMI;
                    totalInteraction += interaction;
                }

                // Adjust score based on interaction
                if (_preferSynergy)
                {
                    // Negative interaction means synergy, so subtract it to prefer synergy
                    score -= totalInteraction / selected.Count;
                }
                else
                {
                    // Positive interaction means redundancy, so add it to penalize redundancy
                    score -= Math.Abs(totalInteraction) / selected.Count;
                }

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                _interactionScores[bestFeature] = bestScore;
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
            throw new InvalidOperationException("InteractionInformation has not been fitted.");

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
        throw new NotSupportedException("InteractionInformation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InteractionInformation has not been fitted.");

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
