using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Joint Mutual Information (JMI) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// JMI maximizes the joint mutual information between selected features and the target.
/// It considers both the relevance of individual features and their complementary
/// information when combined with already selected features.
/// </para>
/// <para><b>For Beginners:</b> Instead of just picking features that are individually
/// informative, JMI picks features that work well together. Two features might each
/// be only moderately useful alone, but together they might be very powerful.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class JointMutualInformation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _miScores;
    private double[]? _jmiScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? MIScores => _miScores;
    public double[]? JMIScores => _jmiScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public JointMutualInformation(
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
            "JointMutualInformation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Discretize features
        var discretized = DiscretizeData(data, n, p);
        var discretizedTarget = DiscretizeTarget(target, n);

        // Compute MI(X_i; Y) for all features
        _miScores = new double[p];
        for (int j = 0; j < p; j++)
            _miScores[j] = ComputeMutualInformation(discretized, j, discretizedTarget, n);

        // Greedy selection using JMI criterion
        var selected = new List<int>();
        _jmiScores = new double[p];

        // Select first feature with highest MI
        int bestFirst = _miScores
            .Select((mi, idx) => (MI: mi, Index: idx))
            .OrderByDescending(x => x.MI)
            .First().Index;

        selected.Add(bestFirst);
        _jmiScores[bestFirst] = _miScores[bestFirst];

        // Select remaining features
        while (selected.Count < _nFeaturesToSelect && selected.Count < p)
        {
            int bestFeature = -1;
            double bestJMI = double.MinValue;

            for (int j = 0; j < p; j++)
            {
                if (selected.Contains(j)) continue;

                // JMI criterion: sum of I(X_j, X_s; Y) for all selected features
                double jmi = 0;
                foreach (int s in selected)
                {
                    jmi += ComputeJointMI(discretized, j, s, discretizedTarget, n);
                }
                jmi /= selected.Count;

                if (jmi > bestJMI)
                {
                    bestJMI = jmi;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                _jmiScores[bestFeature] = bestJMI;
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

    private double ComputeMutualInformation(int[,] discretized, int featureIdx, int[] target, int n)
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
        // I(X_f1, X_f2; Y) = I(X_f1; Y) + I(X_f2; Y | X_f1)
        // Approximated as: I(X_f1; Y) + I(X_f2; Y) - I(X_f1; X_f2)
        double mi1 = _miScores![f1];
        double mi2 = _miScores[f2];
        double miJoint = ComputeFeatureFeatureMI(discretized, f1, f2, n);

        return mi1 + mi2 - miJoint;
    }

    private double ComputeFeatureFeatureMI(int[,] discretized, int f1, int f2, int n)
    {
        var jointCounts = new int[_nBins, _nBins];
        var f1Counts = new int[_nBins];
        var f2Counts = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            int v1 = discretized[i, f1];
            int v2 = discretized[i, f2];
            jointCounts[v1, v2]++;
            f1Counts[v1]++;
            f2Counts[v2]++;
        }

        double mi = 0;
        for (int v1 = 0; v1 < _nBins; v1++)
        {
            for (int v2 = 0; v2 < _nBins; v2++)
            {
                if (jointCounts[v1, v2] > 0)
                {
                    double pJoint = (double)jointCounts[v1, v2] / n;
                    double p1 = (double)f1Counts[v1] / n;
                    double p2 = (double)f2Counts[v2] / n;

                    if (p1 > 0 && p2 > 0)
                        mi += pJoint * Math.Log(pJoint / (p1 * p2) + 1e-10);
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
            throw new InvalidOperationException("JointMutualInformation has not been fitted.");

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
        throw new NotSupportedException("JointMutualInformation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("JointMutualInformation has not been fitted.");

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
