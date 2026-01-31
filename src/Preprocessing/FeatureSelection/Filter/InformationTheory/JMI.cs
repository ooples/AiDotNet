using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Joint Mutual Information (JMI) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// JMI selects features that maximize joint mutual information with the target,
/// considering both the individual feature-target relationship and how features
/// work together.
/// </para>
/// <para><b>For Beginners:</b> JMI looks at how much a feature tells you about
/// the target when combined with previously selected features. A feature might
/// be weak alone but very powerful in combination with others. JMI finds these
/// synergistic feature combinations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class JMI<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _jmiScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? JMIScores => _jmiScores;
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
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

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
        var discretized = Discretize(data, n, p);
        var targetDiscretized = DiscretizeTarget(target, n);

        // Compute individual MI scores first
        var miScores = new double[p];
        for (int j = 0; j < p; j++)
            miScores[j] = ComputeMI(discretized, j, targetDiscretized, n);

        // Greedy feature selection with JMI criterion
        _jmiScores = new double[p];
        var selected = new List<int>();

        for (int k = 0; k < Math.Min(_nFeaturesToSelect, p); k++)
        {
            double bestScore = double.MinValue;
            int bestFeature = -1;

            for (int j = 0; j < p; j++)
            {
                if (selected.Contains(j)) continue;

                double jmiScore = miScores[j];
                if (selected.Count > 0)
                {
                    foreach (int s in selected)
                        jmiScore += ComputeJointMI(discretized, j, s, targetDiscretized, n);
                    jmiScore /= (selected.Count + 1);
                }

                if (jmiScore > bestScore)
                {
                    bestScore = jmiScore;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                _jmiScores[bestFeature] = bestScore;
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private int[,] Discretize(Matrix<T> data, int n, int p)
    {
        var result = new int[n, p];
        for (int j = 0; j < p; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (val < min) min = val;
                if (val > max) max = val;
            }
            double range = max - min;

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                result[i, j] = range > 1e-10
                    ? Math.Min(_nBins - 1, (int)((val - min) / range * (_nBins - 1)))
                    : 0;
            }
        }
        return result;
    }

    private int[] DiscretizeTarget(Vector<T> target, int n)
    {
        var result = new int[n];
        for (int i = 0; i < n; i++)
            result[i] = NumOps.ToDouble(target[i]) >= 0.5 ? 1 : 0;
        return result;
    }

    private double ComputeMI(int[,] disc, int j, int[] target, int n)
    {
        var joint = new int[_nBins, 2];
        var feat = new int[_nBins];
        var tgt = new int[2];

        for (int i = 0; i < n; i++)
        {
            joint[disc[i, j], target[i]]++;
            feat[disc[i, j]]++;
            tgt[target[i]]++;
        }

        double mi = 0;
        for (int f = 0; f < _nBins; f++)
        {
            for (int t = 0; t < 2; t++)
            {
                if (joint[f, t] > 0 && feat[f] > 0 && tgt[t] > 0)
                {
                    double pJoint = (double)joint[f, t] / n;
                    double pFeat = (double)feat[f] / n;
                    double pTgt = (double)tgt[t] / n;
                    mi += pJoint * Math.Log(pJoint / (pFeat * pTgt) + 1e-10);
                }
            }
        }

        return mi;
    }

    private double ComputeJointMI(int[,] disc, int j1, int j2, int[] target, int n)
    {
        var joint = new Dictionary<(int, int, int), int>();
        var feat = new Dictionary<(int, int), int>();
        var tgt = new int[2];

        for (int i = 0; i < n; i++)
        {
            var jKey = (disc[i, j1], disc[i, j2], target[i]);
            var fKey = (disc[i, j1], disc[i, j2]);

            joint[jKey] = joint.GetValueOrDefault(jKey) + 1;
            feat[fKey] = feat.GetValueOrDefault(fKey) + 1;
            tgt[target[i]]++;
        }

        double mi = 0;
        foreach (var kvp in joint)
        {
            var fKey = (kvp.Key.Item1, kvp.Key.Item2);
            int t = kvp.Key.Item3;
            if (feat[fKey] > 0 && tgt[t] > 0)
            {
                double pJoint = (double)kvp.Value / n;
                double pFeat = (double)feat[fKey] / n;
                double pTgt = (double)tgt[t] / n;
                mi += pJoint * Math.Log(pJoint / (pFeat * pTgt) + 1e-10);
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
