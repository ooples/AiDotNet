using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Conditional Infomax Feature Extraction (CIFE) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// CIFE is an information-theoretic method that maximizes the conditional mutual
/// information while considering the interaction between features and the target.
/// </para>
/// <para><b>For Beginners:</b> CIFE evaluates how much new information a feature
/// provides about the target, given what you already know from selected features.
/// It's like asking "what do you know that I don't already know?" for each feature.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CIFE<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly double _beta;

    private double[]? _cifeScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double Beta => _beta;
    public double[]? CIFEScores => _cifeScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CIFE(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        double beta = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));
        if (beta < 0)
            throw new ArgumentException("Beta must be non-negative.", nameof(beta));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _beta = beta;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CIFE requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var discretized = Discretize(data, n, p);
        var targetDiscretized = DiscretizeTarget(target, n);

        var relevance = new double[p];
        for (int j = 0; j < p; j++)
            relevance[j] = ComputeMI(discretized, j, targetDiscretized, n);

        _cifeScores = new double[p];
        var selected = new List<int>();

        for (int k = 0; k < Math.Min(_nFeaturesToSelect, p); k++)
        {
            double bestScore = double.MinValue;
            int bestFeature = -1;

            for (int j = 0; j < p; j++)
            {
                if (selected.Contains(j)) continue;

                double cifeScore = relevance[j];
                if (selected.Count > 0)
                {
                    double condInfo = 0;
                    foreach (int s in selected)
                    {
                        double redundancy = ComputeFeatureMI(discretized, j, s, n);
                        double interaction = ComputeInteraction(discretized, j, s, targetDiscretized, n);
                        condInfo += redundancy - _beta * interaction;
                    }
                    cifeScore -= condInfo;
                }

                if (cifeScore > bestScore)
                {
                    bestScore = cifeScore;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                _cifeScores[bestFeature] = bestScore;
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

    private double ComputeFeatureMI(int[,] disc, int j1, int j2, int n)
    {
        var joint = new int[_nBins, _nBins];
        var f1 = new int[_nBins];
        var f2 = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            joint[disc[i, j1], disc[i, j2]]++;
            f1[disc[i, j1]]++;
            f2[disc[i, j2]]++;
        }

        double mi = 0;
        for (int b1 = 0; b1 < _nBins; b1++)
        {
            for (int b2 = 0; b2 < _nBins; b2++)
            {
                if (joint[b1, b2] > 0 && f1[b1] > 0 && f2[b2] > 0)
                {
                    double pJoint = (double)joint[b1, b2] / n;
                    double p1 = (double)f1[b1] / n;
                    double p2 = (double)f2[b2] / n;
                    mi += pJoint * Math.Log(pJoint / (p1 * p2) + 1e-10);
                }
            }
        }

        return mi;
    }

    private double ComputeInteraction(int[,] disc, int j1, int j2, int[] target, int n)
    {
        var counts = new Dictionary<(int, int, int), int>();
        for (int i = 0; i < n; i++)
        {
            var key = (disc[i, j1], disc[i, j2], target[i]);
            counts[key] = counts.GetValueOrDefault(key) + 1;
        }

        double jointMI = 0;
        foreach (var kvp in counts)
        {
            double pJoint = (double)kvp.Value / n;
            jointMI += pJoint * Math.Log(pJoint + 1e-10);
        }

        return -jointMI;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CIFE has not been fitted.");

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
        throw new NotSupportedException("CIFE does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CIFE has not been fitted.");

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
