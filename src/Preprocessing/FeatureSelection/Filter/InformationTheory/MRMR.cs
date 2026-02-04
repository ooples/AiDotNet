using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Minimum Redundancy Maximum Relevance (mRMR) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// mRMR selects features that have high mutual information with the target (relevance)
/// while having low mutual information with already selected features (redundancy).
/// </para>
/// <para><b>For Beginners:</b> mRMR tries to find features that are both useful and
/// non-redundant. If two features tell you basically the same thing, you only need
/// one of them. This method picks features that give you new, useful information
/// rather than repeating what you already know.
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
    public int NBins => _nBins;
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
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

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

        var discretized = Discretize(data, n, p);
        var targetDiscretized = DiscretizeTarget(target, n);

        // Compute relevance: I(Xi, Y)
        var relevance = new double[p];
        for (int j = 0; j < p; j++)
            relevance[j] = ComputeMI(discretized, j, targetDiscretized, n);

        // Precompute feature-feature MI
        var featureMI = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                double mi = ComputeFeatureMI(discretized, j1, j2, n);
                featureMI[j1, j2] = mi;
                featureMI[j2, j1] = mi;
            }
        }

        _mrmrScores = new double[p];
        var selected = new List<int>();

        for (int k = 0; k < Math.Min(_nFeaturesToSelect, p); k++)
        {
            double bestScore = double.MinValue;
            int bestFeature = -1;

            for (int j = 0; j < p; j++)
            {
                if (selected.Contains(j)) continue;

                double redundancy = 0;
                if (selected.Count > 0)
                {
                    foreach (int s in selected)
                        redundancy += featureMI[j, s];
                    redundancy /= selected.Count;
                }

                double mrmrScore = relevance[j] - redundancy;

                if (mrmrScore > bestScore)
                {
                    bestScore = mrmrScore;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                _mrmrScores[bestFeature] = bestScore;
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
