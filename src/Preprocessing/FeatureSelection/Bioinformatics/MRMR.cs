using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bioinformatics;

/// <summary>
/// Minimum Redundancy Maximum Relevance (mRMR) feature selection.
/// </summary>
/// <remarks>
/// <para>
/// mRMR selects features that have high relevance to the target (maximum relevance)
/// while minimizing redundancy among selected features. Originally developed for
/// gene expression analysis.
/// </para>
/// <para><b>For Beginners:</b> Imagine choosing team members: you want people with
/// the right skills (relevance to the task) but with different specialties so they
/// don't overlap (minimum redundancy). mRMR finds features that are predictive of
/// the target but provide diverse, non-overlapping information.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MRMR<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly MRMRMethod _method;

    private double[]? _mrmrScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public enum MRMRMethod
    {
        /// <summary>MID: Mutual Information Difference (I - R)</summary>
        MID,
        /// <summary>MIQ: Mutual Information Quotient (I / R)</summary>
        MIQ
    }

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? MRMRScores => _mrmrScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MRMR(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        MRMRMethod method = MRMRMethod.MID,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _method = method;
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

        // Discretize data
        var discreteData = new int[n, p];
        var discreteTarget = new int[n];

        for (int j = 0; j < p; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                min = Math.Min(min, val);
                max = Math.Max(max, val);
            }
            double range = max - min;
            if (range < 1e-10) range = 1;

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                int bin = (int)((val - min) / range * (_nBins - 1));
                discreteData[i, j] = Math.Max(0, Math.Min(bin, _nBins - 1));
            }
        }

        // Discretize target
        double tMin = double.MaxValue, tMax = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(target[i]);
            tMin = Math.Min(tMin, val);
            tMax = Math.Max(tMax, val);
        }
        double tRange = tMax - tMin;
        if (tRange < 1e-10) tRange = 1;

        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(target[i]);
            int bin = (int)((val - tMin) / tRange * (_nBins - 1));
            discreteTarget[i] = Math.Max(0, Math.Min(bin, _nBins - 1));
        }

        // Calculate mutual information with target
        var relevance = new double[p];
        for (int j = 0; j < p; j++)
            relevance[j] = MutualInformation(discreteData, j, discreteTarget, n);

        // Greedy forward selection with mRMR criterion
        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToHashSet();
        _mrmrScores = new double[p];

        while (selected.Count < Math.Min(_nFeaturesToSelect, p) && remaining.Count > 0)
        {
            int bestFeature = -1;
            double bestScore = double.MinValue;

            foreach (int j in remaining)
            {
                double redundancy = 0;
                if (selected.Count > 0)
                {
                    foreach (int s in selected)
                        redundancy += FeatureMutualInformation(discreteData, j, s, n);
                    redundancy /= selected.Count;
                }

                double score;
                if (_method == MRMRMethod.MID)
                    score = relevance[j] - redundancy;
                else // MIQ
                    score = redundancy > 1e-10 ? relevance[j] / redundancy : relevance[j];

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                _mrmrScores[bestFeature] = bestScore;
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

    private double MutualInformation(int[,] data, int featureIdx, int[] target, int n)
    {
        // Joint and marginal distributions
        var jointCount = new int[_nBins, _nBins];
        var featureCount = new int[_nBins];
        var targetCount = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            int fVal = data[i, featureIdx];
            int tVal = target[i];
            jointCount[fVal, tVal]++;
            featureCount[fVal]++;
            targetCount[tVal]++;
        }

        double mi = 0;
        for (int f = 0; f < _nBins; f++)
        {
            for (int t = 0; t < _nBins; t++)
            {
                if (jointCount[f, t] > 0)
                {
                    double pJoint = (double)jointCount[f, t] / n;
                    double pFeature = (double)featureCount[f] / n;
                    double pTarget = (double)targetCount[t] / n;

                    if (pFeature > 0 && pTarget > 0)
                        mi += pJoint * Math.Log(pJoint / (pFeature * pTarget) + 1e-10);
                }
            }
        }

        return mi;
    }

    private double FeatureMutualInformation(int[,] data, int f1, int f2, int n)
    {
        var jointCount = new int[_nBins, _nBins];
        var count1 = new int[_nBins];
        var count2 = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            int v1 = data[i, f1];
            int v2 = data[i, f2];
            jointCount[v1, v2]++;
            count1[v1]++;
            count2[v2]++;
        }

        double mi = 0;
        for (int i = 0; i < _nBins; i++)
        {
            for (int j = 0; j < _nBins; j++)
            {
                if (jointCount[i, j] > 0)
                {
                    double pJoint = (double)jointCount[i, j] / n;
                    double p1 = (double)count1[i] / n;
                    double p2 = (double)count2[j] / n;

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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
