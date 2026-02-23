using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Correlation-based Feature Selection (CFS).
/// </summary>
/// <remarks>
/// <para>
/// CFS evaluates feature subsets by considering both feature-target correlation
/// (relevance) and feature-feature correlation (redundancy). Good subsets have
/// high correlation with the target but low intercorrelation among features.
/// </para>
/// <para><b>For Beginners:</b> CFS looks for features that are strongly related
/// to what you want to predict but not too similar to each other. Having two
/// features that are almost identical doesn't help much - you want diverse
/// information sources that all point toward the target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CorrelationBasedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CorrelationBasedFS(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CorrelationBasedFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute all correlations
        var featureTargetCorr = new double[p];
        var featureCorr = new double[p, p];

        // Compute means
        var featureMeans = new double[p];
        double targetMean = 0;

        for (int i = 0; i < n; i++)
            targetMean += NumOps.ToDouble(target[i]);
        targetMean /= n;

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                featureMeans[j] += NumOps.ToDouble(data[i, j]);
            featureMeans[j] /= n;
        }

        // Compute feature-target correlations
        for (int j = 0; j < p; j++)
        {
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - featureMeans[j];
                double yDiff = NumOps.ToDouble(target[i]) - targetMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }
            featureTargetCorr[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        // Compute feature-feature correlations
        for (int j1 = 0; j1 < p; j1++)
        {
            featureCorr[j1, j1] = 1.0;
            for (int j2 = j1 + 1; j2 < p; j2++)
            {
                double sxy = 0, sxx = 0, syy = 0;
                for (int i = 0; i < n; i++)
                {
                    double x1Diff = NumOps.ToDouble(data[i, j1]) - featureMeans[j1];
                    double x2Diff = NumOps.ToDouble(data[i, j2]) - featureMeans[j2];
                    sxy += x1Diff * x2Diff;
                    sxx += x1Diff * x1Diff;
                    syy += x2Diff * x2Diff;
                }
                double corr = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
                featureCorr[j1, j2] = corr;
                featureCorr[j2, j1] = corr;
            }
        }

        // Greedy forward selection using CFS merit
        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToList();
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        _featureScores = new double[p];

        while (selected.Count < numToSelect && remaining.Count > 0)
        {
            double bestMerit = double.NegativeInfinity;
            int bestIdx = -1;
            int bestRemainingPos = -1;

            for (int r = 0; r < remaining.Count; r++)
            {
                int candidate = remaining[r];
                var testSet = new List<int>(selected) { candidate };

                // Compute CFS merit: k * r_cf / sqrt(k + k*(k-1)*r_ff)
                // r_cf = mean feature-target correlation
                // r_ff = mean feature-feature correlation
                double sumCf = 0;
                foreach (int f in testSet)
                    sumCf += featureTargetCorr[f];
                double meanCf = sumCf / testSet.Count;

                double sumFf = 0;
                int countFf = 0;
                for (int i = 0; i < testSet.Count; i++)
                {
                    for (int j = i + 1; j < testSet.Count; j++)
                    {
                        sumFf += featureCorr[testSet[i], testSet[j]];
                        countFf++;
                    }
                }
                double meanFf = countFf > 0 ? sumFf / countFf : 0;

                int k = testSet.Count;
                double denom = Math.Sqrt(k + k * (k - 1) * meanFf);
                double merit = denom > 1e-10 ? k * meanCf / denom : 0;

                if (merit > bestMerit)
                {
                    bestMerit = merit;
                    bestIdx = candidate;
                    bestRemainingPos = r;
                }
            }

            if (bestIdx >= 0)
            {
                selected.Add(bestIdx);
                remaining.RemoveAt(bestRemainingPos);
                _featureScores[bestIdx] = bestMerit;
            }
            else
            {
                break;
            }
        }

        _selectedIndices = [.. selected.OrderBy(x => x)];

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationBasedFS has not been fitted.");

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
        throw new NotSupportedException("CorrelationBasedFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationBasedFS has not been fitted.");

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
