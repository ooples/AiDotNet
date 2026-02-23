using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Online;

/// <summary>
/// Online Streaming Feature Selection (OSFS) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// OSFS performs feature selection in an online/streaming fashion where features
/// arrive one at a time. It maintains a set of selected features and uses
/// conditional independence tests to decide whether to add new features.
/// </para>
/// <para><b>For Beginners:</b> Imagine features arriving one by one like a stream.
/// For each new feature, OSFS asks: "Does this add useful information beyond what
/// I already have?" If yes, it keeps the feature. It also checks if any existing
/// features become redundant after adding the new one.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class OSFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _epsilon;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double Epsilon => _epsilon;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public OSFS(
        int nFeaturesToSelect = 10,
        double alpha = 0.05,
        double epsilon = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _epsilon = epsilon;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "OSFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _featureScores = new double[p];
        var selectedSet = new HashSet<int>();

        // Compute initial scores (correlation with target)
        for (int j = 0; j < p; j++)
            _featureScores[j] = ComputeCorrelation(data, target, j);

        // Process features in order of their initial score (descending)
        var orderedFeatures = _featureScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .ToList();

        foreach (var (score, featureIdx) in orderedFeatures)
        {
            if (selectedSet.Count >= _nFeaturesToSelect)
                break;

            // Test if feature is significantly correlated with target
            if (!IsRelevant(score))
                continue;

            // Test conditional independence given selected features
            bool isIndependent = true;
            foreach (int selectedIdx in selectedSet)
            {
                if (AreConditionallyDependent(data, target, featureIdx, selectedIdx))
                {
                    isIndependent = false;
                    break;
                }
            }

            if (!isIndependent)
                continue;

            // Add feature to selected set
            selectedSet.Add(featureIdx);

            // Check if any existing features become redundant
            var toRemove = new List<int>();
            foreach (int selectedIdx in selectedSet)
            {
                if (selectedIdx == featureIdx)
                    continue;

                // Check if selectedIdx is now redundant given featureIdx
                if (IsRedundant(data, target, selectedIdx, featureIdx, selectedSet))
                    toRemove.Add(selectedIdx);
            }

            foreach (int idx in toRemove)
                selectedSet.Remove(idx);
        }

        _selectedIndices = selectedSet.OrderBy(x => x).ToArray();

        // If not enough selected, add top features by score
        if (_selectedIndices.Length < _nFeaturesToSelect)
        {
            var additional = orderedFeatures
                .Where(x => !selectedSet.Contains(x.Index))
                .Take(_nFeaturesToSelect - _selectedIndices.Length)
                .Select(x => x.Index);

            _selectedIndices = selectedSet.Concat(additional).OrderBy(x => x).ToArray();
        }

        IsFitted = true;
    }

    private double ComputeCorrelation(Matrix<T> data, Vector<T> target, int featureIdx)
    {
        int n = data.Rows;

        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += NumOps.ToDouble(data[i, featureIdx]);
            yMean += NumOps.ToDouble(target[i]);
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = NumOps.ToDouble(data[i, featureIdx]) - xMean;
            double yDiff = NumOps.ToDouble(target[i]) - yMean;
            sxy += xDiff * yDiff;
            sxx += xDiff * xDiff;
            syy += yDiff * yDiff;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
    }

    private bool IsRelevant(double score)
    {
        // A feature is relevant if its correlation exceeds a threshold
        return score > _epsilon;
    }

    private bool AreConditionallyDependent(Matrix<T> data, Vector<T> target, int f1, int f2)
    {
        int n = data.Rows;

        // Compute partial correlation between f1 and target given f2
        double r_f1_y = ComputeCorrelation(data, target, f1);
        double r_f2_y = ComputeCorrelationBetweenFeatures(data, target, f2);
        double r_f1_f2 = ComputeCorrelationBetweenTwoFeatures(data, f1, f2);

        // Partial correlation formula
        double denominator = Math.Sqrt((1 - r_f1_f2 * r_f1_f2) * (1 - r_f2_y * r_f2_y));
        if (denominator < 1e-10)
            return true; // Assume dependent if can't compute

        double partialCorr = (r_f1_y - r_f1_f2 * r_f2_y) / denominator;

        // Test if partial correlation is significant
        double t = partialCorr * Math.Sqrt((n - 3) / (1 - partialCorr * partialCorr + 1e-10));
        double pValue = 2 * (1 - NormalCDF(Math.Abs(t)));

        return pValue < _alpha;
    }

    private double ComputeCorrelationBetweenFeatures(Matrix<T> data, Vector<T> target, int featureIdx)
    {
        return ComputeCorrelation(data, target, featureIdx);
    }

    private double ComputeCorrelationBetweenTwoFeatures(Matrix<T> data, int f1, int f2)
    {
        int n = data.Rows;

        double x1Mean = 0, x2Mean = 0;
        for (int i = 0; i < n; i++)
        {
            x1Mean += NumOps.ToDouble(data[i, f1]);
            x2Mean += NumOps.ToDouble(data[i, f2]);
        }
        x1Mean /= n;
        x2Mean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double x1Diff = NumOps.ToDouble(data[i, f1]) - x1Mean;
            double x2Diff = NumOps.ToDouble(data[i, f2]) - x2Mean;
            sxy += x1Diff * x2Diff;
            sxx += x1Diff * x1Diff;
            syy += x2Diff * x2Diff;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    private bool IsRedundant(Matrix<T> data, Vector<T> target, int candidate, int conditioning, HashSet<int> currentSet)
    {
        // A feature is redundant if it becomes conditionally independent of target
        // given the conditioning feature(s)
        double corrWithTarget = ComputeCorrelation(data, target, candidate);
        double corrWithConditioning = ComputeCorrelationBetweenTwoFeatures(data, candidate, conditioning);

        // If highly correlated with conditioning feature and not much more correlated with target
        return Math.Abs(corrWithConditioning) > 0.9 && corrWithTarget < _epsilon * 2;
    }

    private double NormalCDF(double x)
    {
        double t = 1.0 / (1.0 + 0.2316419 * Math.Abs(x));
        double d = 0.3989423 * Math.Exp(-x * x / 2);
        double p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return x > 0 ? 1 - p : p;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("OSFS has not been fitted.");

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
        throw new NotSupportedException("OSFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("OSFS has not been fitted.");

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
