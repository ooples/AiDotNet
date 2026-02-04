using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Bidirectional (Stepwise) feature selection combining forward and backward steps.
/// </summary>
/// <remarks>
/// <para>
/// Alternates between forward selection (adding features) and backward elimination
/// (removing features). This allows correcting early poor choices and often finds
/// better subsets than pure forward or backward methods.
/// </para>
/// <para><b>For Beginners:</b> Pure forward selection can get stuck with a bad early
/// choice. Bidirectional selection can fix this by occasionally removing a feature
/// that's no longer useful after others were added. It's like being able to change
/// your mind as you build the feature set.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BidirectionalSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alphaAdd;
    private readonly double _alphaRemove;
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _scoringFunction;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BidirectionalSelection(
        int nFeaturesToSelect = 10,
        double alphaAdd = 0.05,
        double alphaRemove = 0.10,
        Func<Matrix<T>, Vector<T>, int[], double>? scoringFunction = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alphaAdd = alphaAdd;
        _alphaRemove = alphaRemove;
        _scoringFunction = scoringFunction;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BidirectionalSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var scorer = _scoringFunction ?? DefaultScorer;

        var selected = new HashSet<int>();
        var available = Enumerable.Range(0, p).ToHashSet();
        _featureImportances = new double[p];

        double currentScore = 0;

        while (selected.Count < _nFeaturesToSelect && available.Count > 0)
        {
            bool changed = false;

            // Forward step: try to add best feature
            int bestToAdd = -1;
            double bestAddScore = currentScore;

            foreach (int j in available)
            {
                var candidate = selected.Append(j).ToArray();
                double score = scorer(data, target, candidate);
                double improvement = score - currentScore;

                if (improvement > _alphaAdd && score > bestAddScore)
                {
                    bestAddScore = score;
                    bestToAdd = j;
                }
            }

            if (bestToAdd >= 0)
            {
                selected.Add(bestToAdd);
                available.Remove(bestToAdd);
                currentScore = bestAddScore;
                _featureImportances[bestToAdd] = bestAddScore;
                changed = true;
            }

            // Backward step: try to remove worst feature
            if (selected.Count > 1)
            {
                int worstToRemove = -1;
                double bestRemoveScore = double.NegativeInfinity;

                foreach (int j in selected)
                {
                    var candidate = selected.Where(x => x != j).ToArray();
                    double score = scorer(data, target, candidate);
                    double degradation = currentScore - score;

                    if (degradation < _alphaRemove && score > bestRemoveScore)
                    {
                        bestRemoveScore = score;
                        worstToRemove = j;
                    }
                }

                if (worstToRemove >= 0 && bestRemoveScore > currentScore - _alphaRemove)
                {
                    selected.Remove(worstToRemove);
                    available.Add(worstToRemove);
                    currentScore = bestRemoveScore;
                    changed = true;
                }
            }

            if (!changed)
                break;
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();

        if (_selectedIndices.Length == 0)
        {
            // Fall back to top by individual correlation
            _selectedIndices = Enumerable.Range(0, p)
                .Select(j => (Index: j, Score: ComputeCorrelation(data, target, j)))
                .OrderByDescending(x => x.Score)
                .Take(Math.Min(_nFeaturesToSelect, p))
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double DefaultScorer(Matrix<T> data, Vector<T> target, int[] features)
    {
        if (features.Length == 0) return 0;

        // Use sum of squared correlations as default score
        double score = 0;
        foreach (int j in features)
        {
            double corr = ComputeCorrelation(data, target, j);
            score += corr * corr;
        }

        // Penalize feature redundancy
        if (features.Length > 1)
        {
            double redundancy = 0;
            int pairs = 0;
            for (int i = 0; i < features.Length; i++)
            {
                for (int k = i + 1; k < features.Length; k++)
                {
                    redundancy += Math.Abs(ComputeFeatureCorrelation(data, features[i], features[k]));
                    pairs++;
                }
            }
            if (pairs > 0)
                redundancy /= pairs;

            score -= redundancy * score * 0.5;
        }

        return score;
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

        double covariance = 0, xVar = 0, yVar = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = NumOps.ToDouble(data[i, featureIdx]) - xMean;
            double yDiff = NumOps.ToDouble(target[i]) - yMean;
            covariance += xDiff * yDiff;
            xVar += xDiff * xDiff;
            yVar += yDiff * yDiff;
        }

        double denom = Math.Sqrt(xVar * yVar);
        return denom > 1e-10 ? Math.Abs(covariance / denom) : 0;
    }

    private double ComputeFeatureCorrelation(Matrix<T> data, int f1, int f2)
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

        double covariance = 0, x1Var = 0, x2Var = 0;
        for (int i = 0; i < n; i++)
        {
            double x1Diff = NumOps.ToDouble(data[i, f1]) - x1Mean;
            double x2Diff = NumOps.ToDouble(data[i, f2]) - x2Mean;
            covariance += x1Diff * x2Diff;
            x1Var += x1Diff * x1Diff;
            x2Var += x2Diff * x2Diff;
        }

        double denom = Math.Sqrt(x1Var * x2Var);
        return denom > 1e-10 ? covariance / denom : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BidirectionalSelection has not been fitted.");

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
        throw new NotSupportedException("BidirectionalSelection does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BidirectionalSelection has not been fitted.");

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
