using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Filter-Wrapper hybrid feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Filter-Wrapper combines the speed of filter methods with the accuracy of
/// wrapper methods. It first uses a filter to reduce the feature space, then
/// applies a wrapper method on the reduced set for fine-tuning.
/// </para>
/// <para><b>For Beginners:</b> Think of it as a two-stage interview process.
/// First, resumes are quickly screened (filter) to remove obviously unqualified
/// candidates. Then, the remaining candidates go through detailed interviews
/// (wrapper) to select the best ones. This is faster than interviewing everyone.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FilterWrapper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _filterRatio;
    private readonly bool _useForwardSelection;

    private double[]? _filterScores;
    private double[]? _wrapperScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int FilterRatio => _filterRatio;
    public bool UseForwardSelection => _useForwardSelection;
    public double[]? FilterScores => _filterScores;
    public double[]? WrapperScores => _wrapperScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FilterWrapper(
        int nFeaturesToSelect = 10,
        int filterRatio = 3,
        bool useForwardSelection = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (filterRatio < 1)
            throw new ArgumentException("Filter ratio must be at least 1.", nameof(filterRatio));

        _nFeaturesToSelect = nFeaturesToSelect;
        _filterRatio = filterRatio;
        _useForwardSelection = useForwardSelection;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FilterWrapper requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Stage 1: Filter - use correlation to select top candidates
        _filterScores = ComputeCorrelationScores(data, target);

        int numFilterSelect = Math.Min(_nFeaturesToSelect * _filterRatio, p);
        var filterCandidates = _filterScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numFilterSelect)
            .Select(x => x.Index)
            .ToList();

        // Stage 2: Wrapper - apply sequential selection on filtered candidates
        _wrapperScores = new double[p];
        HashSet<int> selectedSet;

        if (_useForwardSelection)
        {
            selectedSet = ForwardSelection(data, target, filterCandidates);
        }
        else
        {
            selectedSet = BackwardElimination(data, target, filterCandidates);
        }

        _selectedIndices = selectedSet.OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    private double[] ComputeCorrelationScores(Matrix<T> data, Vector<T> target)
    {
        int n = data.Rows;
        int p = data.Columns;
        var scores = new double[p];

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            scores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        return scores;
    }

    private HashSet<int> ForwardSelection(Matrix<T> data, Vector<T> target, List<int> candidates)
    {
        var selected = new HashSet<int>();
        var available = new HashSet<int>(candidates);
        double currentScore = 0;

        while (selected.Count < _nFeaturesToSelect && available.Count > 0)
        {
            int bestFeature = -1;
            double bestScore = currentScore;

            foreach (int j in available)
            {
                selected.Add(j);
                double score = EvaluateSubset(data, target, selected);
                selected.Remove(j);

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                available.Remove(bestFeature);
                currentScore = bestScore;
                _wrapperScores![bestFeature] = bestScore;
            }
            else
            {
                break;
            }
        }

        return selected;
    }

    private HashSet<int> BackwardElimination(Matrix<T> data, Vector<T> target, List<int> candidates)
    {
        var selected = new HashSet<int>(candidates);
        double currentScore = EvaluateSubset(data, target, selected);

        while (selected.Count > _nFeaturesToSelect)
        {
            int worstFeature = -1;
            double bestScoreAfterRemoval = double.MinValue;

            foreach (int j in selected)
            {
                selected.Remove(j);
                double score = EvaluateSubset(data, target, selected);
                selected.Add(j);

                if (score > bestScoreAfterRemoval)
                {
                    bestScoreAfterRemoval = score;
                    worstFeature = j;
                }
            }

            if (worstFeature >= 0)
            {
                selected.Remove(worstFeature);
                currentScore = bestScoreAfterRemoval;
            }
            else
            {
                break;
            }
        }

        foreach (int j in selected)
            _wrapperScores![j] = _filterScores![j];

        return selected;
    }

    private double EvaluateSubset(Matrix<T> data, Vector<T> target, HashSet<int> subset)
    {
        if (subset.Count == 0) return 0;

        // Simple R² approximation using selected features
        int n = data.Rows;

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        double ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = NumOps.ToDouble(target[i]) - yMean;
            ssTot += diff * diff;
        }

        if (ssTot < 1e-10) return 0;

        // Use sum of correlations as proxy for R²
        double score = 0;
        foreach (int j in subset)
            score += _filterScores![j];

        return score / subset.Count;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FilterWrapper has not been fitted.");

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
        throw new NotSupportedException("FilterWrapper does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FilterWrapper has not been fitted.");

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
