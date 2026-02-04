using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Sequential Backward Floating Selection (SBFS) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SBFS is a wrapper method that starts with all features and iteratively removes
/// the least useful ones. Unlike simple backward elimination, SBFS can conditionally
/// add features back if their removal was a mistake given later removals.
/// </para>
/// <para><b>For Beginners:</b> SBFS starts with all features and removes them one
/// by one, but it's smart enough to add features back if removing them was a bad
/// idea. It's like cleaning out a closet - you might throw something away, then
/// realize you need it after throwing other things out.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SBFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxIterations => _maxIterations;
    public double Tolerance => _tolerance;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SBFS(
        int nFeaturesToSelect = 10,
        int maxIterations = 100,
        double tolerance = 1e-6,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SBFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // Compute individual feature scores
        _featureScores = ComputeFeatureScores(data, target);

        // Start with all features
        var currentSet = new HashSet<int>(Enumerable.Range(0, p));
        var removedSet = new HashSet<int>();
        double currentScore = EvaluateSubset(data, target, currentSet);

        for (int iter = 0; iter < _maxIterations && currentSet.Count > _nFeaturesToSelect; iter++)
        {
            // Backward step: remove least significant feature
            int worstFeature = -1;
            double bestScoreAfterRemoval = double.MinValue;

            foreach (int j in currentSet)
            {
                currentSet.Remove(j);
                double score = EvaluateSubset(data, target, currentSet);
                currentSet.Add(j);

                if (score > bestScoreAfterRemoval)
                {
                    bestScoreAfterRemoval = score;
                    worstFeature = j;
                }
            }

            if (worstFeature >= 0)
            {
                currentSet.Remove(worstFeature);
                removedSet.Add(worstFeature);
                currentScore = bestScoreAfterRemoval;
            }

            // Conditional forward step: add back features if they improve score
            if (removedSet.Count > 0 && currentSet.Count < p)
            {
                int bestToAdd = -1;
                double bestScoreAfterAdding = currentScore;

                foreach (int j in removedSet)
                {
                    currentSet.Add(j);
                    double score = EvaluateSubset(data, target, currentSet);
                    currentSet.Remove(j);

                    if (score > bestScoreAfterAdding + _tolerance)
                    {
                        bestScoreAfterAdding = score;
                        bestToAdd = j;
                    }
                }

                if (bestToAdd >= 0)
                {
                    currentSet.Add(bestToAdd);
                    removedSet.Remove(bestToAdd);
                    currentScore = bestScoreAfterAdding;
                }
            }
        }

        _selectedIndices = currentSet.OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    private double[] ComputeFeatureScores(Matrix<T> data, Vector<T> target)
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

    private double EvaluateSubset(Matrix<T> data, Vector<T> target, HashSet<int> subset)
    {
        if (subset.Count == 0) return 0;

        // Use sum of individual feature scores as a simple evaluation
        double score = 0;
        foreach (int j in subset)
            score += _featureScores![j];

        // Bonus for having the right number of features
        int diff = Math.Abs(subset.Count - _nFeaturesToSelect);
        score -= diff * 0.01;

        return score;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SBFS has not been fitted.");

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
        throw new NotSupportedException("SBFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SBFS has not been fitted.");

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
