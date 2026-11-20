using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Backward Elimination (Sequential Backward Selection) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Starts with all features and greedily removes one feature at a time whose removal
/// causes the least degradation in performance until the desired number is reached.
/// </para>
/// <para><b>For Beginners:</b> Backward Elimination is like downsizing a team. You
/// start with everyone and repeatedly remove the person whose absence hurts performance
/// least. Unlike Forward Selection, this can find features that are only valuable
/// together, since they start included.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BackwardElimination<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly Func<Matrix<T>, Vector<T>, int[], T>? _scoringFunction;

    private Vector<T>? _removalScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? RemovalScores => _removalScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BackwardElimination(
        int nFeaturesToSelect = 10,
        Func<Matrix<T>, Vector<T>, int[], T>? scoringFunction = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _scoringFunction = scoringFunction;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BackwardElimination requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int nToSelect = Math.Min(_nFeaturesToSelect, p);

        var remaining = Enumerable.Range(0, p).ToList();
        // Removal scores (the maximum value for the features never removed)
        var scores = new Dictionary<int, T>();

        while (remaining.Count > nToSelect)
        {
            int worstFeature = -1;
            T bestScoreAfterRemoval = NumOps.MinValue;

            foreach (int candidate in remaining)
            {
                var testSet = remaining.Where(x => x != candidate).ToArray();
                T score = EvaluateSubset(data, target, testSet);

                if (WrapperScoringOps.IsAbove(score, bestScoreAfterRemoval))
                {
                    bestScoreAfterRemoval = score;
                    worstFeature = candidate;
                }
            }

            if (worstFeature >= 0)
            {
                scores[worstFeature] = bestScoreAfterRemoval;
                remaining.Remove(worstFeature);
            }
            else
            {
                break;
            }
        }

        _removalScores = WrapperScoringOps.ScoresVector(p, NumOps.MaxValue, scores);
        _selectedIndices = remaining.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private T EvaluateSubset(Matrix<T> data, Vector<T> target, int[] features)
    {
        if (features.Length == 0)
            return NumOps.MinValue;

        if (_scoringFunction is not null)
            return _scoringFunction(data, target, features);

        return DefaultScoring(data, target, features);
    }

    private T DefaultScoring(Matrix<T> data, Vector<T> target, int[] features)
    {
        // R-squared of the least-squares fit (with intercept) of the target on the subset, as scikit-learn's
        // LinearRegression().fit(X, y).score(X, y); the sum of squared univariate correlations it replaces is only
        // the R-squared when the features are uncorrelated
        int n = data.Rows;
        if (features.Length == 0 || n <= features.Length)
            return NumOps.Zero;

        return WrapperScoringOps.LinearRegressionRSquared(data, target, features);
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BackwardElimination has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BackwardElimination does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BackwardElimination has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
