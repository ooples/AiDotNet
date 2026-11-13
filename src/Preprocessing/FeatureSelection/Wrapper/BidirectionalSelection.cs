using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BidirectionalSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alphaAdd;
    private readonly double _alphaRemove;
    private readonly Func<Matrix<T>, Vector<T>, int[], T>? _scoringFunction;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BidirectionalSelection(
        int nFeaturesToSelect = 10,
        double alphaAdd = 0.05,
        double alphaRemove = 0.10,
        Func<Matrix<T>, Vector<T>, int[], T>? scoringFunction = null,
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

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BidirectionalSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var scorer = _scoringFunction ?? DefaultScorer;

        var selected = new HashSet<int>();
        var available = Enumerable.Range(0, p).ToHashSet();
        var importances = new Dictionary<int, T>();
        T alphaAdd = NumOps.FromDouble(_alphaAdd);
        T alphaRemove = NumOps.FromDouble(_alphaRemove);
        T currentScore = NumOps.Zero;
        T negativeInfinity = NumOps.FromDouble(double.NegativeInfinity);

        while (selected.Count < _nFeaturesToSelect && available.Count > 0)
        {
            bool changed = false;

            // Forward step: try to add best feature
            int bestToAdd = -1;
            T bestAddScore = currentScore;

            foreach (int j in available)
            {
                var candidate = selected.Append(j).ToArray();
                T score = scorer(data, target, candidate);
                T improvement = WrapperScoringOps.Subtract(score, currentScore);

                if (WrapperScoringOps.IsAbove(improvement, alphaAdd) && WrapperScoringOps.IsAbove(score, bestAddScore))
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
                importances[bestToAdd] = bestAddScore;
                changed = true;
            }

            // Backward step: try to remove worst feature
            if (selected.Count > 1)
            {
                int worstToRemove = -1;
                T bestRemoveScore = negativeInfinity;

                foreach (int j in selected)
                {
                    var candidate = selected.Where(x => x != j).ToArray();
                    T score = scorer(data, target, candidate);
                    T degradation = WrapperScoringOps.Subtract(currentScore, score);

                    if (WrapperScoringOps.IsBelow(degradation, alphaRemove) && WrapperScoringOps.IsAbove(score, bestRemoveScore))
                    {
                        bestRemoveScore = score;
                        worstToRemove = j;
                    }
                }

                if (worstToRemove >= 0 && WrapperScoringOps.IsAbove(bestRemoveScore, WrapperScoringOps.Subtract(currentScore, alphaRemove)))
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

        _featureImportances = WrapperScoringOps.ScoresVector(p, NumOps.Zero, importances);
        _selectedIndices = selected.OrderBy(x => x).ToArray();

        if (_selectedIndices.Length == 0)
        {
            // Fall back to top by individual correlation
            var correlations = Engine.TensorAbs(PreprocessingTensorOps.CorrelationsWithDenominatorGuard(
                Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target)));
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(correlations, Math.Min(_nFeaturesToSelect, p));
        }

        IsFitted = true;
    }

    private T DefaultScorer(Matrix<T> data, Vector<T> target, int[] features)
    {
        if (features.Length == 0) return NumOps.Zero;

        // Use sum of squared correlations as default score
        var correlations = PreprocessingTensorOps.CorrelationsWithDenominatorGuard(
            WrapperScoringOps.Columns(data, features), Tensor<T>.FromVector(target));
        T score = Engine.TensorSumOfSquares(correlations);

        // Penalize feature redundancy: score (1 - redundancy / 2) with the mean |r| over the feature pairs
        if (features.Length > 1)
        {
            T redundancy = WrapperScoringOps.MeanAbsPairCorrelationWithDenominatorGuard(data, features);
            score = NumOps.Subtract(score, NumOps.Multiply(NumOps.Multiply(redundancy, score), NumOps.FromDouble(0.5)));
        }

        return score;
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
            throw new InvalidOperationException("BidirectionalSelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BidirectionalSelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BidirectionalSelection has not been fitted.");

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
