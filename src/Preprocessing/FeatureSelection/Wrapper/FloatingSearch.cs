using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Sequential Floating Forward Selection (SFFS) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SFFS is an advanced wrapper method that performs forward selection with
/// conditional backward elimination. After each forward step, it attempts
/// to remove features that may have become redundant.
/// </para>
/// <para><b>For Beginners:</b> Regular forward selection adds features one
/// at a time but never reconsiders its choices. SFFS can "float" - after
/// adding a feature, it checks if any previously added features are now
/// unnecessary and removes them. This leads to better final selections.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FloatingSearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxBacktrack;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxBacktrack => _maxBacktrack;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FloatingSearch(
        int nFeaturesToSelect = 10,
        int maxBacktrack = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxBacktrack < 0)
            throw new ArgumentException("Max backtrack must be non-negative.", nameof(maxBacktrack));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxBacktrack = maxBacktrack;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FloatingSearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // |Pearson r| of every feature with the target
        var featureScores = WrapperScoringOps.AbsTargetCorrelations(data, target, Enumerable.Range(0, p).ToArray());
        var selected = new List<int>();
        var available = new HashSet<int>(Enumerable.Range(0, p));
        var everAdded = new SortedSet<int>();

        while (selected.Count < Math.Min(_nFeaturesToSelect, p))
        {
            // Forward step: add best feature
            int bestFeature = -1;
            T bestScore = NumOps.MinValue;

            foreach (int candidate in available)
            {
                var testSet = selected.Concat(new[] { candidate }).ToArray();
                T score = EvaluateSubset(data, featureScores, testSet);

                if (WrapperScoringOps.IsAbove(score, bestScore))
                {
                    bestScore = score;
                    bestFeature = candidate;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                available.Remove(bestFeature);
                everAdded.Add(bestFeature);
            }

            // Conditional exclusion (backtrack)
            int backtrackCount = 0;
            while (selected.Count > 2 && backtrackCount < _maxBacktrack)
            {
                int worstFeature = -1;
                T bestWithoutOne = EvaluateSubset(data, featureScores, selected.ToArray());

                foreach (int candidate in selected.Take(selected.Count - 1))
                {
                    var testSet = selected.Where(f => f != candidate).ToArray();
                    T score = EvaluateSubset(data, featureScores, testSet);

                    if (WrapperScoringOps.IsAbove(score, bestWithoutOne))
                    {
                        bestWithoutOne = score;
                        worstFeature = candidate;
                    }
                }

                if (worstFeature >= 0)
                {
                    selected.Remove(worstFeature);
                    available.Add(worstFeature);
                    backtrackCount++;
                }
                else
                {
                    break;
                }
            }
        }

        // Each feature ever added keeps its own score (0 for the others)
        var added = everAdded.ToArray();
        _featureScores = WrapperScoringOps.ScoresVector(p, NumOps.Zero, added, WrapperScoringOps.Gather(featureScores, added));
        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private T EvaluateSubset(Matrix<T> data, Tensor<T> featureScores, int[] features)
    {
        if (features.Length == 0) return NumOps.Zero;

        // Total |Pearson r| with the target, penalized by the mean |r| over the feature pairs (redundancy)
        T totalScore = Engine.TensorSum(WrapperScoringOps.Gather(featureScores, features));
        return NumOps.Subtract(totalScore,
            NumOps.Multiply(NumOps.FromDouble(0.5), WrapperScoringOps.MeanAbsPairCorrelation(data, features)));
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
            throw new InvalidOperationException("FloatingSearch has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FloatingSearch does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FloatingSearch has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
