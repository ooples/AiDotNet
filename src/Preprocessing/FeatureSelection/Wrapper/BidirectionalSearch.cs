using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Bidirectional Feature Search for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Bidirectional search combines forward selection and backward elimination in an
/// interleaved manner. It can add or remove features at each step, potentially
/// finding better solutions than either method alone.
/// </para>
/// <para><b>For Beginners:</b> Instead of just adding features (forward) or just
/// removing them (backward), this method does both alternately. This can escape
/// local optima and find feature combinations that neither approach would find alone.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BidirectionalSearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxIterations;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxIterations => _maxIterations;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BidirectionalSearch(
        int nFeaturesToSelect = 10,
        int maxIterations = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxIterations = maxIterations;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BidirectionalSearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // |Pearson r| of every feature with the target; a subset scores the mean of its features' scores
        var featureScores = WrapperScoringOps.AbsTargetCorrelations(data, target, Enumerable.Range(0, p).ToArray());
        var selected = new HashSet<int>();
        var notSelected = new HashSet<int>(Enumerable.Range(0, p));

        // Start with half features randomly selected
        int initialCount = Math.Min(_nFeaturesToSelect, p / 2 + 1);
        var initialFeatures = Enumerable.Range(0, p).OrderBy(_ => random.Next()).Take(initialCount);
        foreach (int f in initialFeatures)
        {
            selected.Add(f);
            notSelected.Remove(f);
        }

        T currentScore = EvaluateSubset(featureScores, selected.ToArray());
        bool improved = true;

        for (int iter = 0; iter < _maxIterations && improved; iter++)
        {
            improved = false;

            // Forward step: try adding features
            if (selected.Count < _nFeaturesToSelect)
            {
                int bestAdd = -1;
                T bestAddScore = currentScore;

                foreach (int candidate in notSelected)
                {
                    selected.Add(candidate);
                    T score = EvaluateSubset(featureScores, selected.ToArray());
                    selected.Remove(candidate);

                    if (WrapperScoringOps.IsAbove(score, bestAddScore))
                    {
                        bestAddScore = score;
                        bestAdd = candidate;
                    }
                }

                if (bestAdd >= 0)
                {
                    selected.Add(bestAdd);
                    notSelected.Remove(bestAdd);
                    currentScore = bestAddScore;
                    improved = true;
                }
            }

            // Backward step: try removing features
            if (selected.Count > 1)
            {
                int bestRemove = -1;
                T bestRemoveScore = currentScore;

                foreach (int candidate in selected.ToList())
                {
                    selected.Remove(candidate);
                    T score = EvaluateSubset(featureScores, selected.ToArray());
                    selected.Add(candidate);

                    if (WrapperScoringOps.IsAbove(score, bestRemoveScore))
                    {
                        bestRemoveScore = score;
                        bestRemove = candidate;
                    }
                }

                if (bestRemove >= 0)
                {
                    selected.Remove(bestRemove);
                    notSelected.Add(bestRemove);
                    currentScore = bestRemoveScore;
                    improved = true;
                }
            }
        }

        // Adjust to exact number of features requested: drop the weakest selected features, then add the strongest
        // unselected ones (each by its own score; the lowest index first among ties)
        var ascending = PreprocessingTensorOps.StableOrder(featureScores, descending: false);
        while (selected.Count > _nFeaturesToSelect)
        {
            int weakest = ascending.First(f => selected.Contains(f));
            selected.Remove(weakest);
            notSelected.Add(weakest);
        }

        var descending = PreprocessingTensorOps.StableOrder(featureScores, descending: true);
        while (selected.Count < _nFeaturesToSelect && notSelected.Any())
        {
            int best = descending.First(f => notSelected.Contains(f));
            selected.Add(best);
            notSelected.Remove(best);
        }

        var chosen = selected.ToArray();
        _featureScores = WrapperScoringOps.ScoresVector(p, NumOps.Zero, chosen, WrapperScoringOps.Gather(featureScores, chosen));

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private T EvaluateSubset(Tensor<T> featureScores, int[] features)
    {
        if (features.Length == 0) return NumOps.Zero;

        return Engine.TensorMean(WrapperScoringOps.Gather(featureScores, features));
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
            throw new InvalidOperationException("BidirectionalSearch has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BidirectionalSearch does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BidirectionalSearch has not been fitted.");

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
