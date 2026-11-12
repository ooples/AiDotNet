using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

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
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FilterWrapper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _filterRatio;
    private readonly bool _useForwardSelection;

    private Vector<T>? _filterScores;
    private Vector<T>? _wrapperScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int FilterRatio => _filterRatio;
    public bool UseForwardSelection => _useForwardSelection;
    public Vector<T>? FilterScores => _filterScores;
    public Vector<T>? WrapperScores => _wrapperScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
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

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FilterWrapper requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });

        // Stage 1: Filter - use correlation to select top candidates (highest first, ties by index)
        var filterScores = FeatureSelectionInformationOps.AbsCorrelationWithTarget(Tensor<T>.FromMatrix(data), y, NumOps.Zero);
        _filterScores = filterScores.ToVector();
        int numFilterSelect = Math.Min(_nFeaturesToSelect * _filterRatio, p);
        var filterCandidates = PreprocessingTensorOps.StableOrder(filterScores, descending: true)
            .Take(Math.Max(numFilterSelect, 0))
            .ToArray();

        // The subset score is the mean filter score of its features (a proxy for R²), or 0 for a constant target
        var centeredY = engine.TensorSubtract(y, engine.ReduceMean(y, new[] { 0 }, true));
        T ssTot = engine.TensorSum(engine.TensorSquare(centeredY));
        int c = filterCandidates.Length;
        var candidateIndex = new Tensor<int>((int[])filterCandidates.Clone(), new[] { c });
        var contribution = NumOps.LessThan(ssTot, NumOps.FromDouble(1e-10))
            ? new Tensor<T>(new[] { c })
            : engine.TensorIndexSelect(filterScores, candidateIndex, 0);

        // Stage 2: Wrapper - apply sequential selection on filtered candidates
        Tensor<T> positionScores;
        List<int> selectedPositions = _useForwardSelection
            ? ForwardSelection(contribution, out positionScores)
            : BackwardElimination(contribution, engine.TensorIndexSelect(filterScores, candidateIndex, 0), out positionScores);

        var placement = engine.TensorOneHot<T>(candidateIndex, p); // [c, p]
        _wrapperScores = c == 0
            ? new Vector<T>(p)
            : engine.Reshape(engine.TensorMatMul(engine.Reshape(positionScores, new[] { 1, c }), placement), new[] { p }).ToVector();
        _selectedIndices = selectedPositions.Select(position => filterCandidates[position]).OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    /// <summary>
    /// Forward selection over the candidate positions: each step adds the candidate that most raises the mean
    /// contribution of the subset and stops when none raises it; ties go to the earlier candidate.
    /// </summary>
    private List<int> ForwardSelection(Tensor<T> contribution, out Tensor<T> positionScores)
    {
        var engine = Engine;
        int c = contribution.Length;
        var selected = new List<int>();
        positionScores = new Tensor<T>(new[] { c });
        if (c == 0)
            return selected;

        var available = PreprocessingTensorOps.Filled(new[] { c }, NumOps.One);
        var selectedSum = new Tensor<T>(new[] { c });
        var currentScore = new Tensor<T>(new[] { c });

        while (selected.Count < _nFeaturesToSelect && selected.Count < c)
        {
            var scores = engine.TensorDivideScalar(
                engine.TensorAdd(selectedSum, contribution), FeatureSelectionTensorOps.Count<T>(selected.Count + 1));
            var improves = engine.TensorMultiply(available, engine.TensorGreaterThan(scores, currentScore));
            var best = FeatureSelectionTensorOps.TopKWhere(scores, improves, 1);
            if (best.Length == 0)
                break;

            int position = best[0];
            var chosen = FeatureSelectionTensorOps.OneHot<T>(position, c);
            selected.Add(position);
            available = engine.TensorMultiply(available, PreprocessingTensorOps.OneMinus(chosen));
            var bestScore = engine.TensorBroadcastTo(engine.TensorSlice(scores, new[] { position }, new[] { 1 }), new[] { c });
            positionScores = engine.TensorAdd(positionScores, engine.TensorMultiply(chosen, bestScore));
            currentScore = bestScore;
            selectedSum = engine.TensorBroadcastTo(engine.TensorSlice(
                engine.TensorAdd(selectedSum, contribution), new[] { position }, new[] { 1 }), new[] { c });
        }

        return selected;
    }

    /// <summary>
    /// Backward elimination over the candidate positions: while more than the target count remain, drop the
    /// candidate whose removal leaves the highest mean contribution (ties drop the earlier candidate). The kept
    /// candidates are scored with their filter scores.
    /// </summary>
    private List<int> BackwardElimination(Tensor<T> contribution, Tensor<T> candidateFilterScores, out Tensor<T> positionScores)
    {
        var engine = Engine;
        int c = contribution.Length;
        var kept = PreprocessingTensorOps.Filled(new[] { c }, NumOps.One);
        int count = c;

        while (count > _nFeaturesToSelect)
        {
            // Mean contribution of the subset without each candidate: (sum - contribution) / (count - 1)
            var total = engine.TensorBroadcastTo(
                engine.Reshape(engine.ReduceSum(engine.TensorMultiply(kept, contribution), new[] { 0 }, true), new[] { 1 }), new[] { c });
            var scores = engine.TensorDivideScalar(engine.TensorSubtract(total, contribution), FeatureSelectionTensorOps.Count<T>(count - 1));
            var worst = FeatureSelectionTensorOps.TopKWhere(scores, kept, 1);
            if (worst.Length == 0)
                break;

            kept = engine.TensorMultiply(kept, PreprocessingTensorOps.OneMinus(FeatureSelectionTensorOps.OneHot<T>(worst[0], c)));
            count--;
        }

        positionScores = engine.TensorMultiply(kept, candidateFilterScores);
        return PreprocessingTensorOps.MaskPositions(kept).ToList();
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
            throw new InvalidOperationException("FilterWrapper has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FilterWrapper does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FilterWrapper has not been fitted.");

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
