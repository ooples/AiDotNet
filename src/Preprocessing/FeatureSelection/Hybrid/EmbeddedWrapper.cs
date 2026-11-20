using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Embedded-Wrapper hybrid feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Embedded-Wrapper combines embedded methods (like regularization) with wrapper
/// search. It uses embedded feature importance to guide the wrapper search,
/// making the search more efficient.
/// </para>
/// <para><b>For Beginners:</b> This is like having an expert advisor (embedded method)
/// suggest which features might be important, then carefully verifying those
/// suggestions (wrapper). The advisor speeds up the search while the verification
/// ensures accuracy.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class EmbeddedWrapper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lambda;
    private readonly int _maxWrapperIterations;

    private Vector<T>? _embeddedImportance;
    private Vector<T>? _wrapperScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Lambda => _lambda;
    public int MaxWrapperIterations => _maxWrapperIterations;
    public Vector<T>? EmbeddedImportance => _embeddedImportance;
    public Vector<T>? WrapperScores => _wrapperScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public EmbeddedWrapper(
        int nFeaturesToSelect = 10,
        double lambda = 0.1,
        int maxWrapperIterations = 50,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lambda < 0)
            throw new ArgumentException("Lambda must be non-negative.", nameof(lambda));

        _nFeaturesToSelect = nFeaturesToSelect;
        _lambda = lambda;
        _maxWrapperIterations = maxWrapperIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "EmbeddedWrapper requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Stage 1: Embedded - compute L1-regularized importance
        var importance = ComputeL1Importance(data, target);
        _embeddedImportance = importance.ToVector();

        // Get initial candidates from embedded importance, highest first (ties by index)
        var orderedCandidates = PreprocessingTensorOps.StableOrder(importance, descending: true)
            .Take(Math.Min(_nFeaturesToSelect * 2, p))
            .ToArray();

        // Stage 2: Wrapper - refine selection with guided search
        var selected = GuidedForwardSelection(data, target, importance, orderedCandidates, out var wrapperScores);
        _wrapperScores = wrapperScores.ToVector();
        _selectedIndices = selected.OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    /// <summary>
    /// |LASSO coefficients| on the standardized data (population std; a constant column is only centred) and the
    /// centred target: 100 sweeps of cyclic coordinate descent on (1/2n)||y - Xb||^2 + lambda ||b||_1, using the
    /// Gram form rho_j = (X'y)_j / n - sum_{k != j} (X'X)_jk b_k / n.
    /// </summary>
    private Tensor<T> ComputeL1Importance(Matrix<T> data, Vector<T> target)
    {
        var engine = Engine;
        int n = data.Rows;
        int p = data.Columns;

        // Standardize data
        var centered = FilterStatisticTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var std = FeatureSelectionTensorOps.CenteredColumnStd(centered);
        var safeStd = engine.TensorWhere(engine.TensorLessThan(std, NumOps.FromDouble(1e-10)), PreprocessingTensorOps.Filled(new[] { p }, NumOps.One), std);
        var X = engine.TensorDivide(centered, engine.Reshape(safeStd, new[] { 1, p }));
        var yColumn = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
        var y = engine.TensorSubtract(yColumn, engine.ReduceMean(yColumn, new[] { 0 }, true));

        var nT = FeatureSelectionTensorOps.Count<T>(n);
        var gram = engine.TensorDivideScalar(engine.TensorMatMul(engine.TensorTranspose(X), X), nT); // [p, p]
        var correlation = engine.TensorDivideScalar(engine.TensorMatMul(engine.TensorTranspose(X), y), nT); // [p, 1]
        var eye = engine.TensorEye<T>(p);
        T lambda = NumOps.FromDouble(_lambda);
        T negativeLambda = NumOps.Negate(lambda);

        // Coordinate descent LASSO
        var coefficients = new Tensor<T>(new[] { p, 1 });
        const int maxIter = 100;
        for (int iter = 0; iter < maxIter; iter++)
        {
            for (int j = 0; j < p; j++)
            {
                var current = engine.TensorSlice(coefficients, new[] { j, 0 }, new[] { 1, 1 });
                var fit = engine.TensorMatMul(engine.TensorSlice(gram, new[] { j, 0 }, new[] { 1, p }), coefficients);
                var own = engine.TensorMultiply(engine.TensorSlice(gram, new[] { j, j }, new[] { 1, 1 }), current);
                var rho = engine.TensorAdd(engine.TensorSubtract(engine.TensorSlice(correlation, new[] { j, 0 }, new[] { 1, 1 }), fit), own);

                // Soft threshold: S(z, lambda) = z - clamp(z, -lambda, lambda)
                var updated = engine.TensorSubtract(rho, engine.TensorClamp(rho, negativeLambda, lambda));
                var step = engine.TensorBroadcastTo(engine.TensorSubtract(updated, current), new[] { p, 1 });
                coefficients = engine.TensorAdd(coefficients, engine.TensorMultiply(engine.TensorSlice(eye, new[] { 0, j }, new[] { p, 1 }), step));
            }
        }

        return engine.Reshape(engine.TensorAbs(coefficients), new[] { p });
    }

    /// <summary>
    /// Forward selection over the candidates (in descending embedded importance): each step adds the candidate that
    /// most raises the subset score, the mean over the subset of 0.5 * |correlation with the target| +
    /// 0.5 * embedded importance, and stops when no candidate raises it. Ties go to the earlier candidate.
    /// </summary>
    private List<int> GuidedForwardSelection(
        Matrix<T> data, Vector<T> target, Tensor<T> importance, int[] orderedCandidates, out Tensor<T> wrapperScores)
    {
        var engine = Engine;
        int n = data.Rows;
        int p = data.Columns;
        int c = orderedCandidates.Length;
        wrapperScores = new Tensor<T>(new[] { p });
        var selected = new List<int>();
        if (c == 0)
            return selected;

        // Per-feature contribution to the subset score, in candidate order
        var correlation = FeatureSelectionInformationOps.AbsCorrelationWithTarget(
            Tensor<T>.FromMatrix(data), engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), NumOps.Zero);
        var contribution = engine.TensorMultiplyScalar(engine.TensorAdd(correlation, importance), NumOps.FromDouble(0.5));
        var candidateIndex = new Tensor<int>((int[])orderedCandidates.Clone(), new[] { c });
        var candidateContribution = engine.TensorIndexSelect(contribution, candidateIndex, 0);

        var available = PreprocessingTensorOps.Filled(new[] { c }, NumOps.One);
        var selectedSum = new Tensor<T>(new[] { c });
        var currentScore = new Tensor<T>(new[] { c });
        var positionScores = new Tensor<T>(new[] { c });

        int iterations = 0;
        while (selected.Count < _nFeaturesToSelect &&
               selected.Count < c &&
               iterations < _maxWrapperIterations)
        {
            // Score of the subset with each candidate added: (sum of selected + candidate) / (|S| + 1)
            var scores = engine.TensorDivideScalar(
                engine.TensorAdd(selectedSum, candidateContribution), FeatureSelectionTensorOps.Count<T>(selected.Count + 1));
            var improves = engine.TensorMultiply(available, engine.TensorGreaterThan(scores, currentScore));
            var best = FeatureSelectionTensorOps.TopKWhere(scores, improves, 1);
            if (best.Length == 0)
                break;

            int position = best[0];
            var chosen = FeatureSelectionTensorOps.OneHot<T>(position, c);
            selected.Add(orderedCandidates[position]);
            available = engine.TensorMultiply(available, PreprocessingTensorOps.OneMinus(chosen));
            var bestScore = engine.TensorBroadcastTo(engine.TensorSlice(scores, new[] { position }, new[] { 1 }), new[] { c });
            positionScores = engine.TensorAdd(positionScores, engine.TensorMultiply(chosen, bestScore));
            currentScore = bestScore;
            selectedSum = engine.TensorBroadcastTo(engine.TensorSlice(
                engine.TensorAdd(selectedSum, candidateContribution), new[] { position }, new[] { 1 }), new[] { c });
            iterations++;
        }

        // Scatter the scores of the chosen candidates back to their feature positions
        var placement = engine.TensorOneHot<T>(candidateIndex, p); // [c, p]
        wrapperScores = engine.Reshape(engine.TensorMatMul(engine.Reshape(positionScores, new[] { 1, c }), placement), new[] { p });
        return selected;
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
            throw new InvalidOperationException("EmbeddedWrapper has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("EmbeddedWrapper does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EmbeddedWrapper has not been fitted.");

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
