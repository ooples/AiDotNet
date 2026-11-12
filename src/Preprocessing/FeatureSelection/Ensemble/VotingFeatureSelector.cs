using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Voting-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines multiple feature selection methods using voting. Each method casts
/// votes for its top features, and features with the most votes are selected.
/// </para>
/// <para><b>For Beginners:</b> Instead of trusting just one method to pick
/// features, this approach asks multiple methods to vote. Features that are
/// chosen by many different methods are more likely to be truly important,
/// giving you more confident selections.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class VotingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _votesPerMethod;

    private int[]? _voteCounts;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int[]? VoteCounts => _voteCounts;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public VotingFeatureSelector(
        int nFeaturesToSelect = 10,
        int votesPerMethod = 20,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _votesPerMethod = votesPerMethod;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "VotingFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // Membership of every row in its (exact) target class
        var classes = PreprocessingTensorOps.SortedUnique(y);
        var classIndex = PreprocessingTensorOps.LookupSorted(classes, y, out _);
        var membership = engine.TensorOneHot<T>(classIndex, classes.Length);

        _voteCounts = new int[p];
        int votesPerMethod = Math.Min(_votesPerMethod, p);

        // Method 1: Correlation
        var corrScores = engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(X, y));
        var corrOrder = PreprocessingTensorOps.StableOrder(corrScores, descending: true);
        foreach (int j in corrOrder.Take(votesPerMethod))
            _voteCounts[j]++;

        // Method 2: Variance (sample variance)
        var varScores = engine.TensorDivideScalar(FeatureSelectionTensorOps.ColumnSumSquaredDeviations(X), NumOps.FromDouble(n - 1));
        foreach (int j in TopK(varScores, votesPerMethod))
            _voteCounts[j]++;

        // Method 3: Fisher score
        var fisherScores = ComputeFisherScore(X, membership, classIndex, n, p);
        foreach (int j in TopK(fisherScores, votesPerMethod))
            _voteCounts[j]++;

        // Method 4: Chi-squared approximation
        var chiScores = ComputeChiSquared(X, membership, n, p);
        foreach (int j in TopK(chiScores, votesPerMethod))
            _voteCounts[j]++;

        // Method 5: Gini importance
        var giniScores = ComputeGiniImportance(X, membership, n, p);
        foreach (int j in TopK(giniScores, votesPerMethod))
            _voteCounts[j]++;

        // Most votes first, ties broken by the larger correlation (then by index)
        var corrRank = new int[p];
        for (int rank = 0; rank < p; rank++)
            corrRank[corrOrder[rank]] = rank;
        var votes = _voteCounts;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => votes[j])
            .ThenBy(j => corrRank[j]) // Tiebreaker
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private static int[] TopK(Tensor<T> scores, int k) =>
        PreprocessingTensorOps.StableOrder(scores, descending: true).Take(k).ToArray();

    private Tensor<T> ComputeFisherScore(Tensor<T> X, Tensor<T> membership, Tensor<int> classIndex, int n, int p)
    {
        // Between-class over within-class sum of squares, 0 when the within-class sum is not above 1e-10
        var engine = Engine;
        int k = membership.Shape[1];
        var counts = engine.ReduceSum(membership, new[] { 0 }, false);
        var countColumn = engine.TensorBroadcastTo(engine.Reshape(counts, new[] { k, 1 }), new[] { k, p });
        var classMeans = engine.TensorDivide(engine.TensorMatMul(engine.TensorTranspose(membership), X), countColumn);
        var overallMean = engine.TensorBroadcastTo(engine.ReduceMean(X, new[] { 0 }, true), new[] { k, p });
        var betweenVar = engine.ReduceSum(engine.TensorMultiply(
            engine.TensorSquare(engine.TensorSubtract(classMeans, overallMean)), countColumn), new[] { 0 }, false);
        var withinVar = engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(X,
            engine.TensorIndexSelect(classMeans, classIndex, 0))), new[] { 0 }, false);

        var usable = engine.TensorGreaterThan(withinVar, NumOps.FromDouble(1e-10));
        var safe = engine.TensorWhere(usable, withinVar, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        return engine.TensorWhere(usable, engine.TensorDivide(betweenVar, safe), new Tensor<T>(new[] { p }));
    }

    private Tensor<T> ComputeChiSquared(Tensor<T> X, Tensor<T> membership, int n, int p)
    {
        // Chi-squared of the equal-width bins (width (max - min) / bins + 1e-10) against the classes
        var engine = Engine;
        int nBins = Math.Min(10, (int)Math.Sqrt(n) + 1);
        int k = membership.Shape[1];
        var bins = PreprocessingTensorOps.EqualWidthBinOneHot(X, nBins, out _);
        var observed = engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(bins), membership), new[] { p, nBins, k });
        var cellShape = new[] { p, nBins, k };
        var rowSums = engine.TensorBroadcastTo(engine.ReduceSum(observed, new[] { 2 }, true), cellShape);
        var colSums = engine.TensorBroadcastTo(
            engine.Reshape(engine.ReduceSum(membership, new[] { 0 }, false), new[] { 1, 1, k }), cellShape);
        var expected = engine.TensorDivideScalar(engine.TensorMultiply(rowSums, colSums), NumOps.FromDouble(n));
        var positive = engine.TensorGreaterThan(expected, NumOps.Zero);
        var safe = engine.TensorWhere(positive, expected, PreprocessingTensorOps.Filled(cellShape, NumOps.One));
        var terms = engine.TensorWhere(positive,
            engine.TensorDivide(engine.TensorSquare(engine.TensorSubtract(observed, expected)), safe), new Tensor<T>(cellShape));
        return engine.ReduceSum(terms, new[] { 1, 2 }, false);
    }

    private Tensor<T> ComputeGiniImportance(Tensor<T> X, Tensor<T> membership, int n, int p)
    {
        // Gini decrease of splitting every feature at its mean (x <= mean goes left); 0 when a side is empty
        var engine = Engine;
        int k = membership.Shape[1];
        T giniParent = FeatureSelectionTensorOps.GiniImpurity(membership);
        var right = engine.TensorGreaterThan(X, engine.TensorBroadcastTo(engine.ReduceMean(X, new[] { 0 }, true), new[] { n, p }));
        var left = PreprocessingTensorOps.Not(right);
        var ones = PreprocessingTensorOps.Filled(new[] { p }, NumOps.One);
        T zero = NumOps.Zero;

        Tensor<T> SideGini(Tensor<T> side, out Tensor<T> size)
        {
            var classCounts = engine.TensorMatMul(engine.TensorTranspose(side), membership);
            size = engine.ReduceSum(side, new[] { 0 }, false);
            var safeSize = engine.TensorWhere(engine.TensorGreaterThan(size, zero), size, ones);
            var proportions = engine.TensorDivide(classCounts,
                engine.TensorBroadcastTo(engine.Reshape(safeSize, new[] { p, 1 }), new[] { p, k }));
            return PreprocessingTensorOps.OneMinus(engine.ReduceSum(engine.TensorSquare(proportions), new[] { 1 }, false));
        }

        var giniLeft = SideGini(left, out var leftSize);
        var giniRight = SideGini(right, out var rightSize);
        var weighted = engine.TensorDivideScalar(engine.TensorAdd(
            engine.TensorMultiply(leftSize, giniLeft), engine.TensorMultiply(rightSize, giniRight)), NumOps.FromDouble(n));
        var bothSides = engine.TensorMultiply(engine.TensorGreaterThan(leftSize, zero),
            engine.TensorGreaterThan(rightSize, zero));
        return engine.TensorWhere(bothSides,
            engine.TensorAddScalar(engine.TensorNegate(weighted), giniParent), new Tensor<T>(new[] { p }));
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
            throw new InvalidOperationException("VotingFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("VotingFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VotingFeatureSelector has not been fitted.");

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
