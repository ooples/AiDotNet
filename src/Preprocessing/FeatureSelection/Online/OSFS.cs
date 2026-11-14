using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Online;

/// <summary>
/// Online Streaming Feature Selection (OSFS) of Wu et al. 2013 ("Online Feature Selection with Streaming Features",
/// IEEE TPAMI 35(5)).
/// </summary>
/// <remarks>
/// <para>
/// The features arrive one at a time, in column order. Each new feature X goes through the paper's two steps:
/// <list type="number">
/// <item><b>Online relevance analysis:</b> X is discarded when it is independent of the target C,
/// <c>Ind(C, X | {})</c>.</item>
/// <item><b>Online redundancy analysis:</b> otherwise X joins the best candidate features BCF, and every Y in BCF
/// is removed when some subset S of the other BCF features (|S| up to <see cref="MaxConditioningSetSize"/>) makes
/// it conditionally independent of the target, <c>Ind(C, Y | S)</c>.</item>
/// </list>
/// The independence tests are Fisher's z test on the partial correlation at level <see cref="Alpha"/>, as in the
/// authors' LOFS toolbox (<c>osfs_z</c>): <c>W = sqrt(n - |S| - 3) atanh(r)</c>, independent when the two-sided
/// p-value <c>2 Phi(-|W|)</c> exceeds alpha.
/// </para>
/// <para>
/// OSFS decides the subset size itself. <see cref="NFeaturesToSelect"/> caps it: when BCF ends larger, the features
/// with the strongest correlation with the target are kept.
/// </para>
/// <para><b>For Beginners:</b> Imagine features arriving one by one like a stream.
/// For each new feature, OSFS asks: "Is this related to the target at all?" If yes, it keeps the feature and then
/// re-checks every kept feature: one that tells nothing more about the target once some of the others are known is
/// dropped.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class OSFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly int _maxConditioningSetSize;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>The significance level of the conditional-independence tests.</summary>
    public double Alpha => _alpha;

    /// <summary>The largest conditioning set the redundancy analysis tries.</summary>
    public int MaxConditioningSetSize => _maxConditioningSetSize;

    /// <summary>The absolute Pearson correlation of every feature with the target.</summary>
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates an OSFS selector.
    /// </summary>
    /// <param name="nFeaturesToSelect">The most features to keep.</param>
    /// <param name="alpha">The significance level of the Fisher z tests (LOFS default 0.05).</param>
    /// <param name="maxConditioningSetSize">The largest conditioning set of the redundancy analysis (LOFS default 3).</param>
    /// <param name="columnIndices">The columns to operate on, or null for all columns.</param>
    public OSFS(
        int nFeaturesToSelect = 10,
        double alpha = 0.05,
        int maxConditioningSetSize = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (!(alpha > 0 && alpha < 1))
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be in (0, 1).");
        if (maxConditioningSetSize < 1)
            throw new ArgumentOutOfRangeException(nameof(maxConditioningSetSize), "The conditioning set size must be at least 1.");

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _maxConditioningSetSize = maxConditioningSetSize;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "OSFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Correlation matrix of the features and the target (the target is variable p)
        var variables = engine.TensorConcatenate(
            new[] { Tensor<T>.FromMatrix(data), engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }) }, 1);
        var correlations = CorrelationMatrix(variables);
        var withTarget = engine.Reshape(engine.TensorSlice(correlations, new[] { 0, p }, new[] { p, 1 }), new[] { p });
        var scores = engine.TensorAbs(withTarget);
        _featureScores = scores.ToVector();

        // Fisher z factors sqrt(n - k - 3) for conditioning sets of size k = 0 .. max (sizes holds k + 1)
        int maxSize = _maxConditioningSetSize;
        var sizes = engine.TensorCumSum(PreprocessingTensorOps.Filled(new[] { maxSize + 1 }, NumOps.One), 0);
        var factors = engine.TensorSqrt(engine.TensorMax(
            engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { maxSize + 1 }, NumOps.FromDouble(n - 2)), sizes),
            NumOps.Zero));
        T alpha = NumOps.FromDouble(_alpha);

        // Online relevance analysis needs no conditioning set, so it is one test per feature
        var marginal = FisherPValues(withTarget, engine.TensorSlice(factors, new[] { 0 }, new[] { 1 }));
        var relevant = PreprocessingTensorOps.MaskPositions(
            PreprocessingTensorOps.OneMinus(engine.TensorGreaterThan(marginal, alpha)));

        var bestCandidates = new List<int>();
        foreach (int feature in relevant)
        {
            bestCandidates.Add(feature);

            // Online redundancy analysis over the current BCF, in the order the features arrived
            foreach (int candidate in bestCandidates.ToArray())
            {
                var others = bestCandidates.Where(f => f != candidate).ToArray();
                if (others.Length > 0 && IsRedundant(correlations, candidate, p, others, factors, alpha))
                    bestCandidates.Remove(candidate);
            }
        }

        var kept = bestCandidates.ToArray();
        if (kept.Length > _nFeaturesToSelect)
        {
            var keptScores = engine.TensorIndexSelect(scores, new Tensor<int>((int[])kept.Clone(), new[] { kept.Length }), 0);
            var strongest = PreprocessingTensorOps.TopKIndicesSorted(keptScores, _nFeaturesToSelect);
            kept = strongest.Select(i => kept[i]).ToArray();
        }

        Array.Sort(kept);
        _selectedIndices = kept;
        IsFitted = true;
    }

    /// <summary>
    /// Whether some conditioning set S of the <paramref name="others"/> (1 &lt;= |S| &lt;= the maximum size) makes
    /// <paramref name="candidate"/> independent of the target, <c>Ind(C, Y | S)</c> (LOFS <c>compter_dep_2</c>).
    /// Every subset of one size is tested in one batch.
    /// </summary>
    private bool IsRedundant(Tensor<T> correlations, int candidate, int targetIndex, int[] others, Tensor<T> factors, T alpha)
    {
        var engine = Engine;
        int width = correlations.Shape[0];
        var flat = engine.Reshape(correlations, new[] { width * width });
        int largest = Math.Min(_maxConditioningSetSize, others.Length);
        for (int size = 1; size <= largest; size++)
        {
            var subsets = Combinations(others.Length, size);
            int count = subsets.Count;
            int order = size + 2;

            // Correlation submatrices of (Y, C, S) for every subset S, gathered as [count, order, order]
            var linear = new int[count * order * order];
            var members = new int[order];
            members[0] = candidate;
            members[1] = targetIndex;
            for (int s = 0; s < count; s++)
            {
                for (int k = 0; k < size; k++)
                    members[k + 2] = others[subsets[s][k]];
                for (int a = 0; a < order; a++)
                    for (int b = 0; b < order; b++)
                        linear[(s * order + a) * order + b] = members[a] * width + members[b];
            }

            var blocks = engine.Reshape(
                engine.TensorIndexSelect(flat, new Tensor<int>(linear, new[] { linear.Length }), 0), new[] { count, order, order });
            var partial = PartialCorrelations(blocks, out var defined);
            var factor = engine.TensorSlice(factors, new[] { size }, new[] { 1 });
            var independent = engine.TensorMultiply(engine.TensorGreaterThan(FisherPValues(partial, factor), alpha), defined);
            if (PreprocessingTensorOps.CountNonZero(independent) > 0)
                return true;
        }

        return false;
    }

    /// <summary>
    /// The partial correlation of the first two variables given the rest, for every [order, order] correlation
    /// block: <c>-P01 / sqrt(P00 P11)</c> with <c>P</c> the inverse of the block. <paramref name="defined"/> is 1
    /// where the block could be inverted; a singular block counts as dependent.
    /// </summary>
    private Tensor<T> PartialCorrelations(Tensor<T> blocks, out Tensor<T> defined)
    {
        var engine = Engine;
        int count = blocks.Shape[0];
        int order = blocks.Shape[1];
        var unit = engine.TensorSlice(
            engine.TensorBroadcastTo(engine.Reshape(engine.TensorEye<T>(order), new[] { 1, order, order }), new[] { count, order, order }),
            new[] { 0, 0, 0 }, new[] { count, order, 2 });
        var (inverse, info) = PreprocessingTensorOps.SolveEx(blocks, unit);
        var solved = engine.TensorEquals(info, 0).Cast<T>();

        var p00 = engine.Reshape(engine.TensorSlice(inverse, new[] { 0, 0, 0 }, new[] { count, 1, 1 }), new[] { count });
        var p10 = engine.Reshape(engine.TensorSlice(inverse, new[] { 0, 1, 0 }, new[] { count, 1, 1 }), new[] { count });
        var p11 = engine.Reshape(engine.TensorSlice(inverse, new[] { 0, 1, 1 }, new[] { count, 1, 1 }), new[] { count });
        var product = engine.TensorWhere(solved, engine.TensorMultiply(p00, p11), new Tensor<T>(new[] { count }));
        defined = engine.TensorGreaterThan(product, NumOps.Zero);
        var safeScale = engine.TensorSqrt(engine.TensorWhere(defined, product, PreprocessingTensorOps.Filled(new[] { count }, NumOps.One)));
        var safeOffDiagonal = engine.TensorWhere(defined, p10, new Tensor<T>(new[] { count }));
        return engine.TensorNegate(engine.TensorDivide(safeOffDiagonal, safeScale));
    }

    /// <summary>
    /// Two-sided p-values of Fisher's z test, <c>2 Phi(-|sqrt(n - |S| - 3) atanh(r)|)</c>, for the correlations
    /// <paramref name="r"/> and the [1] factor <c>sqrt(n - |S| - 3)</c>.
    /// </summary>
    private Tensor<T> FisherPValues(Tensor<T> r, Tensor<T> factor)
    {
        var engine = Engine;
        var clipped = engine.TensorMax(engine.TensorMin(r, NumOps.One), NumOps.Negate(NumOps.One));
        var w = engine.TensorMultiply(engine.TensorAtanh(clipped), engine.TensorBroadcastTo(factor, clipped.Shape.ToArray()));
        return engine.TensorMultiplyScalar(engine.TensorNdtr(engine.TensorNegate(engine.TensorAbs(w))), NumOps.FromDouble(2));
    }

    /// <summary>
    /// Pearson correlation matrix of the columns of an [n, v] tensor, with an exact unit diagonal; a constant column
    /// is uncorrelated with every other.
    /// </summary>
    private Tensor<T> CorrelationMatrix(Tensor<T> x)
    {
        var engine = Engine;
        int n = x.Shape[0];
        int v = x.Shape[1];
        var centered = FilterStatisticTensorOps.CenterColumns(x);
        var std = FeatureSelectionTensorOps.CenteredColumnStd(centered);
        var varying = engine.TensorGreaterThan(std, NumOps.Zero);
        var safeStd = engine.Reshape(engine.TensorWhere(varying, std, PreprocessingTensorOps.Filled(new[] { v }, NumOps.One)), new[] { 1, v });
        var standardized = engine.TensorMultiply(
            engine.TensorDivide(centered, safeStd), engine.TensorBroadcastTo(engine.Reshape(varying, new[] { 1, v }), new[] { n, v }));
        var correlations = engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(standardized), standardized), NumOps.FromDouble(Math.Max(n, 1)));
        var eye = engine.TensorEye<T>(v);
        return engine.TensorAdd(engine.TensorMultiply(correlations, PreprocessingTensorOps.OneMinus(eye)), eye);
    }

    /// <summary>All k-subsets of 0 .. count-1 in lexicographic order (MATLAB <c>nchoosek</c> order).</summary>
    private static List<int[]> Combinations(int count, int k)
    {
        var result = new List<int[]>();
        var current = new int[k];
        for (int i = 0; i < k; i++)
            current[i] = i;

        while (true)
        {
            result.Add((int[])current.Clone());
            int position = k - 1;
            while (position >= 0 && current[position] == count - k + position)
                position--;
            if (position < 0)
                return result;

            current[position]++;
            for (int i = position + 1; i < k; i++)
                current[i] = current[i - 1] + 1;
        }
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
            throw new InvalidOperationException("OSFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("OSFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("OSFS has not been fitted.");

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
