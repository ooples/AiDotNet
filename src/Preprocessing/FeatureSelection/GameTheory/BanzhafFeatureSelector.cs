using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.GameTheory;

/// <summary>
/// Banzhaf Power Index-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Banzhaf power index from game theory to measure feature importance
/// based on the probability of being a swing voter (decisive feature).
/// </para>
/// <para><b>For Beginners:</b> The Banzhaf index measures how often adding a
/// feature changes the outcome (makes a coalition successful). Unlike Shapley,
/// it treats all coalition sizes equally. Features with high Banzhaf values are
/// "swing voters" that often make the difference.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BanzhafFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSamples;
    private readonly double _threshold;
    private readonly int? _randomState;

    private Vector<T>? _banzhafIndices;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Threshold => _threshold;
    public Vector<T>? BanzhafIndices => _banzhafIndices;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BanzhafFeatureSelector(
        int nFeaturesToSelect = 10,
        int nSamples = 100,
        double threshold = 0.5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nSamples = nSamples;
        _threshold = threshold;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BanzhafFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Random coalitions (each feature included with 50% probability), drawn sample by sample
        var membership = new int[_nSamples * p];
        for (int k = 0; k < membership.Length; k++)
            membership[k] = rand.NextDouble() < 0.5 ? 1 : 0;

        // a coalition scores the mean absolute correlation of its features with the target (0 when empty)
        var correlation = AbsoluteCorrelation(FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.AsTensor(target));
        var memberCodes = new Tensor<int>(membership, new[] { _nSamples, p });
        var inCoalition = memberCodes.Cast<T>();
        var shape = new[] { _nSamples, p };
        var sums = Engine.TensorBroadcastTo(
            Engine.TensorMatMul(inCoalition, Engine.Reshape(correlation, new[] { p, 1 })), shape);
        // coalition sizes counted in int (a 0/1 sum in T stops counting at 2048 for Half)
        var counts = Engine.TensorBroadcastTo(Engine.ReduceSum(memberCodes, new[] { 1 }, true).Cast<T>(), shape);
        T threshold = NumOps.FromDouble(_threshold);
        var wins = OneMinusBelow(CoalitionScore(sums, counts), threshold);

        // flipping feature j: add it (+c_j, +1) when absent, remove it (-c_j, -1) when present
        var direction = Engine.TensorSubtract(
            FilterStatisticTensorOps.FilledLike(inCoalition, NumOps.One), Engine.TensorMultiplyScalar(inCoalition, NumOps.FromDouble(2)));
        var flippedSums = Engine.TensorAdd(sums, Engine.TensorMultiply(direction, FilterStatisticTensorOps.RowBroadcast(correlation, _nSamples)));
        var flippedWins = OneMinusBelow(CoalitionScore(flippedSums, Engine.TensorAdd(counts, direction)), threshold);

        // Banzhaf index: the share of sampled coalitions in which the feature is a swing voter
        var swings = Engine.ReduceSum(Engine.TensorAbs(Engine.TensorSubtract(wins, flippedWins)), new[] { 0 }, false);
        var indices = Engine.TensorDivideScalar(swings, NumOps.FromDouble(_nSamples));
        _banzhafIndices = indices.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(indices, Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    // mean of the member correlations: sum / count, 0 for an empty coalition
    private static Tensor<T> CoalitionScore(Tensor<T> sums, Tensor<T> counts) =>
        FilterStatisticTensorOps.GuardedDivide(sums, counts, 0.5, MathHelper.GetNumericOperations<T>().Zero);

    // 1 where the value reaches the threshold, else 0
    private Tensor<T> OneMinusBelow(Tensor<T> values, T threshold) =>
        PreprocessingTensorOps.OneMinus(Engine.TensorLessThan(values, threshold));

    // |Pearson r| of every column with the target; 0 when the column or the target is (near) constant
    private Tensor<T> AbsoluteCorrelation(Tensor<T> x, Tensor<T> y)
    {
        var (sxx, sxy, syy) = FilterStatisticTensorOps.CenteredCrossSums(x, y);
        T tolerance = NumOps.FromDouble(1e-10);
        if (!NumOps.GreaterThan(syy, tolerance))
            return FilterStatisticTensorOps.FilledLike(sxx, NumOps.Zero);
        var correlation = Engine.TensorAbs(FilterStatisticTensorOps.GuardedDivide(
            sxy, Engine.TensorSqrt(Engine.TensorMultiplyScalar(sxx, syy)), 0, NumOps.Zero));
        return FeatureSelectionTensorOps.KeepOrFill(correlation, Engine.TensorGreaterThan(sxx, tolerance), NumOps.Zero);
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
            throw new InvalidOperationException("BanzhafFeatureSelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BanzhafFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BanzhafFeatureSelector has not been fitted.");

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
