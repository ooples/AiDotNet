using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.HighDimensional;

/// <summary>
/// Knockoff Filter for high-dimensional feature selection with FDR control.
/// </summary>
/// <remarks>
/// <para>
/// The Knockoff Filter creates "knockoff" versions of each feature that mimic the
/// correlation structure but are conditionally independent of the target. By comparing
/// real features against their knockoffs, it controls the False Discovery Rate (FDR).
/// </para>
/// <para><b>For Beginners:</b> When testing many features, some will look important
/// just by chance. The Knockoff Filter creates fake versions of each feature and
/// asks: "Is the real feature more predictive than its fake twin?" This helps avoid
/// selecting features that only appear important by luck.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KnockoffFilter<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _fdrThreshold;
    private readonly int? _randomState;

    private Vector<T>? _featureStatistics;
    private Vector<T>? _knockoffStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double FDRThreshold => _fdrThreshold;
    public Vector<T>? FeatureStatistics => _featureStatistics;
    public Vector<T>? KnockoffStatistics => _knockoffStatistics;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public KnockoffFilter(
        int nFeaturesToSelect = 10,
        double fdrThreshold = 0.1,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (fdrThreshold <= 0 || fdrThreshold > 1)
            throw new ArgumentException("FDR threshold must be between 0 and 1.", nameof(fdrThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _fdrThreshold = fdrThreshold;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KnockoffFilter requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Step 1: Generate knockoff features (simplified second-order knockoffs)
        var x = FilterStatisticTensorOps.AsTensor(data);
        var knockoffs = GenerateKnockoffs(x, random);

        // Step 2: absolute correlation with the target for original and knockoff features
        var y = FilterStatisticTensorOps.AsTensor(target);
        var featureStatistics = AbsoluteCorrelation(x, y);
        var knockoffStatistics = AbsoluteCorrelation(knockoffs, y);
        _featureStatistics = featureStatistics.ToVector();
        _knockoffStatistics = knockoffStatistics.ToVector();

        // Step 3: W = |Z_j| - |Z_knockoff_j|
        var w = Engine.TensorSubtract(featureStatistics, knockoffStatistics);

        // Step 4: knockoff+ threshold. For every candidate tau = W_k: #{j : W_j >= tau} and #{j : W_j <= -tau}
        var order = PreprocessingTensorOps.StableOrder(w, descending: true);
        var wj = FilterStatisticTensorOps.RowBroadcast(w, p);
        var tau = FilterStatisticTensorOps.ColumnBroadcast(w, p);
        var positives = Engine.ReduceSum(PreprocessingTensorOps.OneMinus(Engine.TensorLessThan(wj, tau)), new[] { 1 }, false);
        var negatives = Engine.ReduceSum(
            PreprocessingTensorOps.OneMinus(Engine.TensorGreaterThan(wj, Engine.TensorNegate(tau))), new[] { 1 }, false);

        // estimated FDP = (1 + negatives) / max(1, positives)
        var fdp = Engine.TensorDivide(Engine.TensorAddScalar(negatives, NumOps.One), Engine.TensorMax(positives, NumOps.One));
        var controlled = PreprocessingTensorOps.OneMinus(Engine.TensorGreaterThan(fdp, NumOps.FromDouble(_fdrThreshold)))
            .Cast<int>().ToArray();
        var positive = Engine.TensorGreaterThan(w, NumOps.Zero).Cast<int>().ToArray();

        var selected = new List<int>();
        for (int k = 0; k < order.Length && _nFeaturesToSelect > 0; k++)
        {
            int candidate = order[k];
            if (positive[candidate] == 0)
                break;

            if (controlled[candidate] == 1)
            {
                // Select all features with W >= tau: in descending order these are the ones not below W_candidate
                var atLeast = PreprocessingTensorOps.OneMinus(Engine.TensorLessThan(
                    w, Engine.TensorBroadcastTo(FeatureSelectionTensorOps.Entries(w, new[] { candidate }), new[] { p })))
                    .Cast<int>().ToArray();
                selected.AddRange(order.Where(j => atLeast[j] == 1));
                break;
            }
        }

        // If FDR control doesn't select enough, fall back to top W scores
        if (selected.Count == 0)
        {
            _selectedIndices = order.Where(j => positive[j] == 1).Take(_nFeaturesToSelect).OrderBy(j => j).ToArray();
            if (_selectedIndices.Length == 0)
                _selectedIndices = order.Take(_nFeaturesToSelect).OrderBy(j => j).ToArray();
        }
        else
        {
            _selectedIndices = selected.Take(_nFeaturesToSelect).OrderBy(j => j).ToArray();
        }
        IsFitted = true;
    }

    // Each knockoff column is its feature permuted (rows ordered by one random key each, ties by row) plus uniform
    // noise of half-width std / 4, where std is the population standard deviation. Per column the n permutation keys
    // are drawn before the n noise draws, as the original generator did.
    private Tensor<T> GenerateKnockoffs(Tensor<T> x, Random random)
    {
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (p == 0)
            return x;

        var noise = new double[p * n];
        var columns = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            var keys = new int[n];
            for (int i = 0; i < n; i++)
                keys[i] = random.Next();
            for (int i = 0; i < n; i++)
                noise[j * n + i] = random.NextDouble();
            var shuffled = Enumerable.Range(0, n).OrderBy(i => keys[i]).ToArray();
            columns[j] = Engine.TensorGather(PreprocessingTensorOps.ColumnVector(x, j), new Tensor<int>(shuffled, new[] { n }), 0);
        }

        var permuted = Engine.TensorStack(columns, 1);
        var std = Engine.TensorSqrt(FilterStatisticTensorOps.ColumnPopulationVariance(x));
        var uniform = Engine.TensorTranspose(new Tensor<double>(noise, new[] { p, n }).Cast<T>());
        var offset = Engine.TensorMultiply(
            Engine.TensorMultiplyScalar(Engine.TensorSubtractScalar(uniform, NumOps.FromDouble(0.5)), NumOps.FromDouble(0.5)),
            FilterStatisticTensorOps.RowBroadcast(std, n));
        return Engine.TensorAdd(permuted, offset);
    }

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
            throw new InvalidOperationException("KnockoffFilter has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KnockoffFilter does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KnockoffFilter has not been fitted.");

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
