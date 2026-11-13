using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
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

        // Step 1: second-order Gaussian knockoffs with the equicorrelated construction (Barber and Candes 2015;
        // Candes, Fan, Janson and Lv 2018)
        var x = FilterStatisticTensorOps.AsTensor(data);
        var knockoffs = GaussianKnockoffs(x, random.Next());

        // Step 2: absolute correlation with the target for original and knockoff features
        var y = FilterStatisticTensorOps.AsTensor(target);
        var featureStatistics = FeatureSelectionTensorOps.AbsoluteCorrelation(x, y);
        var knockoffStatistics = FeatureSelectionTensorOps.AbsoluteCorrelation(knockoffs, y);
        _featureStatistics = featureStatistics.ToVector();
        _knockoffStatistics = knockoffStatistics.ToVector();

        // Step 3: W = |Z_j| - |Z_knockoff_j|
        var w = Engine.TensorSubtract(featureStatistics, knockoffStatistics);

        // Step 4: the knockoff+ selection {j : W_j >= T}, at most the requested number
        var selected = KnockoffPlusSelection(w, _fdrThreshold);

        // If FDR control selects nothing, fall back to the top W scores
        if (selected.Length == 0)
        {
            var order = PreprocessingTensorOps.StableOrder(w, descending: true);
            var positive = Engine.TensorGreaterThan(w, NumOps.Zero).Cast<int>().ToArray();
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

    /// <summary>
    /// The knockoff+ selection of Barber and Candes (2015): the threshold
    /// <c>T = min { t &gt; 0 : (1 + #{j : W_j &lt;= -t}) / max(1, #{j : W_j &gt;= t}) &lt;= q }</c> over the candidate
    /// values <c>t</c> in <c>{|W_j|} \ {0}</c>, and the features with <c>W_j &gt;= T</c> in descending W order (ties to
    /// the lower index); none when no candidate passes. The estimated FDP is not monotone in t, so the smallest passing
    /// t is searched among all of them.
    /// </summary>
    internal static int[] KnockoffPlusSelection(Tensor<T> w, double q)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = w.Length;
        if (p == 0)
            return Array.Empty<int>();

        // for every candidate t_k = |W_k| (rows) and every feature j (columns): W_j >= t_k and W_j <= -t_k, counted in int
        var magnitude = engine.TensorAbs(w);
        var tau = FilterStatisticTensorOps.ColumnBroadcast(magnitude, p);
        var wj = FilterStatisticTensorOps.RowBroadcast(w, p);
        var positives = engine.ReduceSum(
            PreprocessingTensorOps.OneMinus(engine.TensorLessThan(wj, tau)).Cast<int>(), new[] { 1 }, false).Cast<double>();
        var negatives = engine.ReduceSum(PreprocessingTensorOps.OneMinus(
            engine.TensorGreaterThan(wj, engine.TensorNegate(tau))).Cast<int>(), new[] { 1 }, false).Cast<double>();

        // estimated FDP = (1 + negatives) / max(1, positives); t = 0 is no candidate
        var fdp = engine.TensorDivide(engine.TensorAddScalar(negatives, 1.0), engine.TensorMax(positives, 1.0));
        var passing = engine.TensorMultiply(
            PreprocessingTensorOps.OneMinus(engine.TensorGreaterThan(fdp, q)),
            engine.TensorGreaterThan(magnitude, ops.Zero).Cast<double>());
        var candidates = FilterStatisticTensorOps.MaskPositions(passing);
        if (candidates.Length == 0)
            return Array.Empty<int>();

        // T = the smallest passing |W_k|; select W_j >= T
        var passingMagnitudes = FeatureSelectionTensorOps.Entries(magnitude, candidates);
        var threshold = FeatureSelectionTensorOps.Entries(
            passingMagnitudes, new[] { PreprocessingTensorOps.StableOrder(passingMagnitudes, descending: false)[0] });
        var atLeast = PreprocessingTensorOps.OneMinus(
            engine.TensorLessThan(w, engine.TensorBroadcastTo(threshold, new[] { p }))).Cast<int>().ToArray();
        return PreprocessingTensorOps.StableOrder(w, descending: true).Where(j => atLeast[j] == 1).ToArray();
    }

    /// <summary>
    /// Second-order Gaussian model-X knockoffs with the equicorrelated construction (Candes, Fan, Janson and Lv 2018,
    /// section 3.4; Barber and Candes 2015). On the standardised features Z with correlation matrix
    /// <c>S = V diag(lambda) V'</c> and <c>s = min(2 lambda_min, 1)</c>, the knockoffs are
    /// <c>Z (I - s S^-1) + G C</c> with G standard normal and <c>C'C = 2 s I - s^2 S^-1</c>, i.e.
    /// <c>Z V diag(1 - s / lambda) V' + G diag(sqrt(2 s - s^2 / lambda)) V'</c>, mapped back to the feature scale.
    /// They have the correlation matrix S, and correlation <c>S_jk</c> with every original feature k other than their
    /// own (<c>1 - s</c> with that), which is what makes W symmetric under the null and the knockoff+ threshold control
    /// the FDR. A constant feature is its own knockoff. The normal draws use the engine's seeded generator.
    /// </summary>
    internal static Tensor<T> GaussianKnockoffs(Tensor<T> x, int seed)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (n == 0 || p == 0)
            return x;

        // standardised features; a constant one stays 0 and takes a unit diagonal so S stays a correlation matrix
        var mean = FilterStatisticTensorOps.ColumnMeans(x);
        var std = engine.TensorSqrt(FilterStatisticTensorOps.ColumnPopulationVariance(x));
        var varying = engine.TensorGreaterThan(std, ops.Zero);
        var safeStd = engine.TensorWhere(varying, std, FilterStatisticTensorOps.FilledLike(std, ops.One));
        var z = engine.TensorMultiply(
            engine.TensorDivide(engine.TensorSubtract(x, FilterStatisticTensorOps.RowBroadcast(mean, n)),
                FilterStatisticTensorOps.RowBroadcast(safeStd, n)),
            FilterStatisticTensorOps.RowBroadcast(varying, n));
        var correlation = engine.TensorAdd(
            engine.TensorDivideScalar(engine.TensorMatMul(engine.TensorTranspose(z), z), ops.FromDouble(n)),
            engine.TensorDiag(PreprocessingTensorOps.OneMinus(varying)));

        // S = V diag(lambda) V'; s = min(2 lambda_min, 1), at least 0
        var (lambda, vectors) = LinalgDispatch<T>.Eigh(correlation);
        T twiceSmallest = ops.Multiply(ops.FromDouble(2), engine.TensorMinValue(lambda));
        T s = ops.LessThan(twiceSmallest, ops.One) ? twiceSmallest : ops.One;
        if (ops.LessThan(s, ops.Zero))
            s = ops.Zero;

        // s / lambda (0 where lambda is not positive; s is 0 there too)
        var ratio = FilterStatisticTensorOps.GuardedDivide(
            FilterStatisticTensorOps.FilledLike(lambda, s), lambda, 0.0, ops.Zero);
        var keep = PreprocessingTensorOps.OneMinus(ratio);
        var spread = engine.TensorSqrt(engine.TensorMax(
            engine.TensorSubtract(FilterStatisticTensorOps.FilledLike(lambda, ops.Multiply(ops.FromDouble(2), s)),
                engine.TensorMultiplyScalar(ratio, s)), ops.Zero));
        var vectorsT = engine.TensorTranspose(vectors);
        var shrink = engine.TensorMatMul(engine.TensorMultiply(vectors, FilterStatisticTensorOps.RowBroadcast(keep, p)), vectorsT);
        var noiseMap = engine.TensorMultiply(vectorsT, FilterStatisticTensorOps.ColumnBroadcast(spread, p));
        var noise = engine.TensorRandnLike(z, seed);
        var knockoffZ = engine.TensorAdd(engine.TensorMatMul(z, shrink), engine.TensorMatMul(noise, noiseMap));

        // back to the feature scale (a constant feature has std 0 and returns to its constant)
        return engine.TensorAdd(
            engine.TensorMultiply(knockoffZ, FilterStatisticTensorOps.RowBroadcast(std, n)),
            FilterStatisticTensorOps.RowBroadcast(mean, n));
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
