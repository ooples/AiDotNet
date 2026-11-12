using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Copula-Based Synthesis generator that models marginal distributions independently
/// and couples them with a Gaussian copula to capture inter-feature dependencies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The generator operates in three phases:
/// 1. Fit marginal distributions for each feature using the empirical CDF (the sorted observed values)
/// 2. Transform data to uniform [0,1] via marginal CDFs, then to standard normal via inverse CDF
/// 3. Fit a Gaussian copula (correlation matrix) on the normal-transformed data
/// </para>
/// <para>
/// To generate:
/// 1. Sample from multivariate normal using the learned correlation matrix
/// 2. Transform back to uniform via standard normal CDF
/// 3. Transform to original scale via inverse marginal CDFs (quantile functions)
/// </para>
/// <para>
/// <b>For Beginners:</b> This method works like building synthetic data from two ingredients:
///
/// Ingredient 1 — Marginal shapes:
///   Each feature's histogram is learned independently (e.g., "Age is bell-shaped around 40").
///
/// Ingredient 2 — Correlation structure:
///   How features move together (e.g., "Age and Income go up together").
///
/// To generate: sample correlated random values, then map each one to its feature's histogram.
/// The result preserves both individual distributions and pairwise correlations.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("The Synthetic Data Vault",
    "https://doi.org/10.1109/DSAA.2016.49",
    Year = 2016,
    Authors = "Neha Patki, Roy Wedge, Kalyan Veeramachaneni")]
public class CopulaSynthGenerator<T> : SyntheticTabularGeneratorBase<T>
{
    private readonly CopulaSynthOptions<T> _options;

    // Marginal parameters per feature: sorted observed values, used as the
    // empirical quantile grid. For the probability-integral transform to be
    // rank-preserving (and therefore correlation-preserving), the forward CDF
    // and the inverse CDF MUST share one plotting-position grid — see
    // EmpiricalCdf / InverseEmpiricalCdf.
    private List<Tensor<T>> _sortedValues = new();

    // Per-feature column type, captured at fit time. Categorical and Discrete
    // columns must NOT be linearly interpolated on the inverse transform —
    // doing so leaks float noise into integer/categorical columns (the Copula
    // failure mode). Continuous columns interpolate normally.
    private ColumnDataType[] _columnTypes = Array.Empty<ColumnDataType>();

    // Gaussian copula: a factor F of the correlation matrix (F F^T = R; its Cholesky factor when R is positive
    // definite), used for sampling
    private Tensor<T>? _choleskyCorr;

    private int _numFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="CopulaSynthGenerator{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for the Copula Synthesis model.</param>
    public CopulaSynthGenerator(CopulaSynthOptions<T> options) : base(options.Seed)
    {
        _options = options;
    }

    /// <inheritdoc />
    protected override void FitInternal(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _numFeatures = data.Columns;
        int n = data.Rows;

        // Capture per-column type so the inverse transform can snap categorical
        // and discrete columns to valid values instead of interpolating between
        // them. `columns` was previously accepted and ignored — that omission is
        // exactly why float noise leaked into categorical columns.
        _columnTypes = new ColumnDataType[_numFeatures];
        for (int j = 0; j < _numFeatures; j++)
        {
            _columnTypes[j] = j < columns.Count ? columns[j].DataType : ColumnDataType.Continuous;
        }

        var engine = AiDotNetEngine.Current;
        var x = Tensor<T>.FromMatrix(data);
        T lowU = NumOps.FromDouble(1e-6);
        T highU = NumOps.FromDouble(1.0 - 1e-6);

        // Step 1: Store sorted values per feature for empirical CDF/quantile
        // Step 2: Transform to standard normal via empirical CDF → inverse normal CDF (clamped to avoid infinities)
        _sortedValues = new List<Tensor<T>>(_numFeatures);
        var normalColumns = new Tensor<T>[_numFeatures];
        for (int j = 0; j < _numFeatures; j++)
        {
            var column = engine.TensorSlice(x, [0, j], [n, 1]).Reshape([n]);
            var sorted = engine.TensorSort(column, 0, false).Values;
            _sortedValues.Add(sorted);
            var u = engine.TensorClamp(EmpiricalCdf(sorted, column), lowU, highU);
            normalColumns[j] = SyntheticTensorOps<T>.StandardNormalQuantile(u).Reshape([n, 1]);
        }

        if (_numFeatures == 0)
        {
            _choleskyCorr = new Tensor<T>([0, 0]);
            return;
        }

        // Step 3: Compute correlation matrix on normal-transformed data
        var normalData = _numFeatures == 1 ? normalColumns[0] : engine.TensorConcatenate(normalColumns, 1);
        var corrMatrix = ComputeCorrelation(normalData, n);

        // Step 4: factor the correlation matrix for sampling
        _choleskyCorr = SamplingFactor(corrMatrix);
    }

    /// <inheritdoc />
    protected override Matrix<T> GenerateInternal(int numSamples, Vector<T>? conditionColumn, Vector<T>? conditionValue)
    {
        if (_choleskyCorr is null)
            throw new InvalidOperationException("Generator is not fitted.");
        if (numSamples <= 0 || _numFeatures == 0)
            return new Matrix<T>(Math.Max(0, numSamples), _numFeatures);

        var engine = AiDotNetEngine.Current;

        // Sample standard normal vectors, row by row in the base sampler's order
        var draws = new T[numSamples * _numFeatures];
        for (int k = 0; k < draws.Length; k++)
            draws[k] = SampleStandardNormal();
        var z = new Tensor<T>(draws, [numSamples, _numFeatures]);

        // Multiply by the factor to get correlated normals: x_i = F z_i
        var correlated = engine.TensorMatMul(z, engine.TensorTranspose(_choleskyCorr));

        // Transform: standard normal → uniform → original scale via quantile function
        var u = engine.TensorClamp(SyntheticTensorOps<T>.StandardNormalCdf(correlated),
            NumOps.FromDouble(1e-6), NumOps.FromDouble(1.0 - 1e-6));
        var columns = new Tensor<T>[_numFeatures];
        for (int j = 0; j < _numFeatures; j++)
        {
            var uj = engine.TensorSlice(u, [0, j], [numSamples, 1]).Reshape([numSamples]);
            columns[j] = InverseEmpiricalCdf(j, uj).Reshape([numSamples, 1]);
        }

        var result = _numFeatures == 1 ? columns[0] : engine.TensorConcatenate(columns, 1);
        return result.ToMatrix();
    }

    /// <summary>
    /// Forward empirical CDF for the probability-integral transform. The i-th
    /// order statistic (0-based) is mapped to the Hazen plotting position
    /// <c>(i + 0.5) / n</c>. This is the inverse of <see cref="InverseEmpiricalCdf"/>
    /// on the SAME grid — the two MUST agree, or the round-trip
    /// value→u→normal→…→normal→u→value stops being rank-preserving and the fitted
    /// correlation bleeds off (the 0.82–0.93 degradation). Hazen positions sit
    /// strictly inside (0, 1), so no boundary clamping is needed and the tails
    /// are not squashed.
    /// </summary>
    internal static Tensor<T> EmpiricalCdf(Tensor<T> sorted, Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        int n = sorted.Length;
        if (n <= 1)
        {
            var middle = new Tensor<T>(values._shape);
            middle.Fill(NumOps.FromDouble(0.5));
            return middle;
        }

        // A value's tie block occupies the 0-based sorted indices lower .. upper - 1 (the two bisection bounds), so
        // its mid-rank is (lower + upper - 1) / 2 and its Hazen position (midRank + 0.5) / n = (lower + upper) / (2n).
        // Equal values share one plotting position and the transform stays monotone; a unique value at index i
        // maps to (i + 0.5) / n, which InverseEmpiricalCdf maps back to sorted[i].
        var lower = engine.TensorSearchSorted(sorted, values, right: false).Cast<T>(); // first index >= value
        var upper = engine.TensorSearchSorted(sorted, values, right: true).Cast<T>();  // first index > value
        return engine.TensorMultiplyScalar(engine.TensorAdd(lower, upper), NumOps.FromDouble(0.5 / n));
    }

    /// <summary>
    /// Inverse empirical CDF (quantile function) on the Hazen grid that
    /// <see cref="EmpiricalCdf"/> uses: plotting position <c>(i + 0.5)/n</c> maps
    /// back to <c>sorted[i]</c>. For Continuous columns we interpolate linearly
    /// between adjacent order statistics; for Categorical and Discrete columns we
    /// snap to the nearest order statistic (no interpolation) so the output is
    /// always a value that actually occurred — fixing the float-into-categorical
    /// leak (e.g. Education ∈ {0,1,2,3} can never come back as 1.7).
    /// </summary>
    private Tensor<T> InverseEmpiricalCdf(int featureIndex, Tensor<T> u)
    {
        var engine = AiDotNetEngine.Current;
        var sorted = _sortedValues[featureIndex];
        int n = sorted.Length;
        int m = u.Length;
        if (n == 0) return new Tensor<T>([m]);
        var first = engine.TensorTile(engine.TensorSlice(sorted, [0], [1]), [m]);
        if (n == 1) return first;
        var last = engine.TensorTile(engine.TensorSlice(sorted, [n - 1], [1]), [m]);

        bool snap = _columnTypes.Length > featureIndex
            && _columnTypes[featureIndex] != ColumnDataType.Continuous;
        T zero = NumOps.Zero;
        T one = NumOps.One;
        T lastIndex = NumOps.FromDouble(n - 1);

        // Position on the Hazen grid: u = (pos + 0.5)/n  ⇒  pos = u*n - 0.5.
        var pos = engine.TensorSubtractScalar(engine.TensorMultiplyScalar(u, NumOps.FromDouble(n)), NumOps.FromDouble(0.5));
        var inside = engine.TensorClamp(engine.TensorNanToNum(pos, 0.0), zero, lastIndex);
        var lo = engine.TensorMin(engine.TensorFloor(inside), NumOps.FromDouble(n - 2));
        var frac = engine.TensorSubtract(inside, lo);
        var below = engine.TensorGather(sorted, lo.Cast<int>(), 0);
        var above = engine.TensorGather(sorted, engine.TensorAddScalar(lo, one).Cast<int>(), 0);

        Tensor<T> interior;
        if (snap)
        {
            // Discrete / categorical: pick the nearer order statistic. Never blend.
            var nearerBelow = engine.TensorGreaterThan(engine.ScalarMinusTensor(NumOps.FromDouble(0.5), frac), zero);
            interior = engine.TensorWhere(nearerBelow, below, above);
        }
        else
        {
            interior = engine.TensorAdd(
                engine.TensorMultiply(below, engine.ScalarMinusTensor(one, frac)), engine.TensorMultiply(above, frac));
        }

        // pos <= 0 is the first order statistic and pos >= n - 1 the last.
        var afterFirst = engine.TensorGreaterThan(pos, zero);
        var beforeLast = engine.TensorGreaterThan(engine.ScalarMinusTensor(lastIndex, pos), zero);
        return engine.TensorWhere(afterFirst, engine.TensorWhere(beforeLast, interior, last), first);
    }

    /// <summary>
    /// Computes the correlation matrix from normal-transformed data.
    /// </summary>
    internal static Tensor<T> ComputeCorrelation(Tensor<T> data, int n)
    {
        var engine = AiDotNetEngine.Current;
        int d = data.Shape[1];

        // One row has no sample covariance (its n - 1 denominator is zero): the features are taken as uncorrelated.
        if (n <= 1)
        {
            return engine.TensorEye<T>(d);
        }

        // Sample covariance (n - 1 denominator) and standard deviations, floored at 1e-10.
        var mean = engine.ReduceMean(data, [0], keepDims: true);
        var centered = engine.TensorSubtract(data, engine.TensorTile(mean, [n, 1]));
        var cov = engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered), NumOps.FromDouble(n - 1));
        var identity = engine.TensorEye<T>(d);
        var stds = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(cov, identity), [1], keepDims: false));
        var floor = NumOps.FromDouble(1e-10);
        var floorTensor = new Tensor<T>([d]);
        floorTensor.Fill(floor);
        stds = engine.TensorWhere(engine.TensorGreaterThan(floorTensor, stds), floorTensor, stds);

        // r = cov / (s_i s_j), clamped to [-0.999, 0.999] (NaN stays NaN), with a unit diagonal.
        var r = engine.TensorDivide(cov, engine.TensorMatMul(stds.Reshape([d, 1]), stds.Reshape([1, d])));
        var lowR = new Tensor<T>([d, d]);
        lowR.Fill(NumOps.FromDouble(-0.999));
        var highR = new Tensor<T>([d, d]);
        highR.Fill(NumOps.FromDouble(0.999));
        r = engine.TensorWhere(engine.TensorGreaterThan(lowR, r), lowR, r);
        r = engine.TensorWhere(engine.TensorGreaterThan(r, highR), highR, r);
        return engine.TensorWhere(identity, identity, r);
    }

    /// <summary>
    /// A sampling factor F of the correlation matrix R (F F^T = R): its Cholesky factor L when R is positive
    /// definite, otherwise the eigendecomposition factor of R's positive part.
    /// </summary>
    private static Tensor<T> SamplingFactor(Tensor<T> correlation)
    {
        var (factor, info) = LinalgDispatch<T>.CholeskyEx(correlation);
        if (info.ToArray().All(code => code == 0))
        {
            return factor;
        }

        // Not positive definite (e.g. fewer rows than columns): F = V sqrt(max(Lambda, 0)) from the symmetric
        // eigendecomposition still satisfies F F^T = R on its positive part.
        var engine = AiDotNetEngine.Current;
        int d = correlation.Shape[0];
        var (values, vectors) = LinalgDispatch<T>.Eigh(correlation);
        var scale = engine.TensorSqrt(engine.TensorMax(values, NumOps.Zero));
        return engine.TensorMultiply(vectors, engine.TensorTile(scale.Reshape([1, d]), [d, 1]));
    }

}
