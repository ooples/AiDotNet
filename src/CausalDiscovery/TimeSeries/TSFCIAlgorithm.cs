using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// tsFCI — time series Fast Causal Inference.
/// </summary>
/// <remarks>
/// <para>
/// tsFCI adapts the FCI algorithm for time series data, allowing for the discovery of
/// causal relationships in the presence of latent (unmeasured) confounders. It uses
/// temporal ordering constraints (the future cannot cause the past) combined with
/// conditional independence testing on lagged variables to orient edges and identify
/// latent confounders.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Build a time-expanded graph with nodes X_t^(i) for each variable i at each lag</item>
/// <item>Start with complete graph, remove edges via conditional independence tests on lagged data</item>
/// <item>Apply temporal constraints: remove edges where cause lag > effect lag</item>
/// <item>Orient remaining edges using FCI rules adapted for temporal ordering</item>
/// <item>Mark edges with potential latent confounders as bidirected</item>
/// <item>Compute summary graph: collapse lagged edges to contemporaneous with max weight</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> tsFCI is like FCI but for time series. It can discover causal
/// relationships even when there are hidden variables affecting the observed ones,
/// using the fact that "the future cannot cause the past" to help figure out direction.
/// </para>
/// <para>
/// Reference: Entner and Hoyer (2010), "On Causal Discovery from Time Series Data
/// using FCI", PGM.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("On Causal Discovery from Time Series Data using FCI", "https://doi.org/10.1007/978-3-642-15114-9_12", Year = 2010, Authors = "Doris Entner, Patrik O. Hoyer")]
public class TSFCIAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "tsFCI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    private readonly double _correlationThreshold;

    public TSFCIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        _correlationThreshold = options?.CorrelationThreshold ?? 0.1;
        if (double.IsNaN(_correlationThreshold) || double.IsInfinity(_correlationThreshold) || _correlationThreshold < 0 || _correlationThreshold > 1)
            throw new ArgumentException("CorrelationThreshold must be a finite value between 0 and 1.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int effectiveN = n - MaxLag;
        if (effectiveN < MaxLag + 3 || d < 2) return new Matrix<T>(d, d);

        var cov = ComputeCovarianceMatrix(data);
        T threshold = NumOps.FromDouble(_correlationThreshold);
        var x = Tensor<T>.FromMatrix(data);
        var offDiagonal = OffDiagonalMask(d);

        // Lag-0 correlations C0[i, k] = corr(x_i(t), x_k(t)), the conditioning leg of every partial correlation
        var corr0 = LaggedStatistics(x, 0).Corr;

        // Phase 1: Build lagged covariance structure
        // For each pair (i,j) and lag l, compute lagged correlation
        // skeleton[i,j] = max over lags of |lagged_corr(i→j)|
        // The ARGMAX is kept alongside the max: the lag that wins for (i, j) is the lag at which the
        // Phase 2 independence test runs and at which the edge weight (lagged regression coefficient)
        // is read, so both are carried along with the running max instead of being recomputed.
        // A later lag only wins when it is strictly larger, so the earliest maximizing lag is kept
        // (lag 1 when every lagged correlation is 0).
        var first = LaggedStatistics(x, 1);
        var maxCorr = Engine.TensorZerosLike(first.Corr);
        var bestCoef = first.Coef;
        var bestSurvives = Survives(corr0, first.Corr, threshold);
        for (int lag = 1; lag <= MaxLag; lag++)
        {
            var stats = lag == 1 ? first : LaggedStatistics(x, lag);
            var absCorr = Engine.TensorAbs(stats.Corr);
            var update = Engine.TensorGreaterThan(absCorr, maxCorr);
            maxCorr = Engine.TensorWhere(update, absCorr, maxCorr);
            bestCoef = Engine.TensorWhere(update, stats.Coef, bestCoef);
            bestSurvives = Engine.TensorWhere(update, lag == 1 ? bestSurvives : Survives(corr0, stats.Corr, threshold), bestSurvives);
        }
        var skeleton = Engine.TensorMultiply(maxCorr, offDiagonal);

        // Phase 2: Conditional independence tests — keep edges above threshold whose correlation does not
        // disappear when conditioning on another variable (see Survives).
        var candidate = Engine.TensorMultiply(Engine.TensorMultiply(offDiagonal,
            Engine.TensorGreaterThan(skeleton, threshold)), bestSurvives);

        // Temporal orientation: the direction with the stronger lagged correlation wins.
        // When equal (deterministic data), use the asymmetric covariance ratio |cov_ij / var_i| vs
        // |cov_ij / var_j|, strict > with an i < j tie-break to avoid creating both directions.
        var skeletonT = Engine.TensorTranspose(skeleton);
        var stronger = Engine.TensorGreaterThan(skeleton, skeletonT);
        var weaker = Engine.TensorGreaterThan(skeletonT, skeleton);
        var ones = Engine.TensorOnesLike(skeleton);
        var tie = Engine.TensorMultiply(Engine.TensorSubtract(ones, stronger), Engine.TensorSubtract(ones, weaker));

        var covariance = Tensor<T>.FromMatrix(cov);
        var variance = Engine.TensorDiagonal(covariance);
        var shape = new[] { d, d };
        var varianceRows = Engine.TensorBroadcastTo(variance.Reshape(new[] { d, 1 }), shape);
        var varianceCols = Engine.TensorBroadcastTo(variance.Reshape(new[] { 1, d }), shape);
        T varianceFloor = NumOps.FromDouble(1e-10);
        var varianceOk = Engine.TensorMultiply(Engine.TensorGreaterThan(varianceRows, varianceFloor),
            Engine.TensorGreaterThan(varianceCols, varianceFloor));
        var ratioI = Engine.TensorAbs(Engine.TensorDivide(covariance, varianceRows));
        var ratioJ = Engine.TensorAbs(Engine.TensorDivide(covariance, varianceCols));
        var ratioWins = Engine.TensorAdd(Engine.TensorGreaterThan(ratioI, ratioJ),
            Engine.TensorMultiply(Engine.TensorEquals(ratioI, ratioJ), Engine.TensorTriu(ones, 1)));
        var iToJ = Engine.TensorAdd(stronger, Engine.TensorMultiply(tie, Engine.TensorMultiply(varianceOk, ratioWins)));

        // Weight the edge with the LAGGED regression coefficient at the lag that actually carried the
        // association. Every phase above reasons about lagged structure; the contemporaneous ratio
        // cov[i,j]/cov[i,i] measures a different relationship (on the fork X1 <- X0 -> X3 it ranked the
        // two children of X0 above the true X0->X1 edge). The lagged coefficient recovers the generating
        // SEM's own coefficients.
        var edge = Engine.TensorMultiply(candidate, iToJ);
        return Engine.TensorWhere(edge, bestCoef, Engine.TensorZerosLike(bestCoef)).ToMatrix();
    }

    /// <summary>
    /// Lagged correlations Corr[i, j] = corr(x_i(t), x_j(t + lag)) and lagged regression coefficients
    /// Coef[i, j] = cov(x_i(t), x_j(t + lag)) / var(x_i(t)) over t = 0..n-lag-1, for every ordered pair.
    /// The correlation floors each sum of squares at 1e-15; the coefficient is 0 when the source series
    /// has no variation (sum of squares below 1e-10) since it carries no information about the target,
    /// and both are 0 when fewer than 3 aligned samples exist. For a linear SEM the coefficient is the
    /// structural coefficient on that lagged edge, so edge weights stay on the data-generating scale.
    /// </summary>
    private (Tensor<T> Corr, Tensor<T> Coef) LaggedStatistics(Tensor<T> x, int lag)
    {
        int n = x.Shape[0];
        int d = x.Shape[1];
        int effectiveN = n - lag;
        if (effectiveN < 3)
            return (new Tensor<T>(new[] { d, d }), new Tensor<T>(new[] { d, d }));

        var source = CenterColumns(Engine.TensorSlice(x, new[] { 0, 0 }, new[] { effectiveN, d }));
        var target = CenterColumns(Engine.TensorSlice(x, new[] { lag, 0 }, new[] { effectiveN, d }));
        var cross = Engine.TensorMatMul(Engine.TensorTranspose(source), target);
        var sourceSs = Engine.ReduceSum(Engine.TensorMultiply(source, source), new[] { 0 }, false);
        var targetSs = Engine.ReduceSum(Engine.TensorMultiply(target, target), new[] { 0 }, false);

        T floor = NumOps.FromDouble(1e-15);
        var denom = Engine.TensorSqrt(Engine.TensorOuter(Engine.TensorMax(sourceSs, floor), Engine.TensorMax(targetSs, floor)));
        var corr = Engine.TensorDivide(cross, denom);

        var shape = new[] { d, d };
        var sourceSsRows = Engine.TensorBroadcastTo(sourceSs.Reshape(new[] { d, 1 }), shape);
        var noVariation = Engine.TensorLessThan(sourceSsRows, NumOps.FromDouble(1e-10));
        var coef = Engine.TensorWhere(noVariation, Engine.TensorZerosLike(cross), Engine.TensorDivide(cross, sourceSsRows));
        return (corr, coef);
    }

    /// <summary>
    /// For every ordered pair (i, j): 1 when the lagged association i(t) -> j(t + lag) survives conditioning,
    /// i.e. no conditioning variable k (k ≠ i, j) gives a CONCLUSIVE partial correlation
    /// (r_ij − r_ik r_kj) / sqrt((1 − r_ik²)(1 − r_kj²)) with |value| ≤ threshold. Here r_ij and r_kj are the
    /// lagged correlations at this lag and r_ik the lag-0 correlation, so the test is X_i(t) -> X_j(t+lag)
    /// conditioned on X_k(t) (all three aligned; correlating at the same positive lag would condition on
    /// X_k(t+lag) and let common-cause paths such as X1 &lt;- X0 -&gt; X3 survive).
    /// A partial correlation is INCONCLUSIVE when (1 − r_ik²)(1 − r_kj²) &lt; 1e-10 (k collinear with both
    /// endpoints, as on deterministic fixtures) or NaN; it is skipped rather than treated as "no
    /// association", because FCI's conditional-independence rule requires a VALID test, not a vacuous one
    /// (Spirtes et al. 2000 §3; Entner &amp; Hoyer 2010 inherit the same convention).
    /// </summary>
    private Tensor<T> Survives(Tensor<T> corr0, Tensor<T> corrLag, T threshold)
    {
        int d = corrLag.Shape[0];
        var cube = new[] { d, d, d }; // [i, k, j]
        var rij = Engine.TensorBroadcastTo(corrLag.Reshape(new[] { d, 1, d }), cube);
        var rik = Engine.TensorBroadcastTo(corr0.Reshape(new[] { d, d, 1 }), cube);
        var rkj = Engine.TensorBroadcastTo(corrLag.Reshape(new[] { 1, d, d }), cube);

        var numerator = Engine.TensorSubtract(rij, Engine.TensorMultiply(rik, rkj));
        var ones = Engine.TensorOnesLike(rij);
        var denomSq = Engine.TensorMultiply(
            Engine.TensorSubtract(ones, Engine.TensorMultiply(rik, rik)),
            Engine.TensorSubtract(ones, Engine.TensorMultiply(rkj, rkj)));
        var conclusive = Engine.TensorSubtract(ones, Engine.TensorLessThan(denomSq, NumOps.FromDouble(1e-10)));
        var absPartial = Engine.TensorAbs(Engine.TensorDivide(numerator, Engine.TensorSqrt(denomSq)));

        // |partial| <= threshold; both comparisons are false for NaN, so a NaN partial never removes an edge
        var independent = Engine.TensorAdd(Engine.TensorLessThan(absPartial, threshold), Engine.TensorEquals(absPartial, threshold));

        var identity = Engine.TensorEye<T>(d);
        var notI = Engine.TensorBroadcastTo(Engine.TensorSubtract(Engine.TensorOnesLike(identity), identity).Reshape(new[] { d, d, 1 }), cube);
        var notJ = Engine.TensorBroadcastTo(Engine.TensorSubtract(Engine.TensorOnesLike(identity), identity).Reshape(new[] { 1, d, d }), cube);
        var removes = Engine.TensorMultiply(Engine.TensorMultiply(independent, conclusive), Engine.TensorMultiply(notI, notJ));

        var removers = Engine.ReduceSum(removes, new[] { 1 }, false);
        return Engine.TensorSubtract(Engine.TensorOnesLike(removers), Engine.TensorGreaterThan(removers, NumOps.Zero));
    }
}
