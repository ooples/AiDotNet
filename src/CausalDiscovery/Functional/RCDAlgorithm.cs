using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// RCD (Repetitive Causal Discovery) — LiNGAM extension for latent confounders.
/// </summary>
/// <remarks>
/// <para>
/// RCD extends DirectLiNGAM to handle latent (unobserved) confounders by iteratively
/// identifying "exogenous-like" variables whose residuals are mutually independent,
/// regressing them out, and repeating on the remaining variables.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Standardize the data</item>
/// <item>Initialize the set of remaining variables U = {0, 1, ..., d-1}</item>
/// <item>Repeat until U is empty:</item>
/// <item>  For each variable i in U, compute residuals after regressing out all discovered ancestors</item>
/// <item>  Compute pairwise independence (via mutual information) of residuals</item>
/// <item>  Identify the variable whose residuals are most independent of all others (exogenous)</item>
/// <item>  Add that variable to the causal ordering and record its causal coefficients</item>
/// <item>  Regress out the identified variable from all remaining variables</item>
/// <item>  If no sufficiently independent variable found, flag remaining as confounded and stop</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> RCD is like DirectLiNGAM but more cautious — it checks whether
/// variables are truly "root causes" or might be influenced by hidden (unobserved) factors.
/// When it finds variables that can't be cleanly separated, it marks them as potentially
/// confounded rather than forcing a possibly wrong causal direction.
/// </para>
/// <para>
/// Reference: Maeda and Shimizu (2020), "RCD: Repetitive Causal Discovery of
/// Linear Non-Gaussian Acyclic Models with Latent Confounders", AISTATS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("RCD: Repetitive Causal Discovery of Linear Non-Gaussian Acyclic Models with Latent Confounders", "https://proceedings.mlr.press/v108/maeda20a.html", Year = 2020, Authors = "Takashi Nicholas Maeda, Shohei Shimizu")]
public class RCDAlgorithm<T> : FunctionalBase<T>
{
    private readonly double _threshold;

    /// <summary>
    /// Cutoff on the SCALE-FREE confounding ratio that triggers the "remaining variables are
    /// confounded" stop. For the selected (most-exogenous) candidate the ratio is
    /// <c>Σⱼ min(0, DiffMI(best, j))² / Σⱼ DiffMI(best, j)²</c> — the fraction of the candidate's
    /// squared direction-evidence (DirectLiNGAM entropy criterion) that indicates it is actually an
    /// EFFECT of some other variable rather than a cause.
    /// <para>
    /// This replaces the previous absolute <c>Σ min(0, DiffMI)²</c> sum, whose magnitude (≈ 1e-8–1e-4
    /// on standardized residuals) never approached the 0.05 cutoff, so the stop was dead code. A clean
    /// root scores ≈ 0 (all direction-evidence points outward); a latently-confounded set has no clean
    /// root, so even the best candidate carries substantial wrong-way evidence (≈ 0.5 for a symmetric
    /// common-cause pair). Being a ratio in [0, 1] it needs no per-dataset rescaling, which is exactly
    /// the scale mismatch the raw sum suffered.
    /// </para>
    /// <para>
    /// Sourced from the dedicated <see cref="CausalDiscoveryOptions.ConfoundingEvidenceCutoff"/>
    /// (default <c>0.05</c> = tolerate up to 5% wrong-way evidence before declaring confounding).
    /// </para>
    /// <para>
    /// COVERAGE, stated accurately: what is exercised today is an integration test asserting that
    /// RCD returns a non-empty graph on linear data. The clean-versus-confounded separation this
    /// cutoff is calibrated for -- a clean LiNGAM DAG scoring well under it while a latent-confounder
    /// structure scores well over it, and the first-round stop firing when no clean root exists -- is
    /// NOT covered. <see cref="ConfoundingRatio"/> is exposed <c>internal</c> so those cases can be
    /// asserted directly without a full discovery run; until they are, treat the default as
    /// reasoned rather than measured.
    /// </para>
    /// </summary>
    private readonly double _confoundingScoreCutoff;

    /// <inheritdoc/>
    public override string Name => "RCD";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes RCD with optional configuration.
    /// </summary>
    public RCDAlgorithm(CausalDiscoveryOptions? options = null)
    {
        _threshold = options?.EdgeThreshold ?? 0.1;
        // Scale-free confounding ratio in [0, 1] (see _confoundingScoreCutoff doc): the fraction of
        // the best candidate's squared direction-evidence that points the wrong way. Default 0.05 =
        // tolerate up to 5% wrong-way evidence before declaring the remaining variables confounded.
        double confoundingCutoff = options?.ConfoundingEvidenceCutoff ?? 0.05;
        // The cutoff is compared against a ratio in [0, 1]; NaN would make the stop condition
        // (ratio > cutoff) always false (silently disabling confounding detection), and a value < 0
        // or > 1 would make it always/never trigger. Reject anything outside the valid range at the
        // options boundary rather than letting it distort discovery.
        if (double.IsNaN(confoundingCutoff) || confoundingCutoff < 0.0 || confoundingCutoff > 1.0)
            throw new ArgumentOutOfRangeException(
                nameof(options),
                $"{nameof(CausalDiscoveryOptions.ConfoundingEvidenceCutoff)} must be a finite value in " +
                $"[0, 1]; got {confoundingCutoff}.");
        _confoundingScoreCutoff = confoundingCutoff;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (d < 2)
            throw new ArgumentException($"RCD requires at least 2 variables, got {d}.");
        if (n < d + 3)
            throw new ArgumentException($"RCD requires at least {d + 3} samples for {d} variables, got {n}.");

        var standardized = StandardizeData(data);

        // Working copy of data — columns get regressed out as we identify exogenous variables
        var residualData = Tensor<T>.FromMatrix(standardized);

        var W = new Tensor<T>(new[] { d, d });
        var remaining = new List<int>(Enumerable.Range(0, d));
        T coefficientThreshold = NumOps.FromDouble(_threshold);

        // Iteratively identify exogenous variables
        int maxRounds = d;
        for (int round = 0; round < maxRounds && remaining.Count > 0; round++)
        {
            // Find the most exogenous variable among remaining
            int bestVar = -1;
            double bestScore = double.MaxValue;

            // Find the MOST EXOGENOUS variable among remaining using the DirectLiNGAM / RCD
            // criterion (Shimizu et al. 2011 DirectLiNGAM; Maeda & Shimizu 2020 RCD): the
            // exogenous variable is the one whose regression RESIDUALS of the other variables are
            // most INDEPENDENT of it. For each candidate, regress every other remaining variable
            // on the candidate and measure the mutual information between the candidate and each
            // residual r_j = x_j - reg(x_j ~ x_candidate). A true root leaves those residuals
            // independent of it (score ~0); a descendant does not.
            //
            // NOTE: scoring RAW pairwise MI between the candidate and the other variables (the
            // previous implementation) is wrong — a root cause has HIGH raw MI with its
            // descendants because it drives them, so raw-MI minimisation systematically selects a
            // LEAF and inverts the causal order, leaving the true directed edges undetected.
            // Score each candidate by the DirectLiNGAM entropy criterion: for every other
            // variable j, DiffMutualInfo(candidate, j) > 0 iff candidate → j (candidate is the
            // cause). Accumulate min(0, DiffMI)² — the squared EVIDENCE that the candidate is an
            // EFFECT of some j. The exogenous variable has no such evidence (score ≈ 0), so we
            // pick the argmin. This uses Hyvärinen's max-entropy differential-entropy
            // approximation, which detects the non-Gaussian dependence that identifies causal
            // direction even for near-collinear variables (a histogram MI cannot).
            // ONE evidence computation per candidate, reused by the stop test. Selection and
            // ConfoundingRatio walked the same `remaining` loop, called the same DiffMutualInfo pairs
            // and accumulated the same sum of min(0, diffMI)^2 -- so the selected candidate's
            // DiffMutualInfo calls ran twice per round, and the wrong-way-evidence formula lived in
            // two places that had to be edited together.
            double bestEffectEvidence = 0;
            double bestTotalEvidence = 0;

            foreach (int candidate in remaining)
            {
                var (effectEvidence, totalEvidence) = DirectionEvidence(residualData, candidate, remaining);

                if (effectEvidence < bestScore)
                {
                    bestScore = effectEvidence;
                    bestVar = candidate;
                    bestEffectEvidence = effectEvidence;
                    bestTotalEvidence = totalEvidence;
                }
            }

            // Check if the best variable is sufficiently independent using MI cutoff directly
            if (bestVar < 0)
            {
                break;
            }

            // Scale-free confounding ratio for the selected candidate: the fraction of its squared
            // direction-evidence that points the WRONG way (evidence it is an EFFECT of some other
            // variable, not a cause). ≈ 0 for a clean root (all evidence outward); elevated when no
            // clean root exists (latent confounding), e.g. ≈ 0.5 for a symmetric common-cause pair.
            // Normalizing by the total evidence removes the per-dataset scale drift that made the raw
            // Σ min(0, DiffMI)² sum (≈ 1e-4) never reach the cutoff — the dead-code bug this fixes.
            double confoundingRatio = bestTotalEvidence > 1e-12
                ? bestEffectEvidence / bestTotalEvidence
                : double.NaN;
            // Stop when the best candidate carries too much wrong-way evidence (latent confounding),
            // OR when its direction evidence is degenerate/indeterminate (ConfoundingRatio returns NaN
            // for ~zero or non-finite total evidence — perfectly collinear/deterministic residuals). In
            // BOTH cases the remaining variables cannot be cleanly ordered. Detect this in the FIRST
            // round too (no `ordering.Count > 0` guard): when every observed variable is already
            // confounded there is no clean root to seed the ordering, and picking an arbitrary bestVar
            // and writing directed coefficients would fabricate causal structure that RCD explicitly
            // must NOT assert.
            if (double.IsNaN(confoundingRatio) || confoundingRatio > _confoundingScoreCutoff)
            {
                // Remaining variables are likely confounded (or unidentifiable) — mark as confounded
                // and stop. Per RCD: confounded variables cannot be cleanly ordered, so we do NOT
                // assign edges among them. The zero entries in W for these variables indicate
                // "unidentified due to latent confounding", which is the correct RCD behavior. Callers
                // can check: if W[i,j]==0 AND W[j,i]==0 for remaining variables, those relationships
                // are confounded.
                break;
            }

            // Record causal coefficients from the identified exogenous variable to remaining ones
            remaining.Remove(bestVar);
            if (remaining.Count == 0)
                break;

            var coefficients = RegressCoefficients(residualData, bestVar);
            var targetMask = ColumnMask(d, remaining);
            var recorded = Engine.TensorMultiply(targetMask,
                Engine.TensorGreaterThan(Engine.TensorAbs(coefficients), coefficientThreshold));
            W = Engine.TensorSetSlice(W,
                Engine.TensorWhere(recorded, coefficients, Engine.TensorZerosLike(coefficients)).Reshape(new[] { 1, d }),
                new[] { bestVar, 0 });

            // Regress out the identified variable from all remaining variables
            var source = Engine.TensorSliceAxis(residualData, 1, bestVar).Reshape(new[] { n, 1 });
            var update = Engine.TensorMatMul(source, coefficients.Reshape(new[] { 1, d }));
            residualData = Engine.TensorWhere(Engine.TensorBroadcastTo(targetMask.Reshape(new[] { 1, d }), new[] { n, d }),
                Engine.TensorSubtract(residualData, update), residualData);
        }

        return ThresholdMatrix(W.ToMatrix(), _threshold);
    }

    /// <summary>
    /// A [d] tensor that is one at <paramref name="columns"/> and zero elsewhere.
    /// </summary>
    private static Tensor<T> ColumnMask(int d, IReadOnlyList<int> columns)
    {
        var engine = AiDotNetEngine.Current;
        var mask = new Tensor<T>(new[] { d });
        if (columns.Count == 0)
            return mask;

        var index = new Tensor<int>(columns.ToArray(), new[] { columns.Count });
        return engine.TensorScatter(mask, index, engine.TensorOnesLike(new Tensor<T>(new[] { columns.Count })), 0);
    }

    /// <summary>
    /// Hyvärinen's (1998) maximum-entropy approximation of the differential entropy of a
    /// STANDARDIZED (zero-mean, unit-variance) variable, for every column of <paramref name="u"/> [n x m].
    /// Lower entropy ⇒ more non-Gaussian.
    /// This is the entropy term DirectLiNGAM (Shimizu et al. 2011) — which RCD (Maeda &amp;
    /// Shimizu 2020) builds on — uses to score causal direction, because the non-Gaussianity of
    /// a residual is what identifies direction; a histogram MI cannot resolve this on
    /// near-collinear data.
    /// </summary>
    private static Tensor<T> DifferentialEntropy(Tensor<T> u)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        // Constants from Hyvärinen (1998) "New approximations of differential entropy…":
        // the two non-quadratic contrast functions G1(u)=log cosh(u), G2(u)=u·exp(-u²/2),
        // with gamma = E[G1(standard normal)].
        const double k1 = 79.047;
        const double k2 = 7.4129;
        const double gamma = 0.37457;
        double gaussEntropy = (1.0 + Math.Log(2.0 * Math.PI)) / 2.0;

        var m1 = engine.ReduceMean(engine.TensorLog(engine.TensorCosh(u)), new[] { 0 }, false);
        var m2 = engine.ReduceMean(engine.TensorMultiply(u,
            engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorSquare(u), numOps.FromDouble(-0.5)))),
            new[] { 0 }, false);

        var m1Excess = engine.TensorSquare(engine.TensorSubtractScalar(m1, numOps.FromDouble(gamma)));
        var penalty = engine.TensorAdd(engine.TensorMultiplyScalar(m1Excess, numOps.FromDouble(k1)),
            engine.TensorMultiplyScalar(engine.TensorSquare(m2), numOps.FromDouble(k2)));
        return engine.TensorSubtract(engine.TensorFullLike(penalty, numOps.FromDouble(gaussEntropy)), penalty);
    }

    /// <summary>
    /// Both direction-evidence accumulators for <paramref name="candidate"/> against the other
    /// variables in <paramref name="remaining"/>, in ONE pass.
    /// </summary>
    /// <returns>
    /// <c>EffectEvidence</c> is the sum of squared WRONG-WAY evidence, min(0, DiffMI)^2, which the
    /// discovery loop ranks candidates by. <c>TotalEvidence</c> is the sum of DiffMI^2, the
    /// denominator that makes <see cref="ConfoundingRatio"/> scale-free.
    /// </returns>
    /// <remarks>
    /// One definition of the criterion, called from both the candidate ranking and the confounding
    /// stop. The two used to accumulate the same quantity from the same DiffMutualInfo calls, so the
    /// selected candidate's calls ran twice per round and a change to the criterion had to be made
    /// in two places to stay correct.
    /// </remarks>
    private static (double EffectEvidence, double TotalEvidence) DirectionEvidence(
        Tensor<T> residualData, int candidate, IReadOnlyList<int> remaining)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var others = remaining.Where(other => other != candidate).ToArray();
        if (others.Length == 0)
            return (0, 0);

        var diffMI = DiffMutualInfo(residualData, candidate, others);
        var wrongWay = engine.TensorMin(diffMI, numOps.Zero);
        double effectEvidence = numOps.ToDouble(engine.TensorSum(engine.TensorSquare(wrongWay)));   // sum of min(0, DiffMI)^2
        double totalEvidence = numOps.ToDouble(engine.TensorSum(engine.TensorSquare(diffMI)));      // sum of DiffMI^2

        return (effectEvidence, totalEvidence);
    }

    /// <summary>
    /// Scale-free confounding ratio for <paramref name="candidate"/> against the other variables in
    /// <paramref name="remaining"/>: the fraction of the candidate's squared direction-evidence
    /// (DirectLiNGAM entropy criterion) indicating it is an EFFECT of some other variable rather than
    /// a cause. Returns a value in <c>[0, 1]</c>: near 0 when the candidate is a clean root (all
    /// evidence points outward), rising toward 0.5 for a symmetric common-cause (latently-confounded)
    /// pair where direction is unidentifiable. Degenerate direction evidence (perfectly
    /// collinear/deterministic residuals, so near-zero or non-finite total evidence) returns
    /// <see cref="double.NaN"/> (INDETERMINATE, not a clean root); the discovery loop treats NaN as a
    /// stop, not as an exogenous cause. Exposed <c>internal</c> so the RCD calibration tests can
    /// assert the clean-vs-confounded separation directly without depending on a full discovery run.
    /// </summary>
    internal static double ConfoundingRatio(Matrix<T> residualData, int candidate, IReadOnlyList<int> remaining)
    {
        var (effectEvidence, totalEvidence) = DirectionEvidence(Tensor<T>.FromMatrix(residualData), candidate, remaining);

        // Degenerate direction evidence (perfectly collinear/deterministic residuals, so near-zero or
        // non-finite total evidence): the direction is UNIDENTIFIABLE, which is NOT the same as a clean
        // root (ratio near 0). Return NaN so the caller stops without fabricating edges, rather than
        // treating a variable with no directional evidence as a clean exogenous cause.
        return totalEvidence > 1e-12 ? effectEvidence / totalEvidence : double.NaN;
    }

    /// <summary>
    /// DirectLiNGAM's entropy-based mutual-information difference for the ordered pairs (i, j), one per
    /// column j in <paramref name="others"/>:
    /// a positive value means <c>i → j</c> (variable i is the cause), a negative value means
    /// <c>j → i</c>. Both variables are standardized; the regression residuals in each direction
    /// are r_{i|j} = x_i − ρ·x_j and r_{j|i} = x_j − ρ·x_i (ρ = correlation), and the measure is
    /// [H(x_j) + H(r_{i|j})] − [H(x_i) + H(r_{j|i})] using
    /// <see cref="DifferentialEntropy(Tensor{T})"/>. A pair where either standard deviation is below
    /// 1e-12 scores 0.
    /// </summary>
    private static Tensor<T> DiffMutualInfo(Tensor<T> data, int i, int[] others)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = data.Shape[0];
        int m = others.Length;
        var tiny = numOps.FromDouble(1e-12);

        var di = engine.TensorSubtractScalar(engine.TensorSliceAxis(data, 1, i), engine.TensorMean(engine.TensorSliceAxis(data, 1, i)))
            .Reshape(new[] { n, 1 });
        var columns = engine.TensorIndexSelect(data, new Tensor<int>(others, new[] { m }), 1);
        var dj = engine.TensorSubtract(columns,
            engine.TensorBroadcastTo(engine.ReduceMean(columns, new[] { 0 }, true), new[] { n, m }));

        T stdI = numOps.Sqrt(engine.TensorMean(engine.TensorSquare(di)));
        var stdJ = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(dj), new[] { 0 }, true)); // [1, m]
        var cov = engine.ReduceMean(engine.TensorMultiply(engine.TensorBroadcastTo(di, new[] { n, m }), dj), new[] { 0 }, true);

        // A pair is scored only when neither standard deviation is below 1e-12 (NaN is scored).
        var degenerate = engine.TensorLessThan(stdJ, tiny);
        if (numOps.LessThan(stdI, tiny))
            degenerate = engine.TensorOnesLike(stdJ);
        var safeStdJ = engine.TensorWhere(degenerate, engine.TensorOnesLike(stdJ), stdJ);
        T safeStdI = numOps.LessThan(stdI, tiny) ? numOps.One : stdI;

        var rho = engine.TensorDivideScalar(engine.TensorDivide(cov, safeStdJ), safeStdI);
        rho = engine.TensorWhere(engine.TensorGreaterThan(rho, numOps.One), engine.TensorOnesLike(rho), rho);
        rho = engine.TensorWhere(engine.TensorLessThan(rho, numOps.Negate(numOps.One)),
            engine.TensorFullLike(rho, numOps.Negate(numOps.One)), rho);
        var oneMinusRhoSq = engine.TensorSubtract(engine.TensorOnesLike(rho), engine.TensorSquare(rho));
        var residStd = engine.TensorSqrt(engine.TensorWhere(engine.TensorLessThan(oneMinusRhoSq, tiny),
            engine.TensorFullLike(oneMinusRhoSq, tiny), oneMinusRhoSq));

        var shape = new[] { n, m };
        var xis = engine.TensorBroadcastTo(engine.TensorDivideScalar(di, safeStdI), shape);
        var xjs = engine.TensorDivide(dj, engine.TensorBroadcastTo(safeStdJ, shape));
        var rhoB = engine.TensorBroadcastTo(rho, shape);
        var residStdB = engine.TensorBroadcastTo(residStd, shape);
        var rIgivenJ = engine.TensorDivide(engine.TensorSubtract(xis, engine.TensorMultiply(rhoB, xjs)), residStdB);
        var rJgivenI = engine.TensorDivide(engine.TensorSubtract(xjs, engine.TensorMultiply(rhoB, xis)), residStdB);

        var hXi = DifferentialEntropy(engine.TensorDivideScalar(di, safeStdI));
        var diff = engine.TensorSubtract(
            engine.TensorAdd(DifferentialEntropy(xjs), DifferentialEntropy(rIgivenJ)),
            engine.TensorAdd(engine.TensorBroadcastTo(hXi, new[] { m }), DifferentialEntropy(rJgivenI)));
        return engine.TensorWhere(degenerate.Reshape(new[] { m }), engine.TensorZerosLike(diff), diff);
    }

    /// <summary>
    /// The OLS regression coefficient of <paramref name="source"/> predicting every column ([d]);
    /// 0 when the source's centered sum of squares is not above 1e-10.
    /// </summary>
    private Tensor<T> RegressCoefficients(Tensor<T> data, int source)
    {
        int n = data.Shape[0];
        var centered = CenterColumns(data);
        var ds = Engine.TensorSliceAxis(centered, 1, source).Reshape(new[] { n, 1 });
        T varS = Engine.TensorSum(Engine.TensorSquare(ds));
        var cov = Engine.TensorMatMul(Engine.TensorTranspose(ds), centered).Reshape(new[] { data.Shape[1] });
        return NumOps.GreaterThan(varS, NumOps.FromDouble(1e-10))
            ? Engine.TensorDivideScalar(cov, varS)
            : Engine.TensorZerosLike(cov);
    }
}
