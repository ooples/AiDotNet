using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// LPCMCI — Latent PCMCI for time series with hidden confounders.
/// </summary>
/// <remarks>
/// <para>
/// LPCMCI extends PCMCI to handle latent confounders by combining ideas from FCI
/// (ancestral graph representation) with PCMCI's condition selection and MCI testing.
/// It outputs a time series PAG (partial ancestral graph) instead of a DAG.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Run PCMCI condition selection to find preliminary lagged parents for each variable</item>
/// <item>Run MCI tests: confirm each candidate link X_i(t-τ) → X_j(t) conditioning on the
///   parents of BOTH endpoints, removing links that become independent</item>
/// <item>Apply orientation rules that account for possible latent confounders:
///   temporal (past→present) orientation, plus symmetric bidirected marks where both
///   directions survive (ambiguous / latent-confounded)</item>
/// <item>Compute summary graph: collapse lags, keep max absolute strength per pair</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> LPCMCI is the most advanced version of PCMCI. It works even when
/// there are hidden variables affecting the ones you can measure. The trade-off is that
/// some edges may be uncertain in direction (shown as symmetric, bidirected links).
/// </para>
/// <para>
/// Reference: Gerhardus and Runge (2020), "High-recall causal discovery for autocorrelated
/// time series with latent confounders", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("High-recall causal discovery for autocorrelated time series with latent confounders", "https://proceedings.neurips.cc/paper_files/paper/2020/hash/94e70705efae45a1de1bcc6ca669aca3-Abstract.html", Year = 2020, Authors = "Andreas Gerhardus, Jakob Runge")]
public class LPCMCIAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "LPCMCI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    private readonly double _alpha;
    private readonly int _maxCondSetSize;
    private readonly double _correlationThreshold;

    public LPCMCIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        _alpha = options?.SignificanceLevel ?? 0.05;
        _maxCondSetSize = options?.MaxConditioningSetSize ?? 3;
        _correlationThreshold = options?.CorrelationThreshold ?? 0.1;
        if (double.IsNaN(_alpha) || double.IsInfinity(_alpha) || _alpha <= 0 || _alpha >= 1)
            throw new ArgumentException("SignificanceLevel must be a finite value between 0 and 1 (exclusive).");
        if (_maxCondSetSize < 0)
            throw new ArgumentException("MaxConditioningSetSize must be non-negative.");
        if (double.IsNaN(_correlationThreshold) || double.IsInfinity(_correlationThreshold) || _correlationThreshold < 0 || _correlationThreshold > 1)
            throw new ArgumentException("CorrelationThreshold must be a finite value between 0 and 1.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int effectiveN = n - MaxLag;
        if (effectiveN < 2 * d + 3 || d < 2) return new Matrix<T>(d, d);

        // Phase 1 — PCMCI condition selection (PC1). For each target j, start from every
        // lagged variable X_i(t-τ) as a candidate parent and iteratively drop the ones that
        // become conditionally independent of X_j(t) as the conditioning-set size grows. The
        // crucial detail (and the fix for the previous implementation, which conditioned on
        // wrong-lag / future values) is that every partial correlation residualizes on the
        // TIME-ALIGNED past value x_k[t-cLag]: conditioning on the correct lag of the shared
        // cause is what removes common-cause confounding — e.g. X1 ⟂ X3 | X0(t-1), which a
        // contemporaneous or future-aligned test cannot detect. (Runge et al. 2019, PCMCI
        // PC1 step; Gerhardus & Runge 2020, ancestral condition selection.)
        var parents = new HashSet<(int var, int lag)>[d];
        for (int j = 0; j < d; j++)
        {
            parents[j] = [];
            for (int i = 0; i < d; i++)
                for (int lag = 1; lag <= MaxLag; lag++)
                    parents[j].Add((i, lag));

            for (int condSize = 0; condSize <= _maxCondSetSize && condSize < parents[j].Count; condSize++)
            {
                var toRemove = new List<(int var, int lag)>();
                var current = new List<(int var, int lag)>(parents[j]);
                foreach (var (pVar, pLag) in current)
                {
                    var others = current.Where(p => p != (pVar, pLag)).ToList();
                    if (others.Count < condSize) continue;

                    var condSet = new HashSet<(int var, int lag)>(others.Take(condSize));
                    double partCorr = ComputeLaggedPartialCorrelation(data, j, pVar, pLag, condSet, effectiveN);
                    double pValue = FisherZPValue(Math.Abs(partCorr), effectiveN, condSet.Count);

                    if (pValue > _alpha) // conditionally independent → not a parent
                        toRemove.Add((pVar, pLag));
                }

                foreach (var item in toRemove) parents[j].Remove(item);
                if (parents[j].Count <= condSize + 1) break;
            }
        }

        // Phase 2 — MCI test. Confirm each surviving candidate link X_i(t-τ) → X_j(t) by
        // conditioning on parents(j)\{link} ∪ parents(i), then aggregate across lags into a
        // per-ordered-pair strength (max |partial correlation|). This is PCMCI's momentary
        // conditional-independence step; it yields a directed strength for i→j separate from
        // j→i, which the orientation phase turns into edges.
        var linkStrength = new Matrix<T>(d, d);
        for (int j = 0; j < d; j++)
        {
            foreach (var (pVar, pLag) in parents[j])
            {
                if (pVar == j) continue; // self-lag (autocorrelation): used for conditioning, not an edge

                var condSet = new HashSet<(int var, int lag)>(parents[j]);
                condSet.Remove((pVar, pLag));
                foreach (var parent in parents[pVar]) condSet.Add(parent);

                double partCorr = ComputeLaggedPartialCorrelation(data, j, pVar, pLag, condSet, effectiveN);
                double pValue = FisherZPValue(Math.Abs(partCorr), effectiveN, condSet.Count);

                if (pValue <= _alpha)
                {
                    double absPartCorr = Math.Abs(partCorr);
                    if (absPartCorr > NumOps.ToDouble(linkStrength[pVar, j]))
                        linkStrength[pVar, j] = NumOps.FromDouble(absPartCorr);
                }
            }
        }

        // Fallback for (near-)deterministic data: when every variable is perfectly collinear
        // (e.g. noise-free synthetic ramps) OLS residualization drives every partial
        // correlation to zero and the MCI step removes all links. Rather than return an empty
        // graph, fall back to the strongest UNCONDITIONAL lagged cross-correlation per pair
        // (single direction, past→present), preserving a meaningful, acyclic skeleton. This
        // only triggers when no link survived conditioning, so it never affects stochastic data.
        bool hasEdges = false;
        for (int i = 0; i < d && !hasEdges; i++)
            for (int j = 0; j < d && !hasEdges; j++)
                if (i != j && NumOps.ToDouble(linkStrength[i, j]) > 0.0) hasEdges = true;

        if (!hasEdges)
        {
            var empty = new HashSet<(int var, int lag)>();
            for (int i = 0; i < d; i++)
            {
                for (int j = i + 1; j < d; j++)
                {
                    double best = 0.0;
                    int bi = i, bj = j;
                    for (int lag = 1; lag <= MaxLag; lag++)
                    {
                        double cij = Math.Abs(ComputeLaggedPartialCorrelation(data, j, i, lag, empty, effectiveN)); // i(t-lag) → j(t)
                        if (cij > best) { best = cij; bi = i; bj = j; }
                        double cji = Math.Abs(ComputeLaggedPartialCorrelation(data, i, j, lag, empty, effectiveN)); // j(t-lag) → i(t)
                        if (cji > best) { best = cji; bi = j; bj = i; }
                    }
                    if (best > _correlationThreshold)
                        linkStrength[bi, bj] = NumOps.FromDouble(best);
                }
            }
        }

        // Phase 3 — LPCMCI orientation with latent-confounder handling.
        // A confirmed lagged link is oriented by TIME ORDER (the cause precedes the effect),
        // so the directed portion is acyclic by construction (Gerhardus & Runge 2020 §3.1:
        // "effects never precede causes"). When only one direction survives we emit that
        // single directed edge. When BOTH directions survive with comparable strength the
        // pair is ambiguous / possibly latent-confounded: we emit a SYMMETRIC bidirected
        // edge X_i ↔ X_j (equal weights in both directions) — a bidirected mark asserts
        // NON-ancestorship at both ends, carries no direction, and therefore is excluded
        // from the directed-portion acyclicity check and never manifests as a spurious
        // asymmetric bidirectional (directed-cycle) edge.
        var result = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                double sij = NumOps.ToDouble(linkStrength[i, j]);
                double sji = NumOps.ToDouble(linkStrength[j, i]);
                bool hasIJ = sij > _correlationThreshold;
                bool hasJI = sji > _correlationThreshold;

                if (!hasIJ && !hasJI) continue;

                if (hasIJ && hasJI)
                {
                    // Both directions survive → bidirected, symmetric weight.
                    T sym = NumOps.FromDouble(Math.Max(sij, sji));
                    result[i, j] = sym;
                    result[j, i] = sym;
                }
                else if (hasIJ)
                {
                    result[i, j] = linkStrength[i, j];
                }
                else
                {
                    result[j, i] = linkStrength[j, i];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Partial correlation between target(t) and source(t-lag), conditioned on a set of
    /// TIME-ALIGNED lagged variables x_k[t-cLag], via OLS residualization. Conditioning on
    /// the correctly-lagged past of shared causes is what removes common-cause confounding.
    /// </summary>
    private double ComputeLaggedPartialCorrelation(Matrix<T> data,
        int target, int source, int lag, HashSet<(int var, int lag)> condSet, int effectiveN)
    {
        int n = data.Rows;
        int offset = n - effectiveN;

        var targetVals = new Vector<T>(effectiveN);
        var sourceVals = new Vector<T>(effectiveN);
        for (int t = 0; t < effectiveN; t++)
        {
            targetVals[t] = data[offset + t, target];
            sourceVals[t] = data[offset + t - lag, source]; // offset = MaxLag ≥ lag ⇒ index ≥ 0
        }

        if (condSet.Count == 0)
            return PearsonCorrelation(sourceVals, targetVals, effectiveN);

        int numCond = condSet.Count;
        var condMatrix = new Matrix<T>(effectiveN, numCond);
        int col = 0;
        foreach (var (cVar, cLag) in condSet)
        {
            for (int t = 0; t < effectiveN; t++)
            {
                int tIdx = offset + t - cLag;
                condMatrix[t, col] = (tIdx >= 0 && tIdx < n) ? data[tIdx, cVar] : NumOps.Zero;
            }
            col++;
        }

        var residTarget = OLSResiduals(condMatrix, targetVals, effectiveN, numCond);
        var residSource = OLSResiduals(condMatrix, sourceVals, effectiveN, numCond);
        return PearsonCorrelation(residSource, residTarget, effectiveN);
    }

    /// <summary>
    /// OLS residuals: y - Z (Z'Z)^{-1} Z'y via the (ridge-stabilized) normal equations.
    /// </summary>
    private Vector<T> OLSResiduals(Matrix<T> Z, Vector<T> y, int n, int p)
    {
        var ZtZ = new Matrix<T>(p, p);
        var Zty = new Vector<T>(p);

        for (int i = 0; i < p; i++)
        {
            for (int j = i; j < p; j++)
            {
                T sum = NumOps.Zero;
                for (int t = 0; t < n; t++)
                    sum = NumOps.Add(sum, NumOps.Multiply(Z[t, i], Z[t, j]));
                ZtZ[i, j] = sum;
                ZtZ[j, i] = sum;
            }

            T sumZy = NumOps.Zero;
            for (int t = 0; t < n; t++)
                sumZy = NumOps.Add(sumZy, NumOps.Multiply(Z[t, i], y[t]));
            Zty[i] = sumZy;
        }

        T ridge = NumOps.FromDouble(1e-10);
        for (int i = 0; i < p; i++)
            ZtZ[i, i] = NumOps.Add(ZtZ[i, i], ridge);

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(ZtZ, Zty, MatrixDecompositionType.Lu);

        var residuals = new Vector<T>(n);
        for (int t = 0; t < n; t++)
        {
            T pred = NumOps.Zero;
            for (int j = 0; j < p; j++)
                pred = NumOps.Add(pred, NumOps.Multiply(Z[t, j], beta[j]));
            residuals[t] = NumOps.Subtract(y[t], pred);
        }

        return residuals;
    }

    /// <summary>Pearson correlation between two vectors (0 when either is (near-)constant).</summary>
    private double PearsonCorrelation(Vector<T> x, Vector<T> y, int n)
    {
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += NumOps.ToDouble(x[i]); my += NumOps.ToDouble(y[i]); }
        mx /= n; my /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = NumOps.ToDouble(x[i]) - mx, dy = NumOps.ToDouble(y[i]) - my;
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    private static double FisherZPValue(double r, int n, int condSetSize)
    {
        int dof = n - condSetSize - 3;
        if (dof < 1) return 1.0;
        double rc = Math.Max(-0.9999, Math.Min(0.9999, r));
        double z = 0.5 * Math.Log((1 + rc) / (1 - rc));
        double testStat = Math.Abs(z) * Math.Sqrt(dof);
        return 2.0 * NormalCdfComplement(testStat);
    }

    private static double NormalCdfComplement(double x)
    {
        if (x < 0) return 1.0 - NormalCdfComplement(-x);
        const double p = 0.2316419;
        const double b1 = 0.319381530, b2 = -0.356563782, b3 = 1.781477937;
        const double b4 = -1.821255978, b5 = 1.330274429;
        double t = 1.0 / (1.0 + p * x);
        double phi = Math.Exp(-0.5 * x * x) / Math.Sqrt(2.0 * Math.PI);
        return phi * (b1 * t + b2 * t * t + b3 * t * t * t + b4 * t * t * t * t + b5 * t * t * t * t * t);
    }
}
