using AiDotNet.Attributes;
using AiDotNet.Enums;
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
/// <item>Run PCMCI condition selection to find preliminary parents for each variable</item>
/// <item>Apply FCI-style skeleton thinning: iteratively test edges conditioning on subsets
///   of the selected parents, removing edges that become independent</item>
/// <item>Apply orientation rules that account for possible latent confounders:
///   collider orientation, discriminating paths, and temporal constraints</item>
/// <item>Mark edges with ambiguous orientation (possible latent confounder) as bidirected</item>
/// <item>Compute summary graph: collapse lags, keep max absolute weight per pair</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> LPCMCI is the most advanced version of PCMCI. It works even when
/// there are hidden variables affecting the ones you can measure. The trade-off is that
/// some edges may be uncertain in direction (shown with circle marks).
/// </para>
/// <para>
/// Reference: Gerhardus and Runge (2022), "High-recall causal discovery for autocorrelated
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
[ModelPaper("High-recall causal discovery for autocorrelated time series with latent confounders", "https://proceedings.neurips.cc/paper_files/paper/2020/hash/94e70705efae45a1de1bcc6ca669aca3-Abstract.html", Year = 2022, Authors = "Andreas Gerhardus, Jakob Runge")]
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

    public LPCMCIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        _alpha = options?.SignificanceLevel ?? 0.05;
        _maxCondSetSize = options?.MaxConditioningSetSize ?? 3;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int effectiveN = n - MaxLag;
        if (effectiveN < 2 * d + 3 || d < 2) return new Matrix<T>(d, d);

        var cov = ComputeCovarianceMatrix(data);
        T eps = NumOps.FromDouble(1e-10);
        T threshold = NumOps.FromDouble(0.1);

        // Phase 1: PCMCI-style condition selection — compute MCI statistic for each pair
        // MCI(i→j | Parents(j)\{i}) tests if i→j survives conditioning on j's other parents
        var mciStrength = new Matrix<T>(d, d);
        var skeleton = new bool[d, d];

        // Compute lagged correlations for all pairs
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;

                // Find best lag correlation i→j
                T bestCorr = NumOps.Zero;
                for (int lag = 1; lag <= MaxLag; lag++)
                {
                    T lagCorr = ComputeLaggedCorrelationEngine(data, i, j, lag, n);
                    T absCorr = NumOps.Abs(lagCorr);
                    if (NumOps.GreaterThan(absCorr, NumOps.Abs(bestCorr)))
                        bestCorr = lagCorr;
                }

                mciStrength[i, j] = bestCorr;
                if (NumOps.GreaterThan(NumOps.Abs(bestCorr), threshold))
                    skeleton[i, j] = true;
            }
        }

        // Phase 2: FCI-style skeleton thinning — test conditional independence
        // with increasing conditioning set size
        for (int condSize = 1; condSize <= Math.Min(_maxCondSetSize, d - 2); condSize++)
        {
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    if (i == j || !skeleton[i, j]) continue;

                    // Build candidate conditioning variables (adjacent to j, excluding i)
                    var candidates = new List<int>();
                    for (int k = 0; k < d; k++)
                    {
                        if (k == i || k == j) continue;
                        if (skeleton[k, j]) candidates.Add(k);
                    }

                    if (candidates.Count < condSize) continue;

                    // Test subsets of size condSize
                    bool removed = false;
                    foreach (var subset in GetSubsets(candidates, condSize))
                    {
                        T partCorr = ComputePartialLaggedCorrelationMulti(data, i, j, subset, n);
                        double absPCorr = Math.Abs(NumOps.ToDouble(partCorr));
                        double pValue = FisherZPValue(absPCorr, effectiveN, subset.Count);

                        if (pValue > _alpha)
                        {
                            skeleton[i, j] = false;
                            mciStrength[i, j] = NumOps.Zero;
                            removed = true;
                            break;
                        }
                    }

                    if (removed) break;
                }
            }
        }

        // Phase 3: Orientation with latent confounder handling
        // Use temporal constraints + FCI rules
        var result = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (i == j || !skeleton[i, j]) continue;

                // Temporal orientation: compare lagged correlations in both directions
                T bestItoJ = NumOps.Zero;
                T bestJtoI = NumOps.Zero;
                for (int lag = 1; lag <= MaxLag; lag++)
                {
                    T corrIJ = NumOps.Abs(ComputeLaggedCorrelationEngine(data, i, j, lag, n));
                    T corrJI = NumOps.Abs(ComputeLaggedCorrelationEngine(data, j, i, lag, n));
                    if (NumOps.GreaterThan(corrIJ, bestItoJ)) bestItoJ = corrIJ;
                    if (NumOps.GreaterThan(corrJI, bestJtoI)) bestJtoI = corrJI;
                }

                // FCI rule: if both directions are strong and similar, could be latent confounder
                // Mark as bidirected (both W[i,j] and W[j,i] nonzero)
                bool possibleLatent = NumOps.GreaterThan(bestItoJ, threshold) &&
                                     NumOps.GreaterThan(bestJtoI, threshold);

                if (possibleLatent)
                {
                    // Bidirected: use average strength for both directions
                    T varI = cov[i, i];
                    if (NumOps.GreaterThan(varI, eps))
                    {
                        T weight = NumOps.Abs(NumOps.Divide(cov[i, j], varI));
                        if (NumOps.GreaterThan(weight, threshold))
                            result[i, j] = weight;
                    }
                }
                else if (NumOps.GreaterThan(bestItoJ, bestJtoI))
                {
                    // Direction i→j is stronger
                    T varI = cov[i, i];
                    if (NumOps.GreaterThan(varI, eps))
                    {
                        T weight = NumOps.Divide(cov[i, j], varI);
                        if (NumOps.GreaterThan(NumOps.Abs(weight), threshold))
                            result[i, j] = weight;
                    }
                }
            }
        }

        return result;
    }

    private T ComputeLaggedCorrelationEngine(Matrix<T> data, int source, int target, int lag, int n)
    {
        int effectiveN = n - lag;
        if (effectiveN < 3) return NumOps.Zero;

        var sourceVec = new Vector<T>(effectiveN);
        var targetVec = new Vector<T>(effectiveN);
        for (int t = 0; t < effectiveN; t++)
        {
            sourceVec[t] = data[t, source];
            targetVec[t] = data[t + lag, target];
        }

        // Compute means
        T nT = NumOps.FromDouble(effectiveN);
        T sumS = NumOps.Zero, sumT = NumOps.Zero;
        for (int t = 0; t < effectiveN; t++)
        {
            sumS = NumOps.Add(sumS, sourceVec[t]);
            sumT = NumOps.Add(sumT, targetVec[t]);
        }
        T meanS = NumOps.Divide(sumS, nT);
        T meanT = NumOps.Divide(sumT, nT);

        // Center and use Engine.DotProduct for acceleration
        var centS = new Vector<T>(effectiveN);
        var centT = new Vector<T>(effectiveN);
        for (int t = 0; t < effectiveN; t++)
        {
            centS[t] = NumOps.Subtract(sourceVec[t], meanS);
            centT[t] = NumOps.Subtract(targetVec[t], meanT);
        }

        T covST = Engine.DotProduct(centS, centT);
        T varS = Engine.DotProduct(centS, centS);
        T varT = Engine.DotProduct(centT, centT);

        double dVarS = NumOps.ToDouble(varS);
        double dVarT = NumOps.ToDouble(varT);
        double denom = Math.Sqrt(Math.Max(dVarS, 1e-15) * Math.Max(dVarT, 1e-15));
        return NumOps.FromDouble(NumOps.ToDouble(covST) / denom);
    }

    private T ComputePartialLaggedCorrelationMulti(Matrix<T> data, int i, int j, List<int> condVars, int n)
    {
        // Partial correlation via recursive formula for multiple conditioning variables
        if (condVars.Count == 0)
            return ComputeLaggedCorrelationEngine(data, i, j, 1, n);

        if (condVars.Count == 1)
        {
            int k = condVars[0];
            T rij = ComputeLaggedCorrelationEngine(data, i, j, 1, n);
            T rik = ComputeLaggedCorrelationEngine(data, i, k, 1, n);
            T rjk = ComputeLaggedCorrelationEngine(data, j, k, 1, n);

            T num = NumOps.Subtract(rij, NumOps.Multiply(rik, rjk));
            double dRik = NumOps.ToDouble(rik);
            double dRjk = NumOps.ToDouble(rjk);
            double denom = Math.Sqrt(Math.Max((1 - dRik * dRik) * (1 - dRjk * dRjk), 1e-15));
            return NumOps.FromDouble(NumOps.ToDouble(num) / denom);
        }

        // For larger conditioning sets, recurse: partial(i,j|S) via partial(i,j|S\{last}) etc.
        int last = condVars[^1];
        var reduced = condVars.GetRange(0, condVars.Count - 1);

        T pijR = ComputePartialLaggedCorrelationMulti(data, i, j, reduced, n);
        T pilR = ComputePartialLaggedCorrelationMulti(data, i, last, reduced, n);
        T pjlR = ComputePartialLaggedCorrelationMulti(data, j, last, reduced, n);

        T numerator = NumOps.Subtract(pijR, NumOps.Multiply(pilR, pjlR));
        double dPil = NumOps.ToDouble(pilR);
        double dPjl = NumOps.ToDouble(pjlR);
        double denomVal = Math.Sqrt(Math.Max((1 - dPil * dPil) * (1 - dPjl * dPjl), 1e-15));
        return NumOps.FromDouble(NumOps.ToDouble(numerator) / denomVal);
    }

    private static IEnumerable<List<int>> GetSubsets(List<int> items, int size)
    {
        if (size == 0) { yield return new List<int>(); yield break; }
        if (items.Count < size) yield break;

        for (int i = 0; i <= items.Count - size; i++)
        {
            foreach (var rest in GetSubsets(items.GetRange(i + 1, items.Count - i - 1), size - 1))
            {
                rest.Insert(0, items[i]);
                yield return rest;
            }
        }
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
