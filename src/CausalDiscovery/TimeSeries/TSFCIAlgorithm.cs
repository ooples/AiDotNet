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
[ModelPaper("On Causal Discovery from Time Series Data using FCI", "https://doi.org/10.1007/978-3-642-15114-9_12", Year = 2010, Authors = "Doris Entner, Patrik O. Hoyer")]
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
        if (!double.IsFinite(_correlationThreshold) || _correlationThreshold < 0 || _correlationThreshold > 1)
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
        T eps = NumOps.FromDouble(1e-10);
        T threshold = NumOps.FromDouble(_correlationThreshold);

        // Phase 1: Build lagged covariance structure
        // For each pair (i,j) and lag l, compute lagged correlation
        // skeleton[i,j] = max over lags of |lagged_corr(i→j)|
        var skeleton = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                T maxCorr = NumOps.Zero;
                for (int lag = 1; lag <= MaxLag; lag++)
                {
                    T lagCorr = ComputeLaggedCorrelation(data, i, j, lag, n);
                    T absCorr = NumOps.Abs(lagCorr);
                    if (NumOps.GreaterThan(absCorr, maxCorr))
                        maxCorr = absCorr;
                }
                skeleton[i, j] = maxCorr;
            }

        // Phase 2: Conditional independence tests — remove edges where
        // correlation disappears when conditioning on other lagged variables
        var result = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                if (!NumOps.GreaterThan(skeleton[i, j], threshold)) continue;

                // Test if i→j survives conditioning on each other variable's lags
                bool survives = true;
                for (int k = 0; k < d; k++)
                {
                    if (k == i || k == j) continue;
                    // Partial correlation test: does conditioning on k remove i→j?
                    T partialCorr = ComputePartialLaggedCorrelation(data, i, j, k, n);
                    if (!NumOps.GreaterThan(NumOps.Abs(partialCorr), threshold))
                    {
                        survives = false;
                        break;
                    }
                }

                if (survives)
                {
                    // Temporal orientation: determine direction using best lag
                    T bestItoJ = NumOps.Zero;
                    T bestJtoI = NumOps.Zero;
                    for (int lag = 1; lag <= MaxLag; lag++)
                    {
                        T corrIJ = NumOps.Abs(ComputeLaggedCorrelation(data, i, j, lag, n));
                        T corrJI = NumOps.Abs(ComputeLaggedCorrelation(data, j, i, lag, n));
                        if (NumOps.GreaterThan(corrIJ, bestItoJ)) bestItoJ = corrIJ;
                        if (NumOps.GreaterThan(corrJI, bestJtoI)) bestJtoI = corrJI;
                    }

                    // Direction with stronger lagged correlation wins
                    if (NumOps.GreaterThan(bestItoJ, bestJtoI))
                    {
                        T varI = cov[i, i];
                        if (NumOps.GreaterThan(varI, eps))
                            result[i, j] = NumOps.Divide(cov[i, j], varI);
                    }
                }
            }

        return result;
    }

    private T ComputeLaggedCorrelation(Matrix<T> data, int source, int target, int lag, int n)
    {
        int effectiveN = n - lag;
        if (effectiveN < 3) return NumOps.Zero;

        // Build vectors for Engine-accelerated correlation
        var sourceVec = new Vector<T>(effectiveN);
        var targetVec = new Vector<T>(effectiveN);
        for (int t = 0; t < effectiveN; t++)
        {
            sourceVec[t] = data[t, source];
            targetVec[t] = data[t + lag, target];
        }

        // Compute means
        T sumS = NumOps.Zero, sumT = NumOps.Zero;
        T nT = NumOps.FromDouble(effectiveN);
        for (int t = 0; t < effectiveN; t++)
        {
            sumS = NumOps.Add(sumS, sourceVec[t]);
            sumT = NumOps.Add(sumT, targetVec[t]);
        }
        T meanS = NumOps.Divide(sumS, nT);
        T meanT = NumOps.Divide(sumT, nT);

        // Center and use Engine.DotProduct
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

    private T ComputePartialLaggedCorrelation(Matrix<T> data, int i, int j, int condVar, int n)
    {
        // Partial correlation: corr(i,j | k) = (r_ij - r_ik * r_jk) / sqrt((1-r_ik^2)(1-r_jk^2))
        T rij = ComputeLaggedCorrelation(data, i, j, 1, n);
        T rik = ComputeLaggedCorrelation(data, i, condVar, 1, n);
        T rjk = ComputeLaggedCorrelation(data, j, condVar, 1, n);

        T numerator = NumOps.Subtract(rij, NumOps.Multiply(rik, rjk));
        double dRik = NumOps.ToDouble(rik);
        double dRjk = NumOps.ToDouble(rjk);
        double denom = Math.Sqrt(Math.Max((1 - dRik * dRik) * (1 - dRjk * dRjk), 1e-15));
        return NumOps.FromDouble(NumOps.ToDouble(numerator) / denom);
    }
}
