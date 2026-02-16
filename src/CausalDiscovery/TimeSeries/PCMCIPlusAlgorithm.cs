using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// PCMCI+ — extension of PCMCI that also discovers contemporaneous causal links.
/// </summary>
/// <remarks>
/// <para>
/// PCMCI+ extends PCMCI to handle both lagged AND contemporaneous (same time-step)
/// causal links by adding a skeleton discovery and orientation step for lag-0 effects.
/// It applies the same Fisher z-test based conditional independence testing at lag 0
/// conditioned on the lagged parents discovered by PCMCI.
/// </para>
/// <para>
/// <b>For Beginners:</b> PCMCI only finds "yesterday's X causes today's Y" relationships.
/// PCMCI+ also finds "today's X causes today's Y" relationships, which are important
/// when variables influence each other faster than the measurement interval.
/// </para>
/// <para>
/// Reference: Runge (2020), "Discovering Contemporaneous and Lagged Causal Relations
/// in Autocorrelated Nonlinear Time Series Datasets", UAI.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PCMCIPlusAlgorithm<T> : TimeSeriesCausalBase<T>
{
    private double _alpha = 0.05;

    /// <inheritdoc/>
    public override string Name => "PCMCI+";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public PCMCIPlusAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        if (options?.SignificanceLevel.HasValue == true) _alpha = options.SignificanceLevel.Value;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        int effectiveN = n - MaxLag;
        if (effectiveN < 2 * d) return new Matrix<T>(d, d);

        // Step 1: Run PCMCI to get lagged causal structure
        var pcmci = new PCMCIAlgorithm<T>(new CausalDiscoveryOptions
        {
            SignificanceLevel = _alpha,
            MaxIterations = MaxLag
        });
        var laggedGraph = pcmci.DiscoverStructure(data);

        // Copy lagged edges into result
        var W = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                W[i, j] = laggedGraph.AdjacencyMatrix[i, j];

        // Step 2: Contemporaneous discovery via conditional independence testing
        // Condition on the lagged parents of both variables (discovered from PCMCI)
        int offset = n - effectiveN;
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                // Build conditioning set from lagged parents of both i and j
                var condVarIndices = new List<int>();
                for (int k = 0; k < d; k++)
                {
                    if (Math.Abs(NumOps.ToDouble(laggedGraph.AdjacencyMatrix[k, i])) > 0 ||
                        Math.Abs(NumOps.ToDouble(laggedGraph.AdjacencyMatrix[k, j])) > 0)
                    {
                        condVarIndices.Add(k);
                    }
                }

                double partCorr;
                if (condVarIndices.Count == 0)
                {
                    // No lagged parents: use simple correlation over effective sample window
                    partCorr = ComputeWindowCorrelation(data, effectiveN, offset, i, j);
                }
                else
                {
                    // Residualize both variables on their lagged parents
                    partCorr = ComputeContemporaneousPartialCorrelation(
                        data, i, j, condVarIndices, effectiveN, offset);
                }

                double pValue = FisherZTestPValue(partCorr, effectiveN, condVarIndices.Count);

                if (pValue <= _alpha)
                {
                    // Significant contemporaneous link — orient using causal ordering heuristic
                    int parentsOfI = 0, parentsOfJ = 0;
                    for (int k = 0; k < d; k++)
                    {
                        if (Math.Abs(NumOps.ToDouble(laggedGraph.AdjacencyMatrix[k, i])) > 0) parentsOfI++;
                        if (Math.Abs(NumOps.ToDouble(laggedGraph.AdjacencyMatrix[k, j])) > 0) parentsOfJ++;
                    }

                    T strength = NumOps.FromDouble(Math.Abs(partCorr));
                    if (parentsOfI <= parentsOfJ)
                    {
                        double current = NumOps.ToDouble(W[i, j]);
                        if (Math.Abs(partCorr) > current)
                            W[i, j] = strength;
                    }
                    else
                    {
                        double current = NumOps.ToDouble(W[j, i]);
                        if (Math.Abs(partCorr) > current)
                            W[j, i] = strength;
                    }
                }
            }
        }

        return W;
    }

    /// <summary>
    /// Computes partial correlation between variables i(t) and j(t) at lag 0,
    /// conditioned on lagged parents via OLS residualization.
    /// </summary>
    private double ComputeContemporaneousPartialCorrelation(
        Matrix<T> data, int i, int j,
        List<int> condVarIndices, int effectiveN, int offset)
    {
        int n = data.Rows;

        // Build conditioning matrix from lag-1 values of conditioning variables
        int numCond = condVarIndices.Count;
        var condMatrix = new Matrix<T>(effectiveN, numCond);
        for (int c = 0; c < numCond; c++)
        {
            int cVar = condVarIndices[c];
            for (int t = 0; t < effectiveN; t++)
            {
                int tIdx = offset + t - 1;
                condMatrix[t, c] = (tIdx >= 0 && tIdx < n) ? data[tIdx, cVar] : NumOps.Zero;
            }
        }

        // Build target vectors at lag 0
        var iVals = new Vector<T>(effectiveN);
        var jVals = new Vector<T>(effectiveN);
        for (int t = 0; t < effectiveN; t++)
        {
            iVals[t] = data[offset + t, i];
            jVals[t] = data[offset + t, j];
        }

        // Residualize both on conditioning set
        var residI = OLSResiduals(condMatrix, iVals, effectiveN, numCond);
        var residJ = OLSResiduals(condMatrix, jVals, effectiveN, numCond);

        return PearsonCorrelation(residI, residJ, effectiveN);
    }

    private Vector<T> OLSResiduals(Matrix<T> Z, Vector<T> y, int n, int p)
    {
        if (p == 0)
        {
            T mean = NumOps.Zero;
            for (int t = 0; t < n; t++) mean = NumOps.Add(mean, y[t]);
            mean = NumOps.Divide(mean, NumOps.FromDouble(n));
            var r = new Vector<T>(n);
            for (int t = 0; t < n; t++) r[t] = NumOps.Subtract(y[t], mean);
            return r;
        }

        // Build normal equations: ZtZ * beta = Zty
        var ZtZ = new Matrix<T>(p, p);
        var Zty = new Vector<T>(p);
        for (int a = 0; a < p; a++)
        {
            for (int b = a; b < p; b++)
            {
                T sum = NumOps.Zero;
                for (int t = 0; t < n; t++) sum = NumOps.Add(sum, NumOps.Multiply(Z[t, a], Z[t, b]));
                ZtZ[a, b] = sum;
                ZtZ[b, a] = sum;
            }
            T s = NumOps.Zero;
            for (int t = 0; t < n; t++) s = NumOps.Add(s, NumOps.Multiply(Z[t, a], y[t]));
            Zty[a] = s;
        }

        // Ridge for stability
        T ridge = NumOps.FromDouble(1e-10);
        for (int a = 0; a < p; a++) ZtZ[a, a] = NumOps.Add(ZtZ[a, a], ridge);

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(ZtZ, Zty, MatrixDecompositionType.Lu);

        var residuals = new Vector<T>(n);
        for (int t = 0; t < n; t++)
        {
            T pred = NumOps.Zero;
            for (int c = 0; c < p; c++) pred = NumOps.Add(pred, NumOps.Multiply(Z[t, c], beta[c]));
            residuals[t] = NumOps.Subtract(y[t], pred);
        }

        return residuals;
    }

    private double ComputeWindowCorrelation(Matrix<T> data, int count, int offset, int i, int j)
    {
        double mi = 0, mj = 0;
        for (int k = 0; k < count; k++)
        {
            int t = offset + k;
            mi += NumOps.ToDouble(data[t, i]); mj += NumOps.ToDouble(data[t, j]);
        }
        mi /= count; mj /= count;
        double sij = 0, sii = 0, sjj = 0;
        for (int k = 0; k < count; k++)
        {
            int t = offset + k;
            double di = NumOps.ToDouble(data[t, i]) - mi, dj = NumOps.ToDouble(data[t, j]) - mj;
            sij += di * dj; sii += di * di; sjj += dj * dj;
        }
        return (sii > 1e-10 && sjj > 1e-10) ? sij / Math.Sqrt(sii * sjj) : 0;
    }

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

    private static double FisherZTestPValue(double partialCorrelation, int n, int condSetSize)
    {
        int dof = n - condSetSize - 3;
        if (dof < 1) return 1.0;

        double r = Math.Max(-0.9999, Math.Min(0.9999, partialCorrelation));
        double z = 0.5 * Math.Log((1 + r) / (1 - r));
        double testStat = Math.Abs(z) * Math.Sqrt(dof);

        return 2.0 * NormalCdfComplement(testStat);
    }

    private static double NormalCdfComplement(double x)
    {
        if (x < 0) return 1.0 - NormalCdfComplement(-x);

        const double p = 0.2316419;
        const double b1 = 0.319381530;
        const double b2 = -0.356563782;
        const double b3 = 1.781477937;
        const double b4 = -1.821255978;
        const double b5 = 1.330274429;

        double t = 1.0 / (1.0 + p * x);
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
        double t5 = t4 * t;

        double phi = Math.Exp(-0.5 * x * x) / Math.Sqrt(2.0 * Math.PI);
        return phi * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5);
    }
}
