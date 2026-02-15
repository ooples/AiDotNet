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
        var X = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        int effectiveN = n - MaxLag;
        if (effectiveN < 2 * d) return DoubleArrayToMatrix(new double[d, d]);

        // Step 1: Run PCMCI to get lagged causal structure
        var pcmci = new PCMCIAlgorithm<T>(new CausalDiscoveryOptions
        {
            SignificanceLevel = _alpha,
            MaxIterations = MaxLag
        });
        var laggedGraph = pcmci.DiscoverStructure(data);

        // Copy lagged edges into result
        var W = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                W[i, j] = NumOps.ToDouble(laggedGraph.AdjacencyMatrix[i, j]);

        // Step 2: Contemporaneous discovery via conditional independence testing
        // Condition on the lagged parents of both variables (discovered from PCMCI)
        // This ensures we only find genuine contemporaneous links, not spurious ones
        int offset = n - effectiveN;
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                // Build conditioning set from lagged parents of both i and j
                // Use variables that have lagged causal edges to either i or j
                var condVarIndices = new List<int>();
                for (int k = 0; k < d; k++)
                {
                    if (NumOps.ToDouble(laggedGraph.AdjacencyMatrix[k, i]) > 0 ||
                        NumOps.ToDouble(laggedGraph.AdjacencyMatrix[k, j]) > 0)
                    {
                        condVarIndices.Add(k);
                    }
                }

                double partCorr;
                if (condVarIndices.Count == 0)
                {
                    // No lagged parents: use simple correlation
                    partCorr = ComputeCorrelation(X, n, i, j);
                }
                else
                {
                    // Residualize both variables on their lagged parents
                    partCorr = ComputeContemporaneousPartialCorrelation(
                        X, n, d, i, j, condVarIndices, effectiveN, offset);
                }

                double pValue = FisherZTestPValue(partCorr, effectiveN, condVarIndices.Count);

                if (pValue <= _alpha)
                {
                    // Significant contemporaneous link — orient using causal ordering heuristic
                    // The variable with more lagged parents is more likely to be downstream
                    int parentsOfI = 0, parentsOfJ = 0;
                    for (int k = 0; k < d; k++)
                    {
                        if (NumOps.ToDouble(laggedGraph.AdjacencyMatrix[k, i]) > 0) parentsOfI++;
                        if (NumOps.ToDouble(laggedGraph.AdjacencyMatrix[k, j]) > 0) parentsOfJ++;
                    }

                    double strength = Math.Abs(partCorr);
                    if (parentsOfI <= parentsOfJ)
                        W[i, j] = Math.Max(W[i, j], strength);
                    else
                        W[j, i] = Math.Max(W[j, i], strength);
                }
            }
        }

        return DoubleArrayToMatrix(W);
    }

    /// <summary>
    /// Computes partial correlation between variables i(t) and j(t) at lag 0,
    /// conditioned on lagged parents via OLS residualization.
    /// </summary>
    private static double ComputeContemporaneousPartialCorrelation(
        double[,] X, int n, int d, int i, int j,
        List<int> condVarIndices, int effectiveN, int offset)
    {
        // Build conditioning matrix from lag-1 values of conditioning variables
        int numCond = condVarIndices.Count;
        var condMatrix = new double[effectiveN, numCond];
        for (int c = 0; c < numCond; c++)
        {
            int cVar = condVarIndices[c];
            for (int t = 0; t < effectiveN; t++)
            {
                int tIdx = offset + t - 1;
                condMatrix[t, c] = (tIdx >= 0 && tIdx < n) ? X[tIdx, cVar] : 0.0;
            }
        }

        // Build target vectors at lag 0
        var iVals = new double[effectiveN];
        var jVals = new double[effectiveN];
        for (int t = 0; t < effectiveN; t++)
        {
            iVals[t] = X[offset + t, i];
            jVals[t] = X[offset + t, j];
        }

        // Residualize both on conditioning set
        var residI = OLSResiduals(condMatrix, iVals, effectiveN, numCond);
        var residJ = OLSResiduals(condMatrix, jVals, effectiveN, numCond);

        return PearsonCorrelation(residI, residJ, effectiveN);
    }

    private static double[] OLSResiduals(double[,] Z, double[] y, int n, int p)
    {
        if (p == 0)
        {
            double mean = 0;
            for (int t = 0; t < n; t++) mean += y[t];
            mean /= n;
            var r = new double[n];
            for (int t = 0; t < n; t++) r[t] = y[t] - mean;
            return r;
        }

        // Compute Z'Z and Z'y
        var ZtZ = new double[p, p];
        var Zty = new double[p];
        for (int a = 0; a < p; a++)
        {
            for (int b = a; b < p; b++)
            {
                double sum = 0;
                for (int t = 0; t < n; t++) sum += Z[t, a] * Z[t, b];
                ZtZ[a, b] = sum;
                ZtZ[b, a] = sum;
            }
            double s = 0;
            for (int t = 0; t < n; t++) s += Z[t, a] * y[t];
            Zty[a] = s;
        }

        // Ridge for stability
        for (int a = 0; a < p; a++) ZtZ[a, a] += 1e-10;

        // Solve via Gaussian elimination
        var beta = SolveLinearSystem(ZtZ, Zty, p);

        var residuals = new double[n];
        for (int t = 0; t < n; t++)
        {
            double pred = 0;
            for (int c = 0; c < p; c++) pred += Z[t, c] * beta[c];
            residuals[t] = y[t] - pred;
        }

        return residuals;
    }

    private static double[] SolveLinearSystem(double[,] A, double[] b, int p)
    {
        var M = new double[p, p];
        var rhs = new double[p];
        for (int i = 0; i < p; i++)
        {
            rhs[i] = b[i];
            for (int j = 0; j < p; j++) M[i, j] = A[i, j];
        }

        for (int k = 0; k < p; k++)
        {
            int maxRow = k;
            double maxVal = Math.Abs(M[k, k]);
            for (int i = k + 1; i < p; i++)
            {
                if (Math.Abs(M[i, k]) > maxVal) { maxVal = Math.Abs(M[i, k]); maxRow = i; }
            }

            if (maxRow != k)
            {
                for (int j = 0; j < p; j++) (M[k, j], M[maxRow, j]) = (M[maxRow, j], M[k, j]);
                (rhs[k], rhs[maxRow]) = (rhs[maxRow], rhs[k]);
            }

            if (Math.Abs(M[k, k]) < 1e-14)
            {
                // Near-singular: return zeros
                return new double[p];
            }

            for (int i = k + 1; i < p; i++)
            {
                double factor = M[i, k] / M[k, k];
                for (int j = k + 1; j < p; j++) M[i, j] -= factor * M[k, j];
                rhs[i] -= factor * rhs[k];
                M[i, k] = 0;
            }
        }

        var x = new double[p];
        for (int i = p - 1; i >= 0; i--)
        {
            double sum = rhs[i];
            for (int j = i + 1; j < p; j++) sum -= M[i, j] * x[j];
            x[i] = sum / M[i, i];
        }

        return x;
    }

    private static double ComputeCorrelation(double[,] X, int n, int i, int j)
    {
        double mi = 0, mj = 0;
        for (int k = 0; k < n; k++) { mi += X[k, i]; mj += X[k, j]; }
        mi /= n; mj /= n;
        double sij = 0, sii = 0, sjj = 0;
        for (int k = 0; k < n; k++)
        {
            double di = X[k, i] - mi, dj = X[k, j] - mj;
            sij += di * dj; sii += di * di; sjj += dj * dj;
        }
        return (sii > 1e-10 && sjj > 1e-10) ? sij / Math.Sqrt(sii * sjj) : 0;
    }

    private static double PearsonCorrelation(double[] x, double[] y, int n)
    {
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
        mx /= n; my /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - mx, dy = y[i] - my;
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
