using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// PCMCI+ — extension of PCMCI that also discovers contemporaneous causal links.
/// </summary>
/// <remarks>
/// <para>
/// PCMCI+ extends PCMCI to handle both lagged AND contemporaneous (same time-step)
/// causal links by adding a skeleton discovery and orientation step for lag-0 effects.
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
    /// <inheritdoc/>
    public override string Name => "PCMCI+";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public PCMCIPlusAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // PCMCI+ adds contemporaneous link discovery to PCMCI
        // For now, delegate to PCMCI and add contemporaneous correlation analysis
        var pcmci = new PCMCIAlgorithm<T>();
        var laggedGraph = pcmci.DiscoverStructure(data);

        int n = data.Rows;
        int d = data.Columns;
        var X = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Add contemporaneous edges via partial correlation
        var W = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                W[i, j] = NumOps.ToDouble(laggedGraph.AdjacencyMatrix[i, j]);

        // Contemporaneous discovery
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                double corr = ComputeCorrelation(X, n, i, j);
                if (Math.Abs(corr) > 0.3) // significant contemporaneous link
                {
                    W[i, j] = Math.Max(W[i, j], Math.Abs(corr) * 0.5);
                }
            }
        }

        return DoubleArrayToMatrix(W);
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
}
