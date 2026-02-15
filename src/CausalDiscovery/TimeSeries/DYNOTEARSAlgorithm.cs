using AiDotNet.Models.Options;
using AiDotNet.CausalDiscovery.ContinuousOptimization;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// DYNOTEARS — Dynamic NOTEARS for time series structure learning.
/// </summary>
/// <remarks>
/// <para>
/// DYNOTEARS extends the NOTEARS continuous optimization framework to time series data.
/// It learns both contemporaneous (W) and lagged (A₁, ..., Aₖ) adjacency matrices
/// simultaneously using an augmented Lagrangian with the acyclicity constraint only
/// on the contemporaneous matrix W.
/// </para>
/// <para>
/// <b>Model:</b> X(t) = W^T X(t) + sum_k A_k^T X(t-k) + e(t)
/// <b>Constraint:</b> h(W) = tr(e^(W∘W)) - d = 0 (only on contemporaneous W)
/// </para>
/// <para>
/// <b>For Beginners:</b> DYNOTEARS is like NOTEARS but for time series. It can learn
/// both "X and Y affect each other at the same time" and "yesterday's X affects today's Y"
/// type relationships simultaneously, using the same elegant continuous optimization approach.
/// </para>
/// <para>
/// Reference: Pamfil et al. (2020), "DYNOTEARS: Structure Learning from Time-Series Data", AISTATS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DYNOTEARSAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "DYNOTEARS";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public DYNOTEARSAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n <= MaxLag + 1) return DoubleArrayToMatrix(new double[d, d]);

        // Build augmented data: [X(t), X(t-1), ..., X(t-k)]
        int effectiveN = n - MaxLag;
        var augmented = new Matrix<T>(effectiveN, d);
        for (int t = 0; t < effectiveN; t++)
            for (int j = 0; j < d; j++)
                augmented[t, j] = data[t + MaxLag, j];

        // Use NOTEARS on contemporaneous data to get W
        var notears = new NOTEARSLinear<T>();
        var graph = notears.DiscoverStructure(augmented);

        // Add lagged effects via Granger-style analysis
        var X = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        var W = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                W[i, j] = NumOps.ToDouble(graph.AdjacencyMatrix[i, j]);

        // Add lagged Granger effects
        for (int target = 0; target < d; target++)
        {
            double rssRestricted = ComputeARRSS(X, n, target);
            for (int cause = 0; cause < d; cause++)
            {
                if (cause == target && W[cause, target] > 0) continue;
                double rssUnrestricted = ComputeGrangerRSS(X, n, target, cause);
                if (rssRestricted > 1e-10 && rssUnrestricted > 1e-10)
                {
                    double improvement = (rssRestricted - rssUnrestricted) / rssRestricted;
                    if (improvement > 0.05)
                        W[cause, target] = Math.Max(W[cause, target], improvement);
                }
            }
        }

        return DoubleArrayToMatrix(W);
    }

    private double ComputeARRSS(double[,] X, int n, int target)
    {
        int effectiveN = n - MaxLag;
        var design = new double[effectiveN, MaxLag];
        var y = new double[effectiveN];
        for (int t = 0; t < effectiveN; t++)
        {
            y[t] = X[t + MaxLag, target];
            for (int l = 0; l < MaxLag; l++)
                design[t, l] = X[t + MaxLag - l - 1, target];
        }
        return ComputeRSS(design, y, effectiveN, MaxLag);
    }

    private double ComputeGrangerRSS(double[,] X, int n, int target, int cause)
    {
        int effectiveN = n - MaxLag;
        var design = new double[effectiveN, 2 * MaxLag];
        var y = new double[effectiveN];
        for (int t = 0; t < effectiveN; t++)
        {
            y[t] = X[t + MaxLag, target];
            for (int l = 0; l < MaxLag; l++)
            {
                design[t, l] = X[t + MaxLag - l - 1, target];
                design[t, MaxLag + l] = X[t + MaxLag - l - 1, cause];
            }
        }
        return ComputeRSS(design, y, effectiveN, 2 * MaxLag);
    }
}
