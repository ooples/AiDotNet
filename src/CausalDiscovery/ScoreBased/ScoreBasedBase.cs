using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// Base class for score-based causal discovery algorithms (GES, FGES, Hill Climbing, Tabu, etc.).
/// </summary>
/// <remarks>
/// <para>
/// Score-based methods search over the space of DAGs by evaluating each candidate graph
/// using a scoring criterion (typically BIC or BDeu). They use search strategies like
/// greedy equivalence search, hill climbing, or tabu search to find high-scoring graphs.
/// </para>
/// <para>
/// <b>For Beginners:</b> Score-based methods give each possible causal graph a "grade"
/// (score) based on how well it explains the data. They then search for the graph
/// with the best grade. Higher scores mean the graph fits the data better while
/// remaining simple (not too many edges).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class ScoreBasedBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.ScoreBasedSearch;

    /// <summary>
    /// BIC penalty discount factor. Higher values favor sparser graphs.
    /// </summary>
    protected double PenaltyDiscount { get; set; } = 1.0;

    /// <summary>
    /// Maximum number of parents per node.
    /// </summary>
    protected int MaxParents { get; set; } = int.MaxValue;

    /// <summary>
    /// Applies options from CausalDiscoveryOptions.
    /// </summary>
    protected void ApplyScoreOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        if (options.SparsityPenalty.HasValue) PenaltyDiscount = options.SparsityPenalty.Value;
        if (options.MaxParents.HasValue) MaxParents = options.MaxParents.Value;
        if (options.MaxIterations.HasValue) MaxIterations = options.MaxIterations.Value;
    }

    /// <summary>
    /// Maximum number of search iterations.
    /// </summary>
    protected int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Computes the BIC score for a variable given its parents.
    /// BIC = -n * log(RSS/n) - penalty * (k+1) * log(n)
    /// </summary>
    protected double ComputeBIC(Matrix<T> data, int target, HashSet<int> parents)
    {
        int n = data.Rows;
        if (parents.Count == 0)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += NumOps.ToDouble(data[i, target]);
            mean /= n;
            double rss = 0;
            for (int i = 0; i < n; i++)
            {
                double v = NumOps.ToDouble(data[i, target]) - mean;
                rss += v * v;
            }
            return -n * Math.Log(rss / n + 1e-10) - PenaltyDiscount * Math.Log(n);
        }

        var parentList = parents.ToList();
        int k = parentList.Count;

        // Build XtX and Xty using generic operations
        var XtX = new Matrix<T>(k, k);
        var Xty = new Vector<T>(k);
        for (int i = 0; i < k; i++)
        {
            int fi = parentList[i];
            for (int j = 0; j < k; j++)
            {
                int fj = parentList[j];
                T sum = NumOps.Zero;
                for (int r = 0; r < n; r++)
                    sum = NumOps.Add(sum, NumOps.Multiply(data[r, fi], data[r, fj]));
                XtX[i, j] = sum;
            }
            T sumY = NumOps.Zero;
            for (int r = 0; r < n; r++)
                sumY = NumOps.Add(sumY, NumOps.Multiply(data[r, fi], data[r, target]));
            Xty[i] = sumY;
        }

        // Ridge regularization
        T ridge = NumOps.FromDouble(1e-6);
        for (int i = 0; i < k; i++)
            XtX[i, i] = NumOps.Add(XtX[i, i], ridge);

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(XtX, Xty, MatrixDecompositionType.Lu);

        double totalRss = 0;
        for (int r = 0; r < n; r++)
        {
            T pred = NumOps.Zero;
            for (int i = 0; i < k; i++)
                pred = NumOps.Add(pred, NumOps.Multiply(beta[i], data[r, parentList[i]]));
            double err = NumOps.ToDouble(data[r, target]) - NumOps.ToDouble(pred);
            totalRss += err * err;
        }

        return -n * Math.Log(totalRss / n + 1e-10) - PenaltyDiscount * (k + 1) * Math.Log(n);
    }

    /// <summary>
    /// Computes the absolute Pearson correlation between two columns of data.
    /// </summary>
    protected double ComputeAbsCorrelation(Matrix<T> data, int col1, int col2)
    {
        int n = data.Rows;
        double m1 = 0, m2 = 0;
        for (int k = 0; k < n; k++)
        {
            m1 += NumOps.ToDouble(data[k, col1]);
            m2 += NumOps.ToDouble(data[k, col2]);
        }
        m1 /= n; m2 /= n;

        double sij = 0, sii = 0, sjj = 0;
        for (int k = 0; k < n; k++)
        {
            double d1 = NumOps.ToDouble(data[k, col1]) - m1;
            double d2 = NumOps.ToDouble(data[k, col2]) - m2;
            sij += d1 * d2; sii += d1 * d1; sjj += d2 * d2;
        }

        return (sii > 1e-10 && sjj > 1e-10) ? Math.Abs(sij / Math.Sqrt(sii * sjj)) : 0;
    }

    /// <summary>
    /// Checks if adding an edge from parent to child would create a cycle.
    /// </summary>
    protected static bool WouldCreateCycle(HashSet<int>[] parents, int from, int to, int d)
    {
        var visited = new HashSet<int>();
        var queue = new Queue<int>();
        queue.Enqueue(from);

        while (queue.Count > 0)
        {
            int current = queue.Dequeue();
            if (current == to) return true;
            if (!visited.Add(current)) continue;

            foreach (int parent in parents[current])
                queue.Enqueue(parent);
        }

        return false;
    }

}
