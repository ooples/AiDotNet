using AiDotNet.Enums;

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
    protected double ComputeBIC(double[,] X, int n, int target, HashSet<int> parents)
    {
        int d = X.GetLength(1);
        if (parents.Count == 0)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += X[i, target];
            mean /= n;
            double rss = 0;
            for (int i = 0; i < n; i++) rss += (X[i, target] - mean) * (X[i, target] - mean);
            return -n * Math.Log(rss / n + 1e-10) - PenaltyDiscount * Math.Log(n);
        }

        var parentList = parents.ToList();
        int k = parentList.Count;

        var XtX = new double[k, k];
        var Xty = new double[k];
        for (int i = 0; i < k; i++)
        {
            int fi = parentList[i];
            for (int j = 0; j < k; j++)
            {
                int fj = parentList[j];
                for (int r = 0; r < n; r++) XtX[i, j] += X[r, fi] * X[r, fj];
            }
            for (int r = 0; r < n; r++) Xty[i] += X[r, fi] * X[r, target];
        }

        var beta = SolveSystem(XtX, Xty, k);

        double totalRss = 0;
        for (int r = 0; r < n; r++)
        {
            double pred = 0;
            for (int i = 0; i < k; i++) pred += beta[i] * X[r, parentList[i]];
            double err = X[r, target] - pred;
            totalRss += err * err;
        }

        return -n * Math.Log(totalRss / n + 1e-10) - PenaltyDiscount * (k + 1) * Math.Log(n);
    }

    /// <summary>
    /// Solves a linear system Ax = b using Gaussian elimination.
    /// </summary>
    protected static double[] SolveSystem(double[,] A, double[] b, int size)
    {
        for (int i = 0; i < size; i++) A[i, i] += 1e-6;

        var aug = new double[size, size + 1];
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++) aug[i, j] = A[i, j];
            aug[i, size] = b[i];
        }

        for (int col = 0; col < size; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < size; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;
            for (int j = 0; j <= size; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);
            if (Math.Abs(aug[col, col]) < 1e-10) continue;
            for (int row = col + 1; row < size; row++)
            {
                double factor = aug[row, col] / aug[col, col];
                for (int j = col; j <= size; j++) aug[row, j] -= factor * aug[col, j];
            }
        }

        var x = new double[size];
        for (int i = size - 1; i >= 0; i--)
        {
            x[i] = aug[i, size];
            for (int j = i + 1; j < size; j++) x[i] -= aug[i, j] * x[j];
            x[i] /= (Math.Abs(aug[i, i]) > 1e-10 ? aug[i, i] : 1);
        }

        return x;
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
