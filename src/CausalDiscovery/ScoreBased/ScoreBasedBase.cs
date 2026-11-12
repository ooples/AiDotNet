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
        if (options.SparsityPenalty.HasValue) PenaltyDiscount = Math.Max(0, options.SparsityPenalty.Value);
        if (options.MaxParents.HasValue) MaxParents = Math.Max(1, options.MaxParents.Value);
        if (options.MaxIterations.HasValue) MaxIterations = Math.Max(1, options.MaxIterations.Value);
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
        int k = parents.Count;

        // No parents: the RSS about the mean, one parameter; otherwise k slopes + 1 intercept.
        T rss = k == 0
            ? CausalTensorHelpers.CenteredSumOfSquares(data, target)
            : CausalTensorHelpers.CenteredOlsRss(data, target, parents.ToList(), 1e-6);
        T fit = NumOps.Multiply(NumOps.FromDouble(-n),
            NumOps.Log(NumOps.Add(NumOps.Divide(rss, NumOps.FromDouble(n)), NumOps.FromDouble(1e-10))));
        return NumOps.ToDouble(NumOps.Subtract(fit, NumOps.FromDouble(PenaltyDiscount * (k + 1) * Math.Log(n))));
    }

    /// <summary>
    /// The [d x d] adjacency with weight max(|corr(from, to)|, <paramref name="minimumWeight"/>) on every
    /// edge from -> to with from in <paramref name="parentSets"/>[to], and 0 elsewhere, computed in T.
    /// </summary>
    private protected Matrix<T> CorrelationWeightedAdjacency(Matrix<T> data, HashSet<int>[] parentSets,
        double minimumWeight)
    {
        int d = data.Columns;
        var correlations = new Dictionary<int, T>();
        for (int to = 0; to < d; to++)
            foreach (int from in parentSets[to])
                correlations[from * d + to] = CausalTensorHelpers.PearsonCorrelation(data, from, to);

        if (correlations.Count == 0)
            return new Matrix<T>(d, d);

        var raw = Tensor<T>.FromMatrix(CausalTensorHelpers.FromValues(d, d, correlations));
        var selected = Tensor<T>.FromMatrix(CausalTensorHelpers.KeepPositions(Engine.TensorOnesLike(raw), correlations.Keys));
        var weighted = Engine.TensorMax(Engine.TensorAbs(raw), NumOps.FromDouble(minimumWeight));
        return Engine.TensorWhere(selected, weighted, Engine.TensorZerosLike(raw)).ToMatrix();
    }

    /// <summary>
    /// Computes the absolute Pearson correlation between two columns of data.
    /// </summary>
    protected double ComputeAbsCorrelation(Matrix<T> data, int col1, int col2)
    {
        return Math.Abs(NumOps.ToDouble(CausalTensorHelpers.PearsonCorrelation(data, col1, col2)));
    }

    /// <summary>
    /// Checks if adding an edge from parent to child would create a cycle.
    /// </summary>
    protected static bool WouldCreateCycle(HashSet<int>[] parents, int from, int to)
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
