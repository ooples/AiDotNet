using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// Iterative MCMC — iteratively refined MCMC for Bayesian network structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Iterative MCMC improves MCMC mixing by alternating between different proposal mechanisms:
/// edge additions, deletions, and reversals. It uses multiple restarts with the best DAG
/// from each restart used to seed the next, progressively refining the search.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize DAG from empty graph or previous restart's best</item>
/// <item>Propose: randomly add, delete, or reverse an edge</item>
/// <item>Check acyclicity of proposed DAG</item>
/// <item>Compute BIC score difference</item>
/// <item>Accept/reject via Metropolis-Hastings</item>
/// <item>After burn-in, accumulate edge posterior probabilities</item>
/// <item>Restart with best DAG found so far as seed</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard MCMC can get "stuck" in local optima. Iterative MCMC
/// uses clever tricks to escape these traps and explore more of the solution space,
/// leading to better posterior estimates.
/// </para>
/// <para>
/// Reference: Kuipers et al. (2017), "Efficient Structure Learning and Sampling of
/// Bayesian Networks", arXiv.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Efficient Structure Learning and Sampling of Bayesian Networks", "https://arxiv.org/abs/1803.07859", Year = 2017, Authors = "Jack Kuipers, Giusi Moffa, David Heckerman")]
public class IterativeMCMCAlgorithm<T> : BayesianCausalBase<T>
{
    private readonly int _maxParents;
    private readonly int _numRestarts;

    /// <inheritdoc/>
    public override string Name => "IterativeMCMC";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public IterativeMCMCAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyBayesianOptions(options);
        if (NumSamples < 100)
            throw new ArgumentException("NumSamples must be at least 100 for MCMC convergence.");
        _maxParents = options?.MaxParents ?? 5;
        if (_maxParents < 1)
            throw new ArgumentException("MaxParents must be at least 1.");
        _numRestarts = 5;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        var cov = ComputeCovarianceMatrix(data);
        int samplesPerRestart = NumSamples / _numRestarts;
        int burnIn = samplesPerRestart / 5;

        var edgeCounts = new Matrix<T>(d, d);
        int totalSamples = 0;

        var parentSets = new HashSet<int>[d];
        for (int i = 0; i < d; i++) parentSets[i] = [];

        double bestGlobalScore = double.NegativeInfinity;
        HashSet<int>[]? bestParents = null;

        for (int restart = 0; restart < _numRestarts; restart++)
        {
            if (bestParents != null)
            {
                for (int i = 0; i < d; i++)
                    parentSets[i] = new HashSet<int>(bestParents[i]);
            }
            else
            {
                for (int i = 0; i < d; i++)
                    parentSets[i] = [];
            }

            double currentScore = ComputeDAGScore(data, parentSets, d);
            double bestRestartScore = currentScore;
            HashSet<int>[] bestRestartParents = parentSets.Select(p => new HashSet<int>(p)).ToArray();

            for (int iter = 0; iter < samplesPerRestart; iter++)
            {
                int moveType = rng.Next(3);
                int from = rng.Next(d);
                int to = rng.Next(d);
                if (from == to) continue;

                if (moveType == 0 && !parentSets[to].Contains(from) && parentSets[to].Count < _maxParents)
                {
                    if (!WouldCreateCycle(parentSets, from, to, d))
                    {
                        parentSets[to].Add(from);
                        double newScore = ComputeDAGScore(data, parentSets, d);
                        double logAccept = newScore - currentScore;
                        if (logAccept >= 0 || Math.Log(rng.NextDouble()) < logAccept)
                            currentScore = newScore;
                        else
                            parentSets[to].Remove(from);
                    }
                }
                else if (moveType == 1 && parentSets[to].Contains(from))
                {
                    parentSets[to].Remove(from);
                    double newScore = ComputeDAGScore(data, parentSets, d);
                    double logAccept = newScore - currentScore;
                    if (logAccept >= 0 || Math.Log(rng.NextDouble()) < logAccept)
                        currentScore = newScore;
                    else
                        parentSets[to].Add(from);
                }
                else if (moveType == 2 && parentSets[to].Contains(from) && parentSets[from].Count < _maxParents)
                {
                    parentSets[to].Remove(from);
                    if (!WouldCreateCycle(parentSets, to, from, d))
                    {
                        parentSets[from].Add(to);
                        double newScore = ComputeDAGScore(data, parentSets, d);
                        double logAccept = newScore - currentScore;
                        if (logAccept >= 0 || Math.Log(rng.NextDouble()) < logAccept)
                            currentScore = newScore;
                        else
                        {
                            parentSets[from].Remove(to);
                            parentSets[to].Add(from);
                        }
                    }
                    else
                        parentSets[to].Add(from);
                }

                if (currentScore > bestRestartScore)
                {
                    bestRestartScore = currentScore;
                    bestRestartParents = parentSets.Select(p => new HashSet<int>(p)).ToArray();
                }

                if (iter >= burnIn)
                {
                    for (int j = 0; j < d; j++)
                        foreach (int p in parentSets[j])
                            edgeCounts[p, j] = NumOps.Add(edgeCounts[p, j], NumOps.One);
                    totalSamples++;
                }
            }

            if (bestRestartScore > bestGlobalScore)
            {
                bestGlobalScore = bestRestartScore;
                bestParents = bestRestartParents;
            }
        }

        // Build result with OLS weights for edges with posterior > 0.5
        var result = new Matrix<T>(d, d);
        if (totalSamples > 0)
        {
            T sampleCountT = NumOps.FromDouble(totalSamples);
            T halfT = NumOps.FromDouble(0.5);
            T epsT = NumOps.FromDouble(1e-6);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    T freq = NumOps.Divide(edgeCounts[i, j], sampleCountT);
                    if (NumOps.GreaterThan(freq, halfT))
                    {
                        T weight = ComputeOLSWeight(cov, i, j);
                        if (NumOps.GreaterThan(NumOps.Abs(weight), epsT))
                            result[i, j] = weight;
                    }
                }
        }

        return result;
    }

    private double ComputeDAGScore(Matrix<T> data, HashSet<int>[] parentSets, int d)
    {
        double totalScore = 0;
        for (int j = 0; j < d; j++)
            totalScore -= ComputeBICScore(data, j, [.. parentSets[j]]);
        return totalScore;
    }

    private static bool WouldCreateCycle(HashSet<int>[] parents, int from, int to, int d)
    {
        var visited = new bool[d];
        var queue = new Queue<int>();
        queue.Enqueue(from);
        visited[from] = true;

        while (queue.Count > 0)
        {
            int current = queue.Dequeue();
            foreach (int parent in parents[current])
            {
                if (parent == to) return true;
                if (!visited[parent])
                {
                    visited[parent] = true;
                    queue.Enqueue(parent);
                }
            }
        }

        return false;
    }

    private T ComputeOLSWeight(Matrix<T> cov, int from, int to)
    {
        T varFrom = cov[from, from];
        T eps = NumOps.FromDouble(1e-10);
        if (!NumOps.GreaterThan(varFrom, eps))
            return NumOps.Zero;
        return NumOps.Divide(cov[from, to], varFrom);
    }
}
