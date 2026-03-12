using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// Order MCMC — MCMC sampling over variable orderings for Bayesian structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Order MCMC samples from the posterior over variable orderings (rather than over DAGs
/// directly). Given an ordering, the optimal DAG can be computed efficiently via greedy
/// parent selection with BIC scoring. The Markov chain proposes moves by swapping adjacent
/// elements in the ordering, accepting with Metropolis-Hastings probability.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize a random variable ordering</item>
/// <item>Propose a new ordering by swapping two adjacent elements</item>
/// <item>Compute optimal DAGs for both orderings using greedy BIC parent selection</item>
/// <item>Accept/reject via Metropolis-Hastings: accept with prob min(1, exp(score_new - score_old))</item>
/// <item>After burn-in, accumulate edge frequencies across accepted samples</item>
/// <item>Return edges that appear in more than 50% of posterior samples</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of searching over all possible graphs (very hard), this
/// method searches over all possible orderings of variables (much easier). Once you know
/// the order, finding the best graph is straightforward.
/// </para>
/// <para>
/// Reference: Friedman and Koller (2003), "Being Bayesian About Network Structure", MLJ.
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
[ModelPaper("Being Bayesian About Network Structure", "https://doi.org/10.1023/B:MACH.0000033120.25", Year = 2003, Authors = "Nir Friedman, Daphne Koller")]
public class OrderMCMCAlgorithm<T> : BayesianCausalBase<T>
{
    private readonly int _maxParents;
    private readonly int _burnIn;

    /// <inheritdoc/>
    public override string Name => "OrderMCMC";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public OrderMCMCAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyBayesianOptions(options);
        _maxParents = options?.MaxParents ?? 5;
        _burnIn = Math.Max(NumSamples / 5, 100);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);

        // Initialize random ordering
        var ordering = Enumerable.Range(0, d).ToArray();
        for (int i = d - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (ordering[i], ordering[j]) = (ordering[j], ordering[i]);
        }

        // Compute covariance matrix once for OLS weight computation
        var cov = ComputeCovarianceMatrix(data);

        double currentScore = ComputeOrderingScore(data, ordering, d);
        var edgeCounts = new Matrix<T>(d, d);
        int sampleCount = 0;

        for (int iter = 0; iter < NumSamples + _burnIn; iter++)
        {
            // Propose: swap two adjacent elements
            int pos = rng.Next(d - 1);
            (ordering[pos], ordering[pos + 1]) = (ordering[pos + 1], ordering[pos]);

            double proposedScore = ComputeOrderingScore(data, ordering, d);

            // Metropolis-Hastings acceptance (log-probabilities are scalar metrics)
            double logAccept = proposedScore - currentScore;
            if (logAccept >= 0 || Math.Log(rng.NextDouble()) < logAccept)
            {
                currentScore = proposedScore;
            }
            else
            {
                (ordering[pos], ordering[pos + 1]) = (ordering[pos + 1], ordering[pos]);
            }

            // After burn-in, accumulate edge frequencies
            if (iter >= _burnIn)
            {
                var dag = BuildDAGFromOrdering(data, ordering, d, cov);
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        if (!NumOps.Equals(dag[i, j], NumOps.Zero))
                            edgeCounts[i, j] = NumOps.Add(edgeCounts[i, j], NumOps.One);
                sampleCount++;
            }
        }

        // Return edges in >50% of posterior samples with averaged weights
        var result = new Matrix<T>(d, d);
        T sampleCountT = NumOps.FromDouble(sampleCount);
        T halfT = NumOps.FromDouble(0.5);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                T freq = NumOps.Divide(edgeCounts[i, j], sampleCountT);
                if (NumOps.GreaterThan(freq, halfT))
                    result[i, j] = freq;
            }

        return result;
    }

    /// <summary>
    /// Computes total negative BIC for an ordering (higher is better).
    /// BIC scores are scalar metrics so double is appropriate here.
    /// </summary>
    private double ComputeOrderingScore(Matrix<T> data, int[] ordering, int d)
    {
        double totalScore = 0;
        for (int idx = 0; idx < d; idx++)
        {
            int target = ordering[idx];
            var bestParents = SelectBestParents(data, target, ordering, idx);
            totalScore -= ComputeBICScore(data, target, bestParents);
        }
        return totalScore;
    }

    private int[] SelectBestParents(Matrix<T> data, int target, int[] ordering, int posInOrdering)
    {
        var parents = new List<int>();
        double bestScore = ComputeBICScore(data, target, []);

        for (int predIdx = 0; predIdx < posInOrdering && parents.Count < _maxParents; predIdx++)
        {
            int candidate = ordering[predIdx];
            var trial = parents.Concat([candidate]).ToArray();
            double trialScore = ComputeBICScore(data, target, trial);

            if (trialScore < bestScore)
            {
                parents.Add(candidate);
                bestScore = trialScore;
            }
        }

        return [.. parents];
    }

    private Matrix<T> BuildDAGFromOrdering(Matrix<T> data, int[] ordering, int d, Matrix<T> cov)
    {
        var W = new Matrix<T>(d, d);
        for (int idx = 1; idx < d; idx++)
        {
            int target = ordering[idx];
            var parents = SelectBestParents(data, target, ordering, idx);
            foreach (int parent in parents)
            {
                T weight = ComputeOLSWeight(cov, parent, target);
                T absWeight = NumOps.Abs(weight);
                if (NumOps.GreaterThan(absWeight, NumOps.FromDouble(1e-6)))
                    W[parent, target] = weight;
            }
        }
        return W;
    }

    /// <summary>
    /// Computes OLS regression weight using covariance matrix: w = cov(from,to) / var(from).
    /// Fully generic using Matrix&lt;T&gt; and NumOps.
    /// </summary>
    private T ComputeOLSWeight(Matrix<T> cov, int from, int to)
    {
        T varFrom = cov[from, from];
        T eps = NumOps.FromDouble(1e-10);
        if (!NumOps.GreaterThan(varFrom, eps))
            return NumOps.Zero;
        return NumOps.Divide(cov[from, to], varFrom);
    }
}
