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
[ResearchPaper("Being Bayesian About Network Structure", "https://doi.org/10.1023/B:MACH.0000033120.25888.1e", Year = 2003, Authors = "Nir Friedman, Daphne Koller")]
public class OrderMCMCAlgorithm<T> : BayesianCausalBase<T>
{
    /// <summary>
    /// Minimum absolute weight for an edge to be included in the DAG.
    /// </summary>
    private const double MinEdgeWeight = 1e-6;

    /// <summary>
    /// Regularization epsilon for numerical stability in matrix inversion.
    /// </summary>
    private const double RegularizationEpsilon = 1e-10;

    /// <summary>
    /// Pivot tolerance for detecting singular/near-singular matrices during Gaussian elimination.
    /// </summary>
    private const double PivotTolerance = 1e-12;

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
        var edgeCounts = new Tensor<T>(new[] { d, d });
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
                edgeCounts = AccumulateEdgeCounts(edgeCounts, dag);
                sampleCount++;
            }
        }

        // Return edges in >50% of posterior samples with averaged weights
        return MajorityEdgeFrequencies(edgeCounts, sampleCount);
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
            if (parents.Length == 0)
                continue;

            var weights = ComputeMultivariateOLSWeights(cov, parents, target);
            // BIC already decided these parents improve the model — preserve
            // the edge even when OLS gives a near-zero weight by using a
            // minimum weight, preventing coefficient estimation from erasing
            // structurally selected parents.
            // NaN/Infinity coefficients are skipped (no meaningful causal effect), and
            // coefficients at or below MinEdgeWeight leave the edge as zero.
            var finiteWeights = Engine.TensorWhere(Engine.TensorIsFinite(weights), weights, Engine.TensorZerosLike(weights));
            var kept = KeepWhereAbove(Engine.TensorAbs(finiteWeights), MinEdgeWeight, finiteWeights);
            var column = Engine.TensorScatter(new Tensor<T>(new[] { d }),
                new Tensor<int>(parents, new[] { parents.Length }), kept, 0);
            Engine.SetColumn(W, target, column.ToVector());
        }
        return W;
    }

    /// <summary>
    /// Computes multivariate OLS regression weights: beta = Cov(parents,parents)^{-1} * Cov(parents,target).
    /// This correctly accounts for correlations among parents, yielding unbiased partial regression coefficients.
    /// </summary>
    private Tensor<T> ComputeMultivariateOLSWeights(Matrix<T> cov, int[] parents, int target)
    {
        int k = parents.Length;
        T eps = NumOps.FromDouble(RegularizationEpsilon);

        // Single parent: use simple univariate formula
        if (k == 1)
        {
            T varFrom = cov[parents[0], parents[0]];
            if (!NumOps.GreaterThan(varFrom, eps))
                return new Tensor<T>(new[] { 1 });
            return new Tensor<T>(new[] { NumOps.Divide(cov[parents[0], target], varFrom) }, new[] { 1 });
        }

        // Extract Cov(parents, parents) sub-matrix and Cov(parents, target) vector
        var parentIndex = new Tensor<int>(parents, new[] { k });
        var parentRows = Engine.TensorIndexSelect(Tensor<T>.FromMatrix(cov), parentIndex, 0);
        var covPP = Engine.TensorIndexSelect(parentRows, parentIndex, 1);
        var covPT = Engine.TensorIndexSelect(parentRows, new Tensor<int>(new[] { target }, new[] { 1 }), 1);

        // Add small regularization to diagonal for numerical stability
        covPP = Engine.TensorAdd(covPP, Engine.TensorMultiplyScalar(Engine.TensorEye<T>(k), eps));

        // Gaussian elimination with partial pivoting: singular or near-singular (a pivot below
        // PivotTolerance) returns zeros.
        var (_, _, u) = LinalgDispatch<T>.LU(covPP);
        var smallPivots = Engine.TensorLessThan(Engine.TensorAbs(Engine.TensorDiagonal(u)), NumOps.FromDouble(PivotTolerance));
        if (Engine.TensorCountNonzero(smallPivots) > 0)
            return new Tensor<T>(new[] { k });

        var (solution, _) = LinalgDispatch<T>.SolveEx(covPP, covPT);
        return solution.Reshape(new[] { k });
    }
}
