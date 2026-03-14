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
[ModelPaper("Being Bayesian About Network Structure", "https://doi.org/10.1023/B:MACH.0000033120.25888.1e", Year = 2003, Authors = "Nir Friedman, Daphne Koller")]
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
            if (parents.Length == 0)
                continue;

            var weights = ComputeMultivariateOLSWeights(cov, parents, target);
            // BIC already decided these parents improve the model — preserve
            // the edge even when OLS gives a near-zero weight by using a
            // minimum weight, preventing coefficient estimation from erasing
            // structurally selected parents.
            for (int p = 0; p < parents.Length; p++)
            {
                T absWeight = NumOps.Abs(weights[p]);
                if (NumOps.GreaterThan(absWeight, NumOps.FromDouble(MinEdgeWeight)))
                    W[parents[p], target] = weights[p];
                else
                    W[parents[p], target] = NumOps.FromDouble(MinEdgeWeight);
            }
        }
        return W;
    }

    /// <summary>
    /// Computes multivariate OLS regression weights: beta = Cov(parents,parents)^{-1} * Cov(parents,target).
    /// This correctly accounts for correlations among parents, yielding unbiased partial regression coefficients.
    /// </summary>
    private T[] ComputeMultivariateOLSWeights(Matrix<T> cov, int[] parents, int target)
    {
        int k = parents.Length;
        T eps = NumOps.FromDouble(RegularizationEpsilon);

        // Single parent: use simple univariate formula
        if (k == 1)
        {
            T varFrom = cov[parents[0], parents[0]];
            if (!NumOps.GreaterThan(varFrom, eps))
                return [NumOps.Zero];
            return [NumOps.Divide(cov[parents[0], target], varFrom)];
        }

        // Extract Cov(parents, parents) sub-matrix and Cov(parents, target) vector
        var covPP = new Matrix<T>(k, k);
        var covPT = new T[k];
        for (int i = 0; i < k; i++)
        {
            covPT[i] = cov[parents[i], target];
            for (int j = 0; j < k; j++)
            {
                covPP[i, j] = cov[parents[i], parents[j]];
            }
        }

        // Add small regularization to diagonal for numerical stability
        for (int i = 0; i < k; i++)
        {
            covPP[i, i] = NumOps.Add(covPP[i, i], eps);
        }

        // Solve covPP * beta = covPT using Cholesky-like forward/back substitution
        // (Gaussian elimination with partial pivoting for small k)
        var augmented = new double[k, k + 1];
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < k; j++)
            {
                augmented[i, j] = NumOps.ToDouble(covPP[i, j]);
            }
            augmented[i, k] = NumOps.ToDouble(covPT[i]);
        }

        // Gaussian elimination with partial pivoting
        for (int col = 0; col < k; col++)
        {
            // Find pivot
            int pivotRow = col;
            double pivotVal = Math.Abs(augmented[col, col]);
            for (int row = col + 1; row < k; row++)
            {
                double val = Math.Abs(augmented[row, col]);
                if (val > pivotVal)
                {
                    pivotVal = val;
                    pivotRow = row;
                }
            }

            // Swap rows
            if (pivotRow != col)
            {
                for (int j = 0; j <= k; j++)
                {
                    (augmented[col, j], augmented[pivotRow, j]) = (augmented[pivotRow, j], augmented[col, j]);
                }
            }

            // Singular or near-singular: return zeros
            if (Math.Abs(augmented[col, col]) < PivotTolerance)
            {
                return new T[k];
            }

            // Eliminate below
            for (int row = col + 1; row < k; row++)
            {
                double factor = augmented[row, col] / augmented[col, col];
                for (int j = col; j <= k; j++)
                {
                    augmented[row, j] -= factor * augmented[col, j];
                }
            }
        }

        // Back substitution
        var beta = new double[k];
        for (int i = k - 1; i >= 0; i--)
        {
            double sum = augmented[i, k];
            for (int j = i + 1; j < k; j++)
            {
                sum -= augmented[i, j] * beta[j];
            }
            beta[i] = sum / augmented[i, i];
        }

        var result = new T[k];
        for (int i = 0; i < k; i++)
        {
            result[i] = NumOps.FromDouble(beta[i]);
        }
        return result;
    }
}
