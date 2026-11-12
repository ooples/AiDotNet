using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// CORL — Causal Ordering via Reinforcement Learning.
/// </summary>
/// <remarks>
/// <para>
/// CORL learns a causal ordering of variables using a policy gradient approach inspired
/// by reinforcement learning. The algorithm maintains a scoring function for each position
/// in the ordering and uses policy gradient updates to improve the ordering based on
/// the BIC score of the resulting DAG. Once the ordering is determined, edge weights
/// are learned via OLS regression.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize a score matrix S[i,j] = probability of variable i at position j</item>
/// <item>Sample an ordering from the score matrix (using softmax)</item>
/// <item>Given the ordering, compute the optimal DAG via greedy parent selection with BIC</item>
/// <item>Use the BIC score as a reward signal</item>
/// <item>Update S using REINFORCE-style policy gradient: S += lr * (R - baseline) * grad_log_pi</item>
/// <item>Repeat for multiple episodes, tracking the best ordering found</item>
/// <item>Return the DAG from the best ordering</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of directly optimizing a weight matrix (like NOTEARS),
/// CORL learns the ORDER in which variables cause each other. It uses a technique from
/// AI game-playing (reinforcement learning) where the algorithm tries different orderings
/// and gets "rewarded" for finding ones that explain the data well. Once you know the
/// order (e.g., X causes Y which causes Z), finding the exact relationships is easy.
/// </para>
/// <para>
/// Reference: Wang et al. (2021), "Ordering-Based Causal Discovery with Reinforcement
/// Learning", IJCAI.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Optimization)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Ordering-Based Causal Discovery with Reinforcement Learning", "https://doi.org/10.24963/ijcai.2021/491", Year = 2021, Authors = "Xiaoqiang Wang, Yali Du, Shengyu Zhu, Liangjun Ke, Zhitang Chen, Jianye Hao, Jun Wang")]
public class CORLAlgorithm<T> : ContinuousOptimizationBase<T>
{
    private readonly double _learningRate;
    private readonly int _numEpisodes;
    private readonly int _maxParents;
    private readonly int _seed;

    /// <summary>
    /// Seed used when the caller does not supply one, so that a run is reproducible by default.
    /// </summary>
    /// <remarks>
    /// Previously this fell back to an unseeded secure RNG, which made the model nondeterministic
    /// and its generated model-family tests flaky: the identical test binary produced 13 failures
    /// in one run and 8 in the next over the same 42 tests, which makes regression detection on
    /// this path impossible. Callers still override via the options' Seed property. Matches the
    /// fixed-seed convention already used by DAGMANonlinear in this module.
    /// </remarks>
    private const int DefaultRandomSeed = 42;
    /// <inheritdoc/>
    public override string Name => "CORL";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes CORL with optional configuration.
    /// </summary>
    public CORLAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyOptions(options);
        _learningRate = options?.LearningRate ?? 0.01;
        _numEpisodes = options?.MaxIterations ?? 100;
        _maxParents = options?.MaxParents ?? 5;
        _seed = options?.Seed ?? DefaultRandomSeed;
        if (_learningRate <= 0 || double.IsNaN(_learningRate) || double.IsInfinity(_learningRate))
            throw new ArgumentException("LearningRate must be positive and finite.");
        if (_numEpisodes < 1)
            throw new ArgumentException("MaxIterations must be at least 1.");
        if (_maxParents < 0)
            throw new ArgumentException("MaxParents must be non-negative.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (d < 2)
            throw new ArgumentException($"CORL requires at least 2 variables, got {d}.");
        int minSamples = Math.Max(_maxParents + 3, d + 1);
        if (n < minSamples)
            throw new ArgumentException($"CORL requires at least {minSamples} samples (max parents={_maxParents}, variables={d}), got {n}.");

        // Compute sample covariance using base class utility
        var S = ComputeCovarianceMatrix(data);

        // One tensor copy of the data shared by every candidate regression.
        var source = Tensor<T>.FromMatrix(data);

        // Initialize position scores: scores[i,j] = preference for variable i at position j
        var scores = new Matrix<T>(d, d);
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(_seed);

        int[] bestOrdering = Enumerable.Range(0, d).ToArray();
        double bestReward = double.NegativeInfinity;
        double baselineReward = 0;
        int validEpisodes = 0;

        for (int episode = 0; episode < _numEpisodes; episode++)
        {
            // Sample an ordering using softmax probabilities
            var ordering = SampleOrdering(scores, d, rng);

            // Compute the DAG from this ordering and get BIC reward
            // Use the same maxParents and threshold as BuildDAGFromOrdering
            double reward = ComputeOrderingReward(source, ordering, d, n, S);

            // Update baseline (running average)
            validEpisodes++;
            baselineReward += (reward - baselineReward) / validEpisodes;

            // Track best
            if (reward > bestReward)
            {
                bestReward = reward;
                bestOrdering = (int[])ordering.Clone();
            }

            // REINFORCE gradient matching the masked sampling in SampleOrdering:
            // At each position, only available (not-yet-chosen) variables participate in softmax.
            ApplyReinforceUpdate(scores, ordering, d, reward - baselineReward);
        }

        // Build the DAG from the best ordering using the SAME criteria used in scoring
        return BuildDAGFromOrdering(source, bestOrdering, d, n, S);
    }

    /// <summary>
    /// Samples an ordering using softmax over position scores.
    /// </summary>
    private int[] SampleOrdering(Matrix<T> scores, int d, Random rng)
    {
        var ordering = new int[d];
        var available = new List<int>(Enumerable.Range(0, d));

        for (int pos = 0; pos < d; pos++)
        {
            // Softmax over available variables for this position
            var logits = Engine.TensorIndexSelect(Engine.GetColumn(scores, pos).AsTensor(),
                new Tensor<int>(available.ToArray(), new[] { available.Count }), 0);
            int chosen = SampleSoftmaxIndex(logits, rng);

            ordering[pos] = available[chosen];
            available.RemoveAt(chosen);
        }

        return ordering;
    }

    /// <summary>
    /// Draws an index with probability proportional to exp(logit - max logit): the first index whose
    /// cumulative weight reaches u * total, u uniform in [0, 1); index 0 when none does.
    /// </summary>
    private int SampleSoftmaxIndex(Tensor<T> logits, Random rng)
    {
        int count = logits.Length;
        T maxLogit = Engine.TensorMaxValue(logits);
        var weights = Engine.TensorExp(Engine.TensorSubtract(logits, Engine.TensorFullLike(logits, maxLogit)));
        var cumulative = Engine.TensorCumSum(weights, 0);
        T sumExp = Engine.TensorSum(Engine.TensorSlice(cumulative, new[] { count - 1 }, new[] { 1 }));

        // Sample from softmax
        T u = NumOps.Multiply(NumOps.FromDouble(rng.NextDouble()), sumExp);
        int chosen = Engine.TensorCountNonzero(Engine.TensorLessThan(cumulative, Engine.TensorFullLike(cumulative, u)));
        return chosen < count ? chosen : 0;
    }

    /// <summary>
    /// REINFORCE update of the position scores for one sampled ordering:
    /// scores[v, pos] += lr * advantage * grad_log_pi(v) over the variables still available at pos,
    /// with grad_log_pi[chosen] = 1 - p(chosen) and grad_log_pi[other] = -p(other).
    /// </summary>
    private void ApplyReinforceUpdate(Matrix<T> scores, int[] ordering, int d, double advantageValue)
    {
        T advantage = NumOps.FromDouble(advantageValue);
        T lr = NumOps.FromDouble(_learningRate);
        var availableSet = new HashSet<int>(Enumerable.Range(0, d));
        for (int pos = 0; pos < d; pos++)
        {
            int chosenVar = ordering[pos];
            var available = availableSet.OrderBy(v => v).ToArray();
            var availableIndex = new Tensor<int>(available, new[] { available.Length });
            var column = Engine.GetColumn(scores, pos).AsTensor();

            // Compute softmax only over available variables (matching SampleOrdering)
            var logits = Engine.TensorIndexSelect(column, availableIndex, 0);
            T maxLogit = Engine.TensorMaxValue(logits);
            var expLogits = Engine.TensorExp(Engine.TensorSubtract(logits, Engine.TensorFullLike(logits, maxLogit)));
            var probs = Engine.TensorDivideScalar(expLogits, Engine.TensorSum(expLogits));

            var isChosen = new bool[available.Length];
            isChosen[Array.IndexOf(available, chosenVar)] = true;
            var gradLogPi = Engine.TensorSubtract(
                Engine.TensorWhere(isChosen, Engine.TensorOnesLike(probs), Engine.TensorZerosLike(probs)), probs);
            var update = Engine.TensorMultiplyScalar(Engine.TensorMultiplyScalar(gradLogPi, advantage), lr);

            var updated = Engine.TensorIndexAdd(column, 0, availableIndex, update);
            Engine.SetColumn(scores, pos, updated.ToVector());

            availableSet.Remove(chosenVar);
        }
    }

    /// <summary>
    /// Selects parents for a target from predecessors in the ordering using BIC-based forward selection.
    /// </summary>
    private List<int> SelectParentsBIC(Tensor<T> source, int target, int[] ordering, int idx, int n)
    {
        var parents = new List<int>();
        var candidates = new List<int>();
        for (int predIdx = 0; predIdx < idx; predIdx++)
            candidates.Add(ordering[predIdx]);

        double bestBIC = ComputeBICForParents(source, target, parents, n);

        // Greedy forward selection up to _maxParents
        while (parents.Count < _maxParents && candidates.Count > 0)
        {
            int bestCandidate = -1;
            double bestCandidateBIC = bestBIC;

            foreach (int c in candidates)
            {
                var trial = new List<int>(parents) { c };
                double trialBIC = ComputeBICForParents(source, target, trial, n);
                if (trialBIC > bestCandidateBIC)
                {
                    bestCandidateBIC = trialBIC;
                    bestCandidate = c;
                }
            }

            if (bestCandidate < 0) break;
            parents.Add(bestCandidate);
            candidates.Remove(bestCandidate);
            bestBIC = bestCandidateBIC;
        }

        return parents;
    }

    /// <summary>
    /// Computes BIC score for a target with given parents using multivariate OLS.
    /// </summary>
    private double ComputeBICForParents(Tensor<T> source, int target, List<int> parents, int n)
    {
        double rss = ComputeRSS(source, target, parents, n);
        int k = parents.Count + 1; // +1 for intercept
        return -n * Math.Log(rss / n + 1e-10) - k * Math.Log(n);
    }

    /// <summary>
    /// Computes the total BIC reward for a given ordering.
    /// Uses the same parent selection as BuildDAGFromOrdering.
    /// </summary>
    private double ComputeOrderingReward(Tensor<T> source, int[] ordering, int d, int n, Matrix<T> S)
    {
        double totalBIC = 0;
        for (int idx = 0; idx < d; idx++)
        {
            int target = ordering[idx];
            var parents = SelectParentsBIC(source, target, ordering, idx, n);
            double rss = ComputeRSS(source, target, parents, n);
            double bic = -n * Math.Log(rss / n + 1e-10) - (parents.Count + 1) * Math.Log(n);
            totalBIC += bic;
        }
        return totalBIC;
    }

    /// <summary>
    /// Computes residual sum of squares for target given parents via multivariate OLS.
    /// </summary>
    private double ComputeRSS(Tensor<T> source, int target, List<int> parents, int n)
    {
        if (parents.Count == 0)
        {
            // Residuals around the target mean
            var y = Engine.TensorSliceAxis(source, 1, target);
            var centered = Engine.TensorSubtract(y, Engine.TensorFullLike(y, Engine.TensorMean(y)));
            return NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(centered)));
        }

        // Multivariate OLS on the centered parents, solved by Gaussian elimination
        var (_, residual) = FitCenteredRegression(source, target, parents.ToArray());
        return NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(residual)));
    }

    /// <summary>
    /// Builds the final DAG from the best ordering using the same parent selection as scoring.
    /// </summary>
    private Matrix<T> BuildDAGFromOrdering(Tensor<T> source, int[] ordering, int d, int n, Matrix<T> S)
    {
        var W = new Matrix<T>(d, d);
        T threshold = NumOps.FromDouble(WThreshold);

        for (int idx = 1; idx < d; idx++)
        {
            int target = ordering[idx];
            var parents = SelectParentsBIC(source, target, ordering, idx, n);

            if (parents.Count == 0) continue;

            // Multivariate OLS coefficients; those reaching WThreshold become the edge weights into target.
            var (beta, _) = FitCenteredRegression(source, target, parents.ToArray());
            var kept = Engine.TensorWhere(
                Engine.TensorGreaterEqual(Engine.TensorAbs(beta), Engine.TensorFullLike(beta, threshold)),
                beta, Engine.TensorZerosLike(beta));
            var column = Engine.TensorScatter(new Tensor<T>(new[] { d }),
                new Tensor<int>(parents.ToArray(), new[] { parents.Count }), kept, 0);
            Engine.SetColumn(W, target, column.ToVector());
        }

        return W;
    }

    // SolveSmallSystem is inherited from CausalDiscoveryBase<T>
}
