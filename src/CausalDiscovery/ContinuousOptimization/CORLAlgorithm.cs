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
[ModelPaper("Ordering-Based Causal Discovery with Reinforcement Learning", "https://doi.org/10.24963/ijcai.2021/491", Year = 2021, Authors = "Xiaoqiang Wang, Yali Du, Shengyu Zhu, Liangjun Ke, Zhitang Chen, Jianye Hao, Jun Wang")]
public class CORLAlgorithm<T> : ContinuousOptimizationBase<T>
{
    private readonly double _learningRate;
    private readonly int _numEpisodes;
    private readonly int _maxParents;

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
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        // Compute sample covariance using base class utility
        var S = ComputeCovarianceMatrix(data);

        // Initialize position scores: scores[i,j] = preference for variable i at position j
        var scores = new Matrix<T>(d, d);
        var rng = new Random(42);

        int[] bestOrdering = Enumerable.Range(0, d).ToArray();
        double bestReward = double.NegativeInfinity;
        double baselineReward = 0;
        int validEpisodes = 0;

        for (int episode = 0; episode < _numEpisodes; episode++)
        {
            // Sample an ordering using softmax probabilities
            var ordering = SampleOrdering(scores, d, rng);

            // Compute the DAG from this ordering and get BIC reward
            double reward = ComputeOrderingReward(data, ordering, d, n, S);

            // Update baseline (running average)
            validEpisodes++;
            baselineReward += (reward - baselineReward) / validEpisodes;

            // Track best
            if (reward > bestReward)
            {
                bestReward = reward;
                bestOrdering = (int[])ordering.Clone();
            }

            // REINFORCE update: increase probability of positions that led to good reward
            T advantage = NumOps.FromDouble(reward - baselineReward);
            T lr = NumOps.FromDouble(_learningRate);
            T update = NumOps.Multiply(lr, advantage);
            for (int pos = 0; pos < d; pos++)
            {
                int v = ordering[pos];
                scores[v, pos] = NumOps.Add(scores[v, pos], update);
            }
        }

        // Build the DAG from the best ordering
        return BuildDAGFromOrdering(data, bestOrdering, d, n, S);
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
            var logits = new double[available.Count];
            double maxLogit = double.NegativeInfinity;
            for (int i = 0; i < available.Count; i++)
            {
                logits[i] = NumOps.ToDouble(scores[available[i], pos]);
                maxLogit = Math.Max(maxLogit, logits[i]);
            }

            double sumExp = 0;
            for (int i = 0; i < available.Count; i++)
            {
                logits[i] = Math.Exp(logits[i] - maxLogit);
                sumExp += logits[i];
            }

            // Sample from softmax
            double u = rng.NextDouble() * sumExp;
            double cumSum = 0;
            int chosen = 0;
            for (int i = 0; i < available.Count; i++)
            {
                cumSum += logits[i];
                if (cumSum >= u)
                {
                    chosen = i;
                    break;
                }
            }

            ordering[pos] = available[chosen];
            available.RemoveAt(chosen);
        }

        return ordering;
    }

    /// <summary>
    /// Computes the total BIC reward for a given ordering.
    /// </summary>
    private double ComputeOrderingReward(Matrix<T> data, int[] ordering, int d, int n, Matrix<T> S)
    {
        double totalBIC = 0;

        for (int idx = 0; idx < d; idx++)
        {
            int target = ordering[idx];

            // Find best parents from predecessors based on correlation
            var parents = new List<int>();
            for (int predIdx = 0; predIdx < idx && parents.Count < _maxParents; predIdx++)
            {
                int candidate = ordering[predIdx];
                double stt = NumOps.ToDouble(S[target, target]);
                double scc = NumOps.ToDouble(S[candidate, candidate]);
                double stc = Math.Abs(NumOps.ToDouble(S[target, candidate]));
                double corr = stc / Math.Sqrt((stt + 1e-10) * (scc + 1e-10));

                if (corr > 0.1)
                    parents.Add(candidate);
            }

            // Compute RSS with these parents via OLS
            double rss = ComputeRSS(data, target, parents, n);
            double bic = -n * Math.Log(rss / n + 1e-10) - (parents.Count + 1) * Math.Log(n);
            totalBIC += bic;
        }

        return totalBIC;
    }

    /// <summary>
    /// Computes residual sum of squares for target given parents via OLS.
    /// </summary>
    private double ComputeRSS(Matrix<T> data, int target, List<int> parents, int n)
    {
        T nT = NumOps.FromDouble(n);

        // Compute target mean
        T meanT = NumOps.Zero;
        for (int i = 0; i < n; i++)
            meanT = NumOps.Add(meanT, data[i, target]);
        meanT = NumOps.Divide(meanT, nT);

        if (parents.Count == 0)
        {
            // No parents: RSS = sum of (x - mean)^2
            T rss = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T e = NumOps.Subtract(data[i, target], meanT);
                rss = NumOps.Add(rss, NumOps.Multiply(e, e));
            }
            return NumOps.ToDouble(rss);
        }

        if (parents.Count == 1)
        {
            // Single parent: simple linear regression
            T meanP = NumOps.Zero;
            for (int i = 0; i < n; i++)
                meanP = NumOps.Add(meanP, data[i, parents[0]]);
            meanP = NumOps.Divide(meanP, nT);

            T covTP = NumOps.Zero;
            T varP = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T dp = NumOps.Subtract(data[i, parents[0]], meanP);
                T dt = NumOps.Subtract(data[i, target], meanT);
                covTP = NumOps.Add(covTP, NumOps.Multiply(dp, dt));
                varP = NumOps.Add(varP, NumOps.Multiply(dp, dp));
            }

            double varPd = NumOps.ToDouble(varP);
            double beta = varPd > 1e-10 ? NumOps.ToDouble(covTP) / varPd : 0;
            T betaT = NumOps.FromDouble(beta);

            T rss = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T dp = NumOps.Subtract(data[i, parents[0]], meanP);
                T pred = NumOps.Add(meanT, NumOps.Multiply(betaT, dp));
                T e = NumOps.Subtract(data[i, target], pred);
                rss = NumOps.Add(rss, NumOps.Multiply(e, e));
            }
            return NumOps.ToDouble(rss);
        }

        // Multi-parent: compute means and use sequential regression
        var parentMeans = new T[parents.Count];
        for (int j = 0; j < parents.Count; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
                sum = NumOps.Add(sum, data[i, parents[j]]);
            parentMeans[j] = NumOps.Divide(sum, nT);
        }

        // Multi-parent OLS via sequential residualization
        var residuals = new T[n];
        for (int i = 0; i < n; i++)
            residuals[i] = NumOps.Subtract(data[i, target], meanT);

        for (int j = 0; j < parents.Count; j++)
        {
            T covRP = NumOps.Zero;
            T varPj = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T dp = NumOps.Subtract(data[i, parents[j]], parentMeans[j]);
                covRP = NumOps.Add(covRP, NumOps.Multiply(residuals[i], dp));
                varPj = NumOps.Add(varPj, NumOps.Multiply(dp, dp));
            }

            double varPjd = NumOps.ToDouble(varPj);
            if (varPjd > 1e-10)
            {
                T beta = NumOps.Divide(covRP, varPj);
                for (int i = 0; i < n; i++)
                {
                    T dp = NumOps.Subtract(data[i, parents[j]], parentMeans[j]);
                    residuals[i] = NumOps.Subtract(residuals[i], NumOps.Multiply(beta, dp));
                }
            }
        }

        T totalRss = NumOps.Zero;
        for (int i = 0; i < n; i++)
            totalRss = NumOps.Add(totalRss, NumOps.Multiply(residuals[i], residuals[i]));
        return NumOps.ToDouble(totalRss);
    }

    /// <summary>
    /// Builds the final DAG from the best ordering using greedy parent selection.
    /// </summary>
    private Matrix<T> BuildDAGFromOrdering(Matrix<T> data, int[] ordering, int d, int n, Matrix<T> S)
    {
        var W = new Matrix<T>(d, d);
        T nT = NumOps.FromDouble(n);

        for (int idx = 1; idx < d; idx++)
        {
            int target = ordering[idx];

            for (int predIdx = 0; predIdx < idx; predIdx++)
            {
                int parent = ordering[predIdx];
                double stt = NumOps.ToDouble(S[target, target]);
                double spp = NumOps.ToDouble(S[parent, parent]);
                double stp = Math.Abs(NumOps.ToDouble(S[target, parent]));
                double corr = stp / Math.Sqrt((stt + 1e-10) * (spp + 1e-10));

                if (corr > WThreshold)
                {
                    // Compute OLS weight using generic operations
                    T meanP = NumOps.Zero;
                    T meanT = NumOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        meanP = NumOps.Add(meanP, data[i, parent]);
                        meanT = NumOps.Add(meanT, data[i, target]);
                    }
                    meanP = NumOps.Divide(meanP, nT);
                    meanT = NumOps.Divide(meanT, nT);

                    T covPT = NumOps.Zero;
                    T varP = NumOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        T dp = NumOps.Subtract(data[i, parent], meanP);
                        covPT = NumOps.Add(covPT, NumOps.Multiply(dp,
                                NumOps.Subtract(data[i, target], meanT)));
                        varP = NumOps.Add(varP, NumOps.Multiply(dp, dp));
                    }

                    double varPd = NumOps.ToDouble(varP);
                    if (varPd > 1e-10)
                    {
                        T weight = NumOps.Divide(covPT, varP);
                        if (Math.Abs(NumOps.ToDouble(weight)) >= WThreshold)
                            W[parent, target] = weight;
                    }
                }
            }
        }

        return W;
    }
}
