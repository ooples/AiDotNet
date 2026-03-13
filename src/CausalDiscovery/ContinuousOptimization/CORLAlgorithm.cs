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

        // Initialize position scores: scores[i,j] = preference for variable i at position j
        var scores = new Matrix<T>(d, d);
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);

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

            // REINFORCE gradient matching the masked sampling in SampleOrdering:
            // At each position, only available (not-yet-chosen) variables participate in softmax.
            T advantage = NumOps.FromDouble(reward - baselineReward);
            T lr = NumOps.FromDouble(_learningRate);
            var availableSet = new HashSet<int>(Enumerable.Range(0, d));
            for (int pos = 0; pos < d; pos++)
            {
                int chosenVar = ordering[pos];
                var available = availableSet.ToList();

                // Compute softmax only over available variables (matching SampleOrdering)
                double maxLogit = double.NegativeInfinity;
                foreach (int v in available)
                    maxLogit = Math.Max(maxLogit, NumOps.ToDouble(scores[v, pos]));
                double sumExp = 0;
                var probs = new Dictionary<int, double>();
                foreach (int v in available)
                {
                    double p = Math.Exp(NumOps.ToDouble(scores[v, pos]) - maxLogit);
                    probs[v] = p;
                    sumExp += p;
                }
                foreach (int v in available)
                    probs[v] /= sumExp;

                // REINFORCE: update = lr * advantage * grad_log_pi
                // grad_log_pi[chosen] = 1 - p(chosen), grad_log_pi[other] = -p(other)
                foreach (int v in available)
                {
                    double gradLogPi = (v == chosenVar) ? (1.0 - probs[v]) : (-probs[v]);
                    T update = NumOps.Multiply(lr, NumOps.Multiply(advantage, NumOps.FromDouble(gradLogPi)));
                    scores[v, pos] = NumOps.Add(scores[v, pos], update);
                }

                availableSet.Remove(chosenVar);
            }
        }

        // Build the DAG from the best ordering using the SAME criteria used in scoring
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
    /// Selects parents for a target from predecessors in the ordering using BIC-based forward selection.
    /// </summary>
    private List<int> SelectParentsBIC(Matrix<T> data, int target, int[] ordering, int idx, int n)
    {
        var parents = new List<int>();
        var candidates = new List<int>();
        for (int predIdx = 0; predIdx < idx; predIdx++)
            candidates.Add(ordering[predIdx]);

        double bestBIC = ComputeBICForParents(data, target, parents, n);

        // Greedy forward selection up to _maxParents
        while (parents.Count < _maxParents && candidates.Count > 0)
        {
            int bestCandidate = -1;
            double bestCandidateBIC = bestBIC;

            foreach (int c in candidates)
            {
                var trial = new List<int>(parents) { c };
                double trialBIC = ComputeBICForParents(data, target, trial, n);
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
    private double ComputeBICForParents(Matrix<T> data, int target, List<int> parents, int n)
    {
        double rss = ComputeRSS(data, target, parents, n);
        int k = parents.Count + 1; // +1 for intercept
        return -n * Math.Log(rss / n + 1e-10) - k * Math.Log(n);
    }

    /// <summary>
    /// Computes the total BIC reward for a given ordering.
    /// Uses the same parent selection as BuildDAGFromOrdering.
    /// </summary>
    private double ComputeOrderingReward(Matrix<T> data, int[] ordering, int d, int n, Matrix<T> S)
    {
        double totalBIC = 0;
        for (int idx = 0; idx < d; idx++)
        {
            int target = ordering[idx];
            var parents = SelectParentsBIC(data, target, ordering, idx, n);
            double rss = ComputeRSS(data, target, parents, n);
            double bic = -n * Math.Log(rss / n + 1e-10) - (parents.Count + 1) * Math.Log(n);
            totalBIC += bic;
        }
        return totalBIC;
    }

    /// <summary>
    /// Computes residual sum of squares for target given parents via multivariate OLS.
    /// </summary>
    private double ComputeRSS(Matrix<T> data, int target, List<int> parents, int n)
    {
        // Compute target mean
        double meanT = 0;
        for (int i = 0; i < n; i++)
            meanT += NumOps.ToDouble(data[i, target]);
        meanT /= n;

        if (parents.Count == 0)
        {
            double rss = 0;
            for (int i = 0; i < n; i++)
            {
                double e = NumOps.ToDouble(data[i, target]) - meanT;
                rss += e * e;
            }
            return rss;
        }

        int p = parents.Count;

        // Compute parent means
        var parentMeans = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                parentMeans[j] += NumOps.ToDouble(data[i, parents[j]]);
            parentMeans[j] /= n;
        }

        // Build normal equations X'X and X'y for multivariate OLS
        var XtX = new double[p, p];
        var Xty = new double[p];

        for (int i = 0; i < n; i++)
        {
            double dy = NumOps.ToDouble(data[i, target]) - meanT;
            var dx = new double[p];
            for (int j = 0; j < p; j++)
                dx[j] = NumOps.ToDouble(data[i, parents[j]]) - parentMeans[j];

            for (int a = 0; a < p; a++)
            {
                Xty[a] += dx[a] * dy;
                for (int b = a; b < p; b++)
                    XtX[a, b] += dx[a] * dx[b];
            }
        }

        // Symmetrize and add ridge
        for (int a = 0; a < p; a++)
        {
            XtX[a, a] += 1e-10;
            for (int b = a + 1; b < p; b++)
                XtX[b, a] = XtX[a, b];
        }

        // Solve for beta via Gaussian elimination
        var beta = SolveSmallSystem(XtX, Xty, p);

        // Compute RSS
        double totalRss = 0;
        for (int i = 0; i < n; i++)
        {
            double pred = meanT;
            for (int j = 0; j < p; j++)
                pred += beta[j] * (NumOps.ToDouble(data[i, parents[j]]) - parentMeans[j]);
            double e = NumOps.ToDouble(data[i, target]) - pred;
            totalRss += e * e;
        }

        return totalRss;
    }

    /// <summary>
    /// Builds the final DAG from the best ordering using the same parent selection as scoring.
    /// </summary>
    private Matrix<T> BuildDAGFromOrdering(Matrix<T> data, int[] ordering, int d, int n, Matrix<T> S)
    {
        var W = new Matrix<T>(d, d);

        for (int idx = 1; idx < d; idx++)
        {
            int target = ordering[idx];
            var parents = SelectParentsBIC(data, target, ordering, idx, n);

            if (parents.Count == 0) continue;

            int p = parents.Count;

            // Compute multivariate OLS coefficients
            var parentMeans = new double[p];
            double meanT = 0;
            for (int i = 0; i < n; i++)
            {
                meanT += NumOps.ToDouble(data[i, target]);
                for (int j = 0; j < p; j++)
                    parentMeans[j] += NumOps.ToDouble(data[i, parents[j]]);
            }
            meanT /= n;
            for (int j = 0; j < p; j++) parentMeans[j] /= n;

            var XtX = new double[p, p];
            var Xty = new double[p];
            for (int i = 0; i < n; i++)
            {
                double dy = NumOps.ToDouble(data[i, target]) - meanT;
                var dx = new double[p];
                for (int j = 0; j < p; j++)
                    dx[j] = NumOps.ToDouble(data[i, parents[j]]) - parentMeans[j];
                for (int a = 0; a < p; a++)
                {
                    Xty[a] += dx[a] * dy;
                    for (int b = a; b < p; b++)
                        XtX[a, b] += dx[a] * dx[b];
                }
            }
            for (int a = 0; a < p; a++)
            {
                XtX[a, a] += 1e-10;
                for (int b = a + 1; b < p; b++)
                    XtX[b, a] = XtX[a, b];
            }

            var beta = SolveSmallSystem(XtX, Xty, p);
            for (int j = 0; j < p; j++)
            {
                if (Math.Abs(beta[j]) >= WThreshold)
                    W[parents[j], target] = NumOps.FromDouble(beta[j]);
            }
        }

        return W;
    }

    private static double[] SolveSmallSystem(double[,] A, double[] b, int p)
    {
        var aug = new double[p, p + 1];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++) aug[i, j] = A[i, j];
            aug[i, p] = b[i];
        }
        for (int col = 0; col < p; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < p; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col])) maxRow = row;
            if (maxRow != col)
                for (int j = col; j <= p; j++)
                    (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);
            double pivot = aug[col, col];
            if (Math.Abs(pivot) < 1e-15) continue;
            for (int row = col + 1; row < p; row++)
            {
                double factor = aug[row, col] / pivot;
                for (int j = col; j <= p; j++) aug[row, j] -= factor * aug[col, j];
            }
        }
        var x = new double[p];
        for (int row = p - 1; row >= 0; row--)
        {
            double sum = aug[row, p];
            for (int j = row + 1; j < p; j++) sum -= aug[row, j] * x[j];
            double diag = aug[row, row];
            x[row] = Math.Abs(diag) > 1e-15 ? sum / diag : 0;
        }
        return x;
    }
}
