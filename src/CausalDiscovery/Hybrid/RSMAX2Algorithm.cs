using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Hybrid;

/// <summary>
/// RSMAX2 — Restricted Maximization, a hybrid constraint-based + score-based algorithm.
/// </summary>
/// <remarks>
/// <para>
/// RSMAX2 (Restricted Structural Maximization, 2-phase) is a general framework for
/// hybrid causal discovery. Phase 1 uses conditional independence tests to learn each
/// variable's candidate parent/child set (restricting the search space). Phase 2 uses
/// greedy hill climbing with BIC scoring restricted to the candidate sets.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item><b>Restrict phase:</b> For each variable, find candidate parents/children
///   using conditional independence tests (MMPC-like forward-backward selection)</item>
/// <item><b>Maximize phase:</b> Starting from an empty graph, greedily add edges
///   between variables and their candidates that improve BIC score</item>
/// <item>Enforce acyclicity by rejecting edge additions that create cycles</item>
/// <item>Apply backward deletion to remove edges that no longer improve score</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> RSMAX2 is a general framework for combining any "candidate finder"
/// with any "best structure finder." It first quickly identifies which variables MIGHT be
/// connected, then carefully selects the best connections from those candidates.
/// </para>
/// <para>
/// Reference: Scutari (2010), "Learning Bayesian Networks with the bnlearn R Package", JSS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Learning Bayesian Networks with the bnlearn R Package", "https://www.jstatsoft.org/article/view/v035i03", Year = 2010, Authors = "Marco Scutari")]
public class RSMAX2Algorithm<T> : HybridBase<T>
{
    private readonly int _maxConditioningSetSize;

    /// <inheritdoc/>
    public override string Name => "RSMAX2";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes RSMAX2 with optional configuration.
    /// </summary>
    public RSMAX2Algorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyHybridOptions(options);
        _maxConditioningSetSize = options?.MaxConditioningSetSize ?? 3;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        // Phase 1: Restrict — find candidate parent sets using CI tests
        var candidates = FindCandidateSets(data, d);

        // Phase 2: Maximize — greedy hill climbing restricted to candidates
        var parentSets = new HashSet<int>[d];
        for (int i = 0; i < d; i++)
            parentSets[i] = [];

        // Forward pass: greedily add best edge
        bool improved = true;
        while (improved)
        {
            improved = false;
            int bestTarget = -1, bestParent = -1;
            double bestDelta = 0;

            for (int target = 0; target < d; target++)
            {
                if (parentSets[target].Count >= MaxParents) continue;

                foreach (int candidate in candidates[target])
                {
                    if (parentSets[target].Contains(candidate)) continue;
                    if (WouldCreateCycle(parentSets, candidate, target)) continue;

                    var oldScore = ComputeBIC(data, target, parentSets[target].ToList());
                    var newParents = new List<int>(parentSets[target]) { candidate };
                    var newScore = ComputeBIC(data, target, newParents);

                    double delta = oldScore - newScore; // Lower BIC is better in this formulation
                    if (delta > bestDelta)
                    {
                        bestDelta = delta;
                        bestTarget = target;
                        bestParent = candidate;
                    }
                }
            }

            if (bestTarget >= 0 && bestDelta > 0)
            {
                parentSets[bestTarget].Add(bestParent);
                improved = true;
            }
        }

        // Backward pass: try removing each edge
        bool removedAny = true;
        while (removedAny)
        {
            removedAny = false;
            for (int target = 0; target < d; target++)
            {
                foreach (int parent in parentSets[target].ToList())
                {
                    var currentScore = ComputeBIC(data, target, parentSets[target].ToList());
                    var reducedParents = parentSets[target].Where(p => p != parent).ToList();
                    var reducedScore = ComputeBIC(data, target, reducedParents);

                    if (reducedScore <= currentScore)
                    {
                        parentSets[target].Remove(parent);
                        removedAny = true;
                    }
                }
            }
        }

        // Build adjacency matrix with OLS edge weights
        return BuildWeightedMatrix(data, parentSets, d, n);
    }

    /// <summary>
    /// Phase 1: Find candidate parent/child sets using conditional independence tests.
    /// </summary>
    private HashSet<int>[] FindCandidateSets(Matrix<T> data, int d)
    {
        var candidates = new HashSet<int>[d];
        for (int i = 0; i < d; i++)
            candidates[i] = [];

        for (int target = 0; target < d; target++)
        {
            // Forward: add variables with strongest marginal association
            for (int candidate = 0; candidate < d; candidate++)
            {
                if (candidate == target) continue;

                double corr = Math.Abs(ComputeCorrelation(data, target, candidate));
                int n = data.Rows;
                double z = Math.Sqrt(n - 3) * 0.5 * Math.Log((1 + corr) / (1 - corr + 1e-10));
                double pValue = 2 * (1 - NormalCDF(Math.Abs(z)));

                if (pValue < Alpha)
                {
                    candidates[target].Add(candidate);
                }
            }

            // Backward: remove false positives
            foreach (int member in candidates[target].ToList())
            {
                var rest = candidates[target].Where(v => v != member).Take(_maxConditioningSetSize).ToList();
                if (rest.Count > 0)
                {
                    double partialCorr = ComputePartialCorrelation(data, target, member, rest);
                    int n = data.Rows;
                    int dof = n - rest.Count - 3;
                    if (dof > 0)
                    {
                        double clamped = Math.Max(-0.999999, Math.Min(partialCorr, 0.999999));
                        double z = Math.Sqrt(dof) * 0.5 * Math.Log((1 + clamped) / (1 - clamped));
                        double pValue = 2 * (1 - NormalCDF(Math.Abs(z)));
                        if (pValue > Alpha)
                            candidates[target].Remove(member);
                    }
                }
            }
        }

        return candidates;
    }

    private double ComputePartialCorrelation(Matrix<T> data, int i, int j, List<int> condSet)
    {
        if (condSet.Count == 0)
            return ComputeCorrelation(data, i, j);

        // Residualize both i and j on condSet
        int n = data.Rows;
        var residI = Residualize(data, i, condSet, n);
        var residJ = Residualize(data, j, condSet, n);

        double meanI = residI.Average();
        double meanJ = residJ.Average();

        double sxy = 0, sxx = 0, syy = 0;
        for (int k = 0; k < n; k++)
        {
            double di = residI[k] - meanI;
            double dj = residJ[k] - meanJ;
            sxy += di * dj;
            sxx += di * di;
            syy += dj * dj;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    private double[] Residualize(Matrix<T> data, int target, List<int> predictors, int n)
    {
        var residuals = new double[n];
        var means = new double[predictors.Count + 1];

        // Compute means
        for (int i = 0; i < n; i++)
        {
            means[0] += NumOps.ToDouble(data[i, target]);
            for (int j = 0; j < predictors.Count; j++)
                means[j + 1] += NumOps.ToDouble(data[i, predictors[j]]);
        }
        for (int j = 0; j <= predictors.Count; j++) means[j] /= n;

        // Simple OLS via normal equations
        if (predictors.Count == 1)
        {
            double cov = 0, varX = 0;
            int pred = predictors[0];
            for (int i = 0; i < n; i++)
            {
                double dx = NumOps.ToDouble(data[i, pred]) - means[1];
                double dy = NumOps.ToDouble(data[i, target]) - means[0];
                cov += dx * dy;
                varX += dx * dx;
            }
            double beta = varX > 1e-10 ? cov / varX : 0;
            for (int i = 0; i < n; i++)
                residuals[i] = NumOps.ToDouble(data[i, target]) - means[0] -
                               beta * (NumOps.ToDouble(data[i, pred]) - means[1]);
        }
        else
        {
            // Multi-predictor OLS via normal equations
            int p = predictors.Count;
            var XtX = new double[p, p];
            var Xty = new double[p];
            for (int i = 0; i < n; i++)
            {
                double dy = NumOps.ToDouble(data[i, target]) - means[0];
                var dx = new double[p];
                for (int j = 0; j < p; j++)
                    dx[j] = NumOps.ToDouble(data[i, predictors[j]]) - means[j + 1];
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
            for (int i = 0; i < n; i++)
            {
                residuals[i] = NumOps.ToDouble(data[i, target]) - means[0];
                for (int j = 0; j < p; j++)
                    residuals[i] -= beta[j] * (NumOps.ToDouble(data[i, predictors[j]]) - means[j + 1]);
            }
        }

        return residuals;
    }

    private static bool WouldCreateCycle(HashSet<int>[] parents, int from, int to)
    {
        var visited = new HashSet<int>();
        var queue = new Queue<int>();
        queue.Enqueue(from);

        while (queue.Count > 0)
        {
            int current = queue.Dequeue();
            if (current == to) return true;
            if (!visited.Add(current)) continue;
            foreach (int p in parents[current])
                queue.Enqueue(p);
        }

        return false;
    }

    private Matrix<T> BuildWeightedMatrix(Matrix<T> data, HashSet<int>[] parentSets, int d, int n)
    {
        var W = new Matrix<T>(d, d);
        for (int child = 0; child < d; child++)
        {
            foreach (int parent in parentSets[child])
            {
                double meanP = 0, meanC = 0;
                for (int i = 0; i < n; i++)
                {
                    meanP += NumOps.ToDouble(data[i, parent]);
                    meanC += NumOps.ToDouble(data[i, child]);
                }
                meanP /= n; meanC /= n;

                double cov = 0, varP = 0;
                for (int i = 0; i < n; i++)
                {
                    double dp = NumOps.ToDouble(data[i, parent]) - meanP;
                    cov += dp * (NumOps.ToDouble(data[i, child]) - meanC);
                    varP += dp * dp;
                }

                W[parent, child] = NumOps.FromDouble(varP > 1e-10 ? cov / varP : 0);
            }
        }
        return W;
    }

    private static double NormalCDF(double x)
    {
        double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
        double a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x) / Math.Sqrt(2);
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return 0.5 * (1.0 + sign * y);
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
