using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Hybrid;

/// <summary>
/// H2PC — Hybrid HPC (Hybrid Parents and Children) algorithm.
/// </summary>
/// <remarks>
/// <para>
/// H2PC uses a two-phase approach where the first phase identifies candidate parents
/// and children for each variable using a combination of marginal and conditional
/// association tests (the HPC algorithm), and the second phase uses hill climbing with
/// BIC scoring restricted to the candidates. The HPC phase differs from MMPC by using
/// a recursive decomposition: first find spouses of target's neighbors to improve accuracy.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item><b>HPC phase (per variable):</b></item>
/// <item>  Forward: add variable with max min-association (like MMPC)</item>
/// <item>  Backward: remove false positives via CI tests</item>
/// <item>  Spouse discovery: for each neighbor n of target, check if any variable
///   is associated with target given n (spouse relationship)</item>
/// <item><b>Hill climbing phase:</b> greedy DAG construction using BIC within candidate sets</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> H2PC is a refined version of MMHC with a smarter first phase.
/// It finds candidate parents/children more accurately before running the scoring step,
/// leading to better results especially with smaller datasets.
/// </para>
/// <para>
/// Reference: Gasse et al. (2014), "A Hybrid Algorithm for Bayesian Network Structure
/// Learning with Application to Multi-Label Learning", Expert Systems with Applications.
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
[ModelPaper("A Hybrid Algorithm for Bayesian Network Structure Learning with Application to Multi-Label Learning", "https://doi.org/10.1016/j.eswa.2014.03.032", Year = 2014, Authors = "Maxime Gasse, Alex Aussem, Haytham Elghazel")]
public class H2PCAlgorithm<T> : HybridBase<T>
{
    private readonly int _maxConditioningSetSize;

    /// <inheritdoc/>
    public override string Name => "H2PC";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes H2PC with optional configuration.
    /// </summary>
    public H2PCAlgorithm(CausalDiscoveryOptions? options = null)
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

        // Phase 1: HPC — find parents/children and spouses for each variable
        var candidates = new HashSet<int>[d];
        for (int target = 0; target < d; target++)
        {
            candidates[target] = FindHPC(data, target, d, n);
        }

        // Phase 2: Hill climbing restricted to candidate sets
        var parentSets = new HashSet<int>[d];
        for (int i = 0; i < d; i++)
            parentSets[i] = [];

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
                    if (WouldCreateCycle(parentSets, candidate, target, d)) continue;

                    var oldScore = ComputeBIC(data, target, parentSets[target].ToList());
                    var newParents = new List<int>(parentSets[target]) { candidate };
                    var newScore = ComputeBIC(data, target, newParents);

                    double delta = oldScore - newScore;
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

        // Build weighted adjacency matrix
        var W = new Matrix<T>(d, d);
        for (int child = 0; child < d; child++)
        {
            foreach (int parent in parentSets[child])
            {
                double weight = ComputeOLSWeight(data, parent, child, n);
                W[parent, child] = NumOps.FromDouble(weight);
            }
        }

        return W;
    }

    /// <summary>
    /// HPC (Hybrid Parents and Children) — finds candidate PC set with spouse discovery.
    /// </summary>
    private HashSet<int> FindHPC(Matrix<T> data, int target, int d, int n)
    {
        var pc = new HashSet<int>();

        // Forward phase: add variables with strongest marginal association
        for (int candidate = 0; candidate < d; candidate++)
        {
            if (candidate == target) continue;
            double corr = Math.Abs(ComputeCorrelation(data, target, candidate));
            double z = Math.Sqrt(n - 3) * 0.5 * Math.Log((1 + corr) / (1 - corr + 1e-10));
            double pValue = 2 * (1 - NormalCDF(Math.Abs(z)));

            if (pValue < Alpha)
                pc.Add(candidate);
        }

        // Backward phase: remove false positives
        foreach (int member in pc.ToList())
        {
            var rest = pc.Where(v => v != member).Take(_maxConditioningSetSize).ToList();
            if (rest.Count > 0 && IsConditionallyIndependent(data, target, member, rest, n))
            {
                pc.Remove(member);
            }
        }

        // Spouse discovery: for each neighbor, check for spouses
        var spouses = new HashSet<int>();
        foreach (int neighbor in pc)
        {
            for (int candidate = 0; candidate < d; candidate++)
            {
                if (candidate == target || candidate == neighbor || pc.Contains(candidate)) continue;

                // Check if candidate is associated with target GIVEN neighbor
                // (spouse = both are parents of the same child "neighbor")
                var condSet = new List<int> { neighbor };
                if (!IsConditionallyIndependent(data, target, candidate, condSet, n))
                {
                    spouses.Add(candidate);
                }
            }
        }

        foreach (int spouse in spouses)
            pc.Add(spouse);

        return pc;
    }

    private bool IsConditionallyIndependent(Matrix<T> data, int i, int j, List<int> condSet, int n)
    {
        double partialCorr = ComputePartialCorrelation(data, i, j, condSet, n);
        int dof = n - condSet.Count - 3;
        if (dof <= 0) return true;

        double clamped = Math.Max(-0.999999, Math.Min(partialCorr, 0.999999));
        double z = Math.Sqrt(dof) * 0.5 * Math.Log((1 + clamped) / (1 - clamped));
        double pValue = 2 * (1 - NormalCDF(Math.Abs(z)));
        return pValue > Alpha;
    }

    private double ComputePartialCorrelation(Matrix<T> data, int i, int j, List<int> condSet, int n)
    {
        if (condSet.Count == 0)
            return ComputeCorrelation(data, i, j);

        // Residualize i and j on condSet via OLS
        var residI = new double[n];
        var residJ = new double[n];

        double meanI = 0, meanJ = 0;
        for (int k = 0; k < n; k++)
        {
            meanI += NumOps.ToDouble(data[k, i]);
            meanJ += NumOps.ToDouble(data[k, j]);
        }
        meanI /= n; meanJ /= n;

        if (condSet.Count == 1)
        {
            int c = condSet[0];
            double meanC = 0;
            for (int k = 0; k < n; k++) meanC += NumOps.ToDouble(data[k, c]);
            meanC /= n;

            double covIC = 0, covJC = 0, varC = 0;
            for (int k = 0; k < n; k++)
            {
                double dc = NumOps.ToDouble(data[k, c]) - meanC;
                covIC += (NumOps.ToDouble(data[k, i]) - meanI) * dc;
                covJC += (NumOps.ToDouble(data[k, j]) - meanJ) * dc;
                varC += dc * dc;
            }

            double betaI = varC > 1e-10 ? covIC / varC : 0;
            double betaJ = varC > 1e-10 ? covJC / varC : 0;

            for (int k = 0; k < n; k++)
            {
                double dc = NumOps.ToDouble(data[k, c]) - meanC;
                residI[k] = NumOps.ToDouble(data[k, i]) - meanI - betaI * dc;
                residJ[k] = NumOps.ToDouble(data[k, j]) - meanJ - betaJ * dc;
            }
        }
        else
        {
            // Multi-variable conditioning: use multivariate OLS to partial out all conditioning vars
            int p = condSet.Count;
            var condMeans = new double[p];
            for (int ci = 0; ci < p; ci++)
            {
                for (int k = 0; k < n; k++)
                    condMeans[ci] += NumOps.ToDouble(data[k, condSet[ci]]);
                condMeans[ci] /= n;
            }

            // Build normal equations X'X and X'y for both i and j
            var XtX = new double[p, p];
            var XtI = new double[p];
            var XtJ = new double[p];
            var dx = new double[p];
            for (int k = 0; k < n; k++)
            {
                for (int ci = 0; ci < p; ci++)
                    dx[ci] = NumOps.ToDouble(data[k, condSet[ci]]) - condMeans[ci];
                double di = NumOps.ToDouble(data[k, i]) - meanI;
                double dj = NumOps.ToDouble(data[k, j]) - meanJ;
                for (int a = 0; a < p; a++)
                {
                    XtI[a] += dx[a] * di;
                    XtJ[a] += dx[a] * dj;
                    for (int b = a; b < p; b++)
                        XtX[a, b] += dx[a] * dx[b];
                }
            }
            // Scale-aware ridge: proportional to average diagonal magnitude
            double avgDiag = 0;
            for (int a = 0; a < p; a++)
                avgDiag += Math.Abs(XtX[a, a]);
            double ridge = Math.Max(1e-10, avgDiag / p * 1e-8);
            for (int a = 0; a < p; a++)
            {
                XtX[a, a] += ridge;
                for (int b = a + 1; b < p; b++)
                    XtX[b, a] = XtX[a, b];
            }

            var betaI = SolveSmallSystem(XtX, XtI, p);
            var betaJ = SolveSmallSystem(XtX, XtJ, p);

            for (int k = 0; k < n; k++)
            {
                residI[k] = NumOps.ToDouble(data[k, i]) - meanI;
                residJ[k] = NumOps.ToDouble(data[k, j]) - meanJ;
                for (int ci = 0; ci < p; ci++)
                {
                    double dc = NumOps.ToDouble(data[k, condSet[ci]]) - condMeans[ci];
                    residI[k] -= betaI[ci] * dc;
                    residJ[k] -= betaJ[ci] * dc;
                }
            }
        }

        double mrI = residI.Average(), mrJ = residJ.Average();
        double sxy = 0, sxx = 0, syy = 0;
        for (int k = 0; k < n; k++)
        {
            double di = residI[k] - mrI;
            double dj = residJ[k] - mrJ;
            sxy += di * dj;
            sxx += di * di;
            syy += dj * dj;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    private double ComputeOLSWeight(Matrix<T> data, int parent, int child, int n)
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

        return varP > 1e-10 ? cov / varP : 0;
    }

    private static bool WouldCreateCycle(HashSet<int>[] parents, int from, int to, int d)
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
