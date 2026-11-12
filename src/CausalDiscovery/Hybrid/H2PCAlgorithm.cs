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
[ResearchPaper("A Hybrid Algorithm for Bayesian Network Structure Learning with Application to Multi-Label Learning", "https://doi.org/10.1016/j.eswa.2014.03.032", Year = 2014, Authors = "Maxime Gasse, Alex Aussem, Haytham Elghazel")]
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
        var weights = new Dictionary<int, double>();
        for (int child = 0; child < d; child++)
        {
            foreach (int parent in parentSets[child])
            {
                weights[parent * d + child] = ComputeOLSWeight(data, parent, child, n);
            }
        }

        return CausalTensorHelpers.FromStatistics<T>(d, d, weights);
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

        // Spouse discovery: for each neighbor, check for spouses.
        // A spouse of target through neighbor means: both target and candidate are parents
        // of neighbor. We detect this by checking if candidate is associated with neighbor
        // given target (not associated with target given neighbor — that was the old buggy logic).
        // Per Gasse et al. (2014): "for each Y in PC(X), test if Z is dependent on Y given X"
        var spouses = new HashSet<int>();
        foreach (int neighbor in pc)
        {
            for (int candidate = 0; candidate < d; candidate++)
            {
                if (candidate == target || candidate == neighbor || pc.Contains(candidate)) continue;

                // Check if candidate is associated with NEIGHBOR given TARGET
                var condSet = new List<int> { target };
                if (!IsConditionallyIndependent(data, neighbor, candidate, condSet, n))
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

        // Residualize i and j on condSet via OLS on centered columns
        var x = Tensor<T>.FromMatrix(data);
        var conditioning = CenterColumns(SelectColumns(x, condSet.ToArray()));
        var yi = CenterColumns(SelectColumns(x, new[] { i }));
        var yj = CenterColumns(SelectColumns(x, new[] { j }));
        Tensor<T> residI, residJ;

        if (condSet.Count == 1)
        {
            // Simple regression on the one conditioning variable; slope 0 unless its variance is above 1e-10.
            T varC = Engine.TensorSum(Engine.TensorSquare(conditioning));
            bool usable = NumOps.GreaterThan(varC, NumOps.FromDouble(1e-10));
            T betaI = usable ? NumOps.Divide(Engine.TensorSum(Engine.TensorMultiply(conditioning, yi)), varC) : NumOps.Zero;
            T betaJ = usable ? NumOps.Divide(Engine.TensorSum(Engine.TensorMultiply(conditioning, yj)), varC) : NumOps.Zero;
            residI = Engine.TensorSubtract(yi, Engine.TensorMultiplyScalar(conditioning, betaI));
            residJ = Engine.TensorSubtract(yj, Engine.TensorMultiplyScalar(conditioning, betaJ));
        }
        else
        {
            // Multi-variable conditioning: use multivariate OLS to partial out all conditioning vars
            int p = condSet.Count;
            var conditioningT = Engine.TensorTranspose(conditioning);
            var gram = CausalTensorHelpers.Symmetrize(Engine.TensorMatMul(conditioningT, conditioning));

            // Scale-aware ridge: proportional to average diagonal magnitude
            T ridge = NumOps.Multiply(Engine.TensorMean(Engine.TensorAbs(Engine.TensorDiagonal(gram))), NumOps.FromDouble(1e-8));
            T minRidge = NumOps.FromDouble(1e-10);
            if (NumOps.LessThan(ridge, minRidge)) ridge = minRidge;
            gram = Engine.TensorAdd(gram, Engine.TensorMultiplyScalar(Engine.TensorEye<T>(p), ridge));

            var betaI = SolveSmallSystem(gram, Engine.TensorMatMul(conditioningT, yi).Reshape(new[] { p }));
            var betaJ = SolveSmallSystem(gram, Engine.TensorMatMul(conditioningT, yj).Reshape(new[] { p }));
            residI = Engine.TensorSubtract(yi, Engine.TensorMatMul(conditioning, betaI.Reshape(new[] { p, 1 })));
            residJ = Engine.TensorSubtract(yj, Engine.TensorMatMul(conditioning, betaJ.Reshape(new[] { p, 1 })));
        }

        return NumOps.ToDouble(CausalTensorHelpers.PearsonCorrelation(residI, residJ));
    }

    private double ComputeOLSWeight(Matrix<T> data, int parent, int child, int n)
    {
        return NumOps.ToDouble(CausalTensorHelpers.RegressionSlope(data, parent, child));
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

    // SolveSmallSystem is inherited from CausalDiscoveryBase<T>
}
