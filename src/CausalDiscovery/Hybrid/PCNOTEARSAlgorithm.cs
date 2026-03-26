using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Hybrid;

/// <summary>
/// PC-NOTEARS — Hybrid of PC skeleton discovery with NOTEARS continuous optimization.
/// </summary>
/// <remarks>
/// <para>
/// PC-NOTEARS combines the constraint-based PC algorithm's efficient skeleton discovery
/// with NOTEARS' continuous optimization for edge weight estimation and orientation.
/// Phase 1 runs PC-style CI tests to identify which variable pairs are connected.
/// Phase 2 runs a NOTEARS-like optimization restricted to the PC skeleton, estimating
/// edge weights while enforcing the acyclicity constraint h(W) = 0.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item><b>PC skeleton phase:</b> Start with complete graph, remove edges via CI tests
///   at increasing conditioning set sizes</item>
/// <item><b>NOTEARS phase:</b> Initialize W from PC skeleton (OLS weights for connected pairs,
///   zero for removed pairs). Run gradient descent on L(W) + lambda * ||W||_1
///   subject to h(W) = tr(e^{W*W}) - d = 0, but only update entries in the skeleton</item>
/// <item>Threshold small weights and return the DAG</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> This hybrid first uses statistical tests to quickly figure out which
/// variable pairs MIGHT be connected, then uses optimization to find the exact edge weights
/// and directions — getting both speed and accuracy.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
public class PCNOTEARSAlgorithm<T> : HybridBase<T>
{
    private readonly double _lambda;
    private readonly double _threshold;
    private readonly int _maxConditioningSetSize;
    private readonly int _maxOptIterations;

    /// <inheritdoc/>
    public override string Name => "PC-NOTEARS";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes PC-NOTEARS with optional configuration.
    /// </summary>
    public PCNOTEARSAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyHybridOptions(options);
        _lambda = options?.SparsityPenalty ?? 0.1;
        _threshold = options?.EdgeThreshold ?? 0.3;
        _maxConditioningSetSize = options?.MaxConditioningSetSize ?? 3;
        _maxOptIterations = options?.MaxIterations ?? 50;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        // Phase 1: PC skeleton discovery
        var skeleton = DiscoverSkeleton(data, d, n);

        // Phase 2: NOTEARS-like optimization restricted to skeleton
        var weights = OptimizeWeights(data, skeleton, d, n);

        // Threshold small weights and convert to Matrix<T>
        var W = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (Math.Abs(weights[i, j]) >= _threshold)
                    W[i, j] = NumOps.FromDouble(weights[i, j]);
            }
        }

        return W;
    }

    /// <summary>
    /// Phase 1: PC-style skeleton discovery via conditional independence tests.
    /// </summary>
    private bool[,] DiscoverSkeleton(Matrix<T> data, int d, int n)
    {
        var adj = new bool[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                adj[i, j] = (i != j);

        for (int condSize = 0; condSize <= _maxConditioningSetSize; condSize++)
        {
            for (int i = 0; i < d; i++)
            {
                for (int j = i + 1; j < d; j++)
                {
                    if (!adj[i, j]) continue;

                    var neighbors = new List<int>();
                    for (int k = 0; k < d; k++)
                        if (k != i && k != j && (adj[i, k] || adj[j, k]))
                            neighbors.Add(k);

                    if (neighbors.Count < condSize) continue;

                    foreach (var condSet in GetCombinations(neighbors, condSize))
                    {
                        if (IsCI(data, i, j, condSet, n))
                        {
                            adj[i, j] = false;
                            adj[j, i] = false;
                            break;
                        }
                    }
                }
            }
        }

        return adj;
    }

    /// <summary>
    /// Phase 2: Optimize edge weights using coordinate descent restricted to skeleton.
    /// Uses the NOTEARS least-squares objective with L1 penalty.
    /// </summary>
    private double[,] OptimizeWeights(Matrix<T> data, bool[,] skeleton, int d, int n)
    {
        // Compute sample covariance
        var means = new double[d];
        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += NumOps.ToDouble(data[i, j]);
            means[j] /= n;
        }

        var S = new double[d, d];
        for (int a = 0; a < d; a++)
        {
            for (int b = 0; b < d; b++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                    sum += (NumOps.ToDouble(data[i, a]) - means[a]) *
                           (NumOps.ToDouble(data[i, b]) - means[b]);
                S[a, b] = sum / n;
            }
        }

        // Initialize W from OLS on skeleton
        var W = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (i != j && skeleton[i, j])
                {
                    // OLS coefficient
                    double sii = S[i, i] > 1e-10 ? S[i, i] : 1e-10;
                    W[i, j] = S[i, j] / sii;
                }
            }
        }

        // Coordinate descent with L1 penalty
        for (int iter = 0; iter < _maxOptIterations; iter++)
        {
            bool changed = false;

            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < d; i++)
                {
                    if (i == j || !skeleton[i, j]) continue;

                    // Gradient of least squares: -(S[i,j] - sum_k W[k,j]*S[i,k])
                    double gradient = -S[i, j];
                    for (int k = 0; k < d; k++)
                    {
                        if (k == j) continue;
                        gradient += W[k, j] * S[i, k];
                    }

                    // Proximal gradient update: w_new = soft_threshold(w_old - gradient/S[i,i], lambda/S[i,i])
                    double sii = S[i, i] + 1e-10;
                    double oldW = W[i, j];
                    double updated = oldW - gradient / sii;
                    double newW = SoftThreshold(updated, _lambda / sii);

                    if (Math.Abs(newW - oldW) > 1e-8)
                    {
                        W[i, j] = newW;
                        changed = true;
                    }
                }
            }

            if (!changed) break;
        }

        // Enforce DAG: remove weakest edges that create cycles
        // Use topological ordering via edge removal
        EnforceDag(W, d);

        var result = new double[d, d];
        Array.Copy(W, result, W.Length);
        return result;
    }

    private static void EnforceDag(double[,] W, int d)
    {
        // Iteratively remove weakest edge participating in a cycle until acyclic
        while (true)
        {
            // Find a cycle via DFS and trace the exact cycle path
            var visited = new int[d]; // 0=unvisited, 1=in-stack, 2=done
            var parent = new int[d];
            for (int i = 0; i < d; i++) { parent[i] = -1; }

            int cycleStart = -1, cycleEnd = -1;
            bool hasCycle = false;

            for (int start = 0; start < d && !hasCycle; start++)
            {
                if (visited[start] != 0) continue;
                var stack = new Stack<(int node, int nextChild)>();
                stack.Push((start, 0));
                visited[start] = 1;

                while (stack.Count > 0 && !hasCycle)
                {
                    var (node, nextChild) = stack.Pop();
                    bool pushed = false;
                    for (int child = nextChild; child < d; child++)
                    {
                        if (Math.Abs(W[node, child]) < 1e-12) continue;
                        if (visited[child] == 1)
                        {
                            // Back-edge found: cycle goes from child -> ... -> node -> child
                            cycleStart = child;
                            cycleEnd = node;
                            hasCycle = true;
                            break;
                        }
                        if (visited[child] == 0)
                        {
                            parent[child] = node;
                            stack.Push((node, child + 1));
                            stack.Push((child, 0));
                            visited[child] = 1;
                            pushed = true;
                            break;
                        }
                    }
                    if (!pushed && !hasCycle)
                        visited[node] = 2;
                }
            }

            if (!hasCycle) break;

            // Trace the cycle path: cycleStart -> ... -> cycleEnd -> cycleStart
            // Collect edges on this exact cycle
            int weakI = cycleEnd, weakJ = cycleStart;
            double weakVal = Math.Abs(W[cycleEnd, cycleStart]);

            int cur = cycleEnd;
            while (cur != cycleStart)
            {
                int prev = parent[cur];
                if (prev < 0) break; // safety
                double edgeVal = Math.Abs(W[prev, cur]);
                if (edgeVal < weakVal)
                {
                    weakVal = edgeVal;
                    weakI = prev;
                    weakJ = cur;
                }
                cur = prev;
            }

            W[weakI, weakJ] = 0;
        }
    }

    private static double SoftThreshold(double z, double threshold)
    {
        if (z > threshold) return z - threshold;
        if (z < -threshold) return z + threshold;
        return 0;
    }

    private bool IsCI(Matrix<T> data, int i, int j, List<int> condSet, int n)
    {
        double partialCorr;
        if (condSet.Count == 0)
        {
            partialCorr = ComputeCorrelation(data, i, j);
        }
        else
        {
            // Simple partial correlation via residualization
            partialCorr = ComputePartialCorrelation(data, i, j, condSet, n);
        }

        int dof = n - condSet.Count - 3;
        if (dof <= 0) return false; // insufficient samples: conservatively assume NOT independent (keep edge)

        double clamped = Math.Max(-0.999999, Math.Min(partialCorr, 0.999999));
        double z = Math.Sqrt(dof) * 0.5 * Math.Log((1 + clamped) / (1 - clamped));
        double pValue = 2 * (1 - NormalCDF(Math.Abs(z)));
        return pValue > Alpha;
    }

    private double ComputePartialCorrelation(Matrix<T> data, int i, int j, List<int> condSet, int n)
    {
        if (condSet.Count == 0) return ComputeCorrelation(data, i, j);

        double meanI = 0, meanJ = 0;
        for (int k = 0; k < n; k++)
        {
            meanI += NumOps.ToDouble(data[k, i]);
            meanJ += NumOps.ToDouble(data[k, j]);
        }
        meanI /= n; meanJ /= n;

        var residI = new double[n];
        var residJ = new double[n];
        for (int k = 0; k < n; k++)
        {
            residI[k] = NumOps.ToDouble(data[k, i]) - meanI;
            residJ[k] = NumOps.ToDouble(data[k, j]) - meanJ;
        }

        foreach (int c in condSet)
        {
            double meanC = 0;
            for (int k = 0; k < n; k++) meanC += NumOps.ToDouble(data[k, c]);
            meanC /= n;

            double covIC = 0, covJC = 0, varC = 0;
            for (int k = 0; k < n; k++)
            {
                double dc = NumOps.ToDouble(data[k, c]) - meanC;
                covIC += residI[k] * dc;
                covJC += residJ[k] * dc;
                varC += dc * dc;
            }

            if (varC > 1e-10)
            {
                double bI = covIC / varC;
                double bJ = covJC / varC;
                for (int k = 0; k < n; k++)
                {
                    double dc = NumOps.ToDouble(data[k, c]) - meanC;
                    residI[k] -= bI * dc;
                    residJ[k] -= bJ * dc;
                }
            }
        }

        double sxy = 0, sxx = 0, syy = 0;
        for (int k = 0; k < n; k++)
        {
            sxy += residI[k] * residJ[k];
            sxx += residI[k] * residI[k];
            syy += residJ[k] * residJ[k];
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    private static IEnumerable<List<int>> GetCombinations(List<int> items, int size)
    {
        if (size == 0) { yield return []; yield break; }
        for (int i = 0; i <= items.Count - size; i++)
        {
            foreach (var rest in GetCombinations(items.Skip(i + 1).ToList(), size - 1))
            {
                var combination = new List<int> { items[i] };
                combination.AddRange(rest);
                yield return combination;
            }
        }
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
}
