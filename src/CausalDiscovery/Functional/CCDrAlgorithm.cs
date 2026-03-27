using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// CCDr (Concave penalized Coordinate Descent with reparameterization) for DAG learning.
/// </summary>
/// <remarks>
/// <para>
/// CCDr learns a Bayesian network structure by minimizing a penalized negative log-likelihood
/// using coordinate descent with a concave penalty (MCP — Minimax Concave Penalty). The MCP
/// penalty provides near-unbiased estimation of large coefficients while still inducing sparsity.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize adjacency matrix W = 0</item>
/// <item>For each target variable j, cycle through candidate parents i != j</item>
/// <item>Update W[i,j] via coordinate descent on the penalized Gaussian log-likelihood</item>
/// <item>Apply soft-thresholding with MCP penalty derivative</item>
/// <item>Enforce acyclicity: reject updates that would create a cycle</item>
/// <item>Repeat until convergence or max iterations</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CCDr finds causal relationships by fitting a statistical model
/// where each variable depends on a sparse set of "parent" variables. It uses a clever
/// penalty (MCP) that encourages most connections to be zero while keeping strong ones
/// unbiased. The algorithm checks each potential connection one at a time and only keeps
/// it if removing it would significantly hurt the model fit.
/// </para>
/// <para>
/// Reference: Aragam and Zhou (2015), "Concave Penalized Estimation of Sparse
/// Gaussian Bayesian Networks", JMLR.
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
[ModelPaper("Concave Penalized Estimation of Sparse Gaussian Bayesian Networks", "https://jmlr.org/papers/v16/aragam15a.html", Year = 2015, Authors = "Bryon Aragam, Qing Zhou")]
public class CCDrAlgorithm<T> : FunctionalBase<T>
{
    private readonly double _lambda;
    private readonly double _gamma;
    private readonly double _threshold;
    private readonly int _maxIterations;

    /// <inheritdoc/>
    public override string Name => "CCDr";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes CCDr with optional configuration.
    /// </summary>
    public CCDrAlgorithm(CausalDiscoveryOptions? options = null)
    {
        // Default lambda: use a conservative value that works for small datasets.
        // Theory suggests sqrt(log(d)/n), but in practice a smaller value is needed
        // to avoid over-penalizing true edges with moderate sample sizes.
        _lambda = options?.SparsityPenalty ?? 0.01;
        _gamma = options?.ConcavityParameter ?? 2.0;
        _threshold = options?.EdgeThreshold ?? 0.1;
        _maxIterations = options?.MaxIterations ?? 100;
        if (_gamma <= 1.0)
            throw new ArgumentException("ConcavityParameter (MCP gamma) must be > 1.0.");
        if (_lambda < 0)
            throw new ArgumentException("SparsityPenalty must be non-negative.");
        if (_maxIterations < 1)
            throw new ArgumentException("MaxIterations must be at least 1.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        var standardized = StandardizeData(data);

        // Compute sample covariance matrix S = (1/n) X^T X
        var S = ComputeCovarianceMatrix(standardized);

        // Initialize weight matrix W = 0
        var W = new Matrix<T>(d, d);

        // Coordinate descent
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            bool changed = false;

            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < d; i++)
                {
                    if (i == j) continue;

                    // Compute the partial residual for W[i,j]
                    double sij = NumOps.ToDouble(S[i, j]);
                    double sii = NumOps.ToDouble(S[i, i]);

                    // Subtract contributions from other parents of j
                    double partialResidual = sij;
                    for (int k = 0; k < d; k++)
                    {
                        if (k == i || k == j) continue;
                        double wkj = NumOps.ToDouble(W[k, j]);
                        if (Math.Abs(wkj) > 1e-15)
                        {
                            partialResidual -= wkj * NumOps.ToDouble(S[i, k]);
                        }
                    }

                    // MCP soft-thresholding
                    double oldW = NumOps.ToDouble(W[i, j]);
                    double newW = MCPProximal(partialResidual, sii, _lambda, _gamma);

                    // Check acyclicity: would setting W[i,j] = newW create a cycle?
                    if (Math.Abs(newW) > 1e-15 && Math.Abs(oldW) < 1e-15)
                    {
                        // New edge i -> j: check if j already reaches i
                        W[i, j] = NumOps.FromDouble(newW);
                        if (HasCycle(W, d))
                        {
                            W[i, j] = NumOps.FromDouble(oldW); // Revert
                            continue;
                        }
                    }
                    else
                    {
                        W[i, j] = NumOps.FromDouble(newW);
                    }

                    if (Math.Abs(newW - oldW) > 1e-8)
                        changed = true;
                }
            }

            if (!changed) break;
        }

        // Threshold small weights
        return ThresholdMatrix(W, _threshold);
    }

    /// <summary>
    /// MCP (Minimax Concave Penalty) proximal operator.
    /// </summary>
    private static double MCPProximal(double z, double denominator, double lambda, double gamma)
    {
        if (denominator < 1e-15) return 0;

        double u = z / denominator;
        double absU = Math.Abs(u);

        if (absU <= lambda)
        {
            return 0; // Below threshold
        }
        else if (absU <= gamma * lambda)
        {
            // Concave penalty region: shrink less than LASSO
            double sign = Math.Sign(u);
            return sign * (absU - lambda) / (1.0 - 1.0 / gamma);
        }
        else
        {
            // Beyond gamma * lambda: no penalty (unbiased)
            return u;
        }
    }

    /// <summary>
    /// Checks if the adjacency matrix contains a cycle using DFS.
    /// </summary>
    private bool HasCycle(Matrix<T> W, int d)
    {
        // 0 = unvisited, 1 = in-progress, 2 = done
        var state = new int[d];

        for (int i = 0; i < d; i++)
        {
            if (state[i] == 0 && DFSHasCycle(W, d, i, state))
                return true;
        }

        return false;
    }

    private bool DFSHasCycle(Matrix<T> W, int d, int node, int[] state)
    {
        state[node] = 1;

        for (int j = 0; j < d; j++)
        {
            if (Math.Abs(NumOps.ToDouble(W[node, j])) < 1e-15) continue;

            if (state[j] == 1) return true; // Back edge = cycle
            if (state[j] == 0 && DFSHasCycle(W, d, j, state)) return true;
        }

        state[node] = 2;
        return false;
    }
}
