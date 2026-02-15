using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// NOTEARS Linear — continuous optimization for DAG structure learning with linear relationships.
/// </summary>
/// <remarks>
/// <para>
/// NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for
/// Structure learning) reformulates the combinatorial DAG constraint as a smooth equality constraint:
/// </para>
/// <para>
/// <b>Objective:</b> minimize F(W) = (1/2n)||X - XW||²_F + lambda1 * ||W||_1
/// </para>
/// <para>
/// <b>Subject to:</b> h(W) = tr(e^(W∘W)) - d = 0 (acyclicity constraint)
/// </para>
/// <para>
/// The acyclicity constraint h(W) is zero if and only if W encodes a DAG. The optimization
/// uses an augmented Lagrangian method with L-BFGS inner solver.
/// </para>
/// <para>
/// <b>For Beginners:</b> NOTEARS finds which variables cause which other variables by solving
/// a math optimization problem. It learns a weight matrix W where W[i,j] != 0 means "variable i
/// directly causes variable j." The clever trick is a special math formula (the trace of a matrix
/// exponential) that equals zero if and only if the graph has no cycles — which is required for
/// a valid causal graph. This avoids checking every possible graph structure.
/// </para>
/// <para>
/// <b>Algorithm (matching original paper):</b>
/// <list type="number">
/// <item>Initialize W = 0, alpha = 0, rho = 1</item>
/// <item>Inner loop: minimize augmented Lagrangian using L-BFGS with bounds [0, inf)</item>
/// <item>If h(W_new) > 0.25 * h(W_old): rho *= 10, repeat inner loop</item>
/// <item>Else: update alpha += rho * h, proceed</item>
/// <item>Stop when h &lt; h_tol or rho >= rho_max</item>
/// <item>Threshold small weights: W[|W| &lt; w_threshold] = 0</item>
/// </list>
/// </para>
/// <para>
/// <b>Default hyperparameters (from original paper):</b>
/// lambda1 = 0.1, max_iter = 100, h_tol = 1e-8, rho_max = 1e+16, w_threshold = 0.3
/// </para>
/// <para>
/// Reference: Zheng et al. (2018), "DAGs with NO TEARS: Continuous Optimization for Structure Learning",
/// Advances in Neural Information Processing Systems (NeurIPS).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NOTEARSLinear<T> : ContinuousOptimizationBase<T>
{
    #region Constants

    /// <summary>
    /// Maximum penalty parameter before termination.
    /// </summary>
    /// <remarks>When rho exceeds this value, the algorithm stops regardless of convergence.</remarks>
    private const double DEFAULT_RHO_MAX = 1e+16;

    /// <summary>
    /// Initial penalty parameter for the augmented Lagrangian.
    /// </summary>
    private const double DEFAULT_RHO_INIT = 1.0;

    /// <summary>
    /// Factor by which rho is increased when the constraint is not sufficiently reduced.
    /// </summary>
    /// <remarks>The paper uses 10x increases for aggressive constraint enforcement.</remarks>
    private const double RHO_MULTIPLY_FACTOR = 10.0;

    /// <summary>
    /// Threshold for determining if the constraint has been sufficiently reduced.
    /// </summary>
    /// <remarks>If h_new > 0.25 * h_old, rho is increased.</remarks>
    private const double CONSTRAINT_REDUCTION_THRESHOLD = 0.25;

    /// <summary>
    /// Maximum number of L-BFGS iterations for the inner optimization loop.
    /// </summary>
    private const int INNER_MAX_ITERATIONS = 1000;

    /// <summary>
    /// Convergence tolerance for L-BFGS inner loop.
    /// </summary>
    private const double INNER_TOLERANCE = 1e-6;

    /// <summary>
    /// L-BFGS memory size (number of past iterations to store).
    /// </summary>
    private const int LBFGS_MEMORY = 10;

    /// <summary>
    /// Learning rate for gradient descent steps within L-BFGS.
    /// </summary>
    private const double LBFGS_LEARNING_RATE = 1.0;

    #endregion

    #region Fields

    private double _rhoMax = DEFAULT_RHO_MAX;
    private int _lastIterations;
    private double _lastH;
    private double _lastLoss;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override string Name => "NOTEARS Linear";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of NOTEARSLinear with optional configuration.
    /// </summary>
    /// <param name="options">Causal discovery options. If null, paper defaults are used.</param>
    public NOTEARSLinear(CausalDiscoveryOptions? options = null)
    {
        ApplyOptions(options);
        if (options?.MaxPenalty.HasValue == true)
        {
            _rhoMax = options.MaxPenalty.Value;
        }
    }

    #endregion

    #region Core Algorithm

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        double[,] X = MatrixToDoubleArray(data);
        int d = data.Columns;

        // Initialize: W = 0, alpha = 0, rho = rho_init
        var W = new double[d, d];
        double alpha = 0.0;
        double rho = DEFAULT_RHO_INIT;

        // Initial constraint value
        var (hOld, _) = ComputeNOTEARSConstraint(W);

        _lastIterations = 0;

        // Augmented Lagrangian outer loop
        for (int outerIter = 0; outerIter < MaxIterations; outerIter++)
        {
            _lastIterations = outerIter + 1;

            // Inner loop: minimize augmented Lagrangian via L-BFGS
            W = SolveLBFGSSubproblem(X, W, alpha, rho, d);

            // Evaluate constraint at new W
            var (hNew, _) = ComputeNOTEARSConstraint(W);

            if (hNew > CONSTRAINT_REDUCTION_THRESHOLD * hOld)
            {
                // Constraint not sufficiently reduced — increase penalty
                rho *= RHO_MULTIPLY_FACTOR;
            }
            else
            {
                // Good progress — update Lagrange multiplier
                alpha += rho * hNew;
                hOld = hNew;
            }

            _lastH = hNew;

            // Check convergence
            if (hNew < HTolerance)
            {
                break;
            }

            if (rho >= _rhoMax)
            {
                break;
            }
        }

        // Compute final loss for reporting
        var (finalLoss, _) = ComputeL2Loss(X, W);
        _lastLoss = finalLoss + Lambda1 * ComputeL1Norm(W);

        // Threshold and clean
        var WThresholded = ThresholdAndClean(W, WThreshold);

        return DoubleArrayToMatrix(WThresholded);
    }

    /// <summary>
    /// Solves the augmented Lagrangian subproblem using L-BFGS.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Minimizes: L(W) = loss(W) + lambda1*||W||_1 + alpha*h(W) + (rho/2)*h(W)^2
    /// using a limited-memory BFGS algorithm with projected gradient for the L1 term.
    /// </para>
    /// </remarks>
    private double[,] SolveLBFGSSubproblem(double[,] X, double[,] W, double alpha, double rho, int d)
    {
        // Flatten W to vector for L-BFGS (excluding diagonal)
        int vecLen = d * d;
        double[] w = FlattenMatrix(W, d);
        double[] prevW = (double[])w.Clone();
        double[] prevGrad = new double[vecLen];

        // L-BFGS memory
        var sVectors = new List<double[]>();
        var yVectors = new List<double[]>();

        for (int iter = 0; iter < INNER_MAX_ITERATIONS; iter++)
        {
            // Compute objective and gradient
            var currentW = UnflattenMatrix(w, d);
            var (grad, objValue) = ComputeAugmentedLagrangianGradient(X, currentW, alpha, rho, d);
            double[] g = FlattenMatrix(grad, d);

            // Add L1 subgradient
            for (int i = 0; i < vecLen; i++)
            {
                int row = i / d;
                int col = i % d;
                if (row != col) // Skip diagonal
                {
                    g[i] += Lambda1 * Math.Sign(w[i]);
                }
            }

            // Check convergence: ||gradient||_inf < tolerance
            double maxGrad = 0;
            for (int i = 0; i < vecLen; i++)
            {
                maxGrad = Math.Max(maxGrad, Math.Abs(g[i]));
            }

            if (maxGrad < INNER_TOLERANCE)
            {
                break;
            }

            // Compute L-BFGS direction
            double[] direction = ComputeLBFGSDirection(g, sVectors, yVectors);

            // Line search with Armijo condition
            double stepSize = LBFGS_LEARNING_RATE;
            for (int ls = 0; ls < 20; ls++)
            {
                var trialW = new double[vecLen];
                for (int i = 0; i < vecLen; i++)
                {
                    trialW[i] = w[i] + stepSize * direction[i];
                }

                // Project: zero out diagonal entries
                for (int i = 0; i < d; i++)
                {
                    trialW[i * d + i] = 0;
                }

                var trialWMat = UnflattenMatrix(trialW, d);
                double trialObj = ComputeAugmentedLagrangianObjective(X, trialWMat, alpha, rho)
                    + Lambda1 * ComputeL1NormFromFlat(trialW, d);

                double directionalDeriv = 0;
                for (int i = 0; i < vecLen; i++)
                {
                    directionalDeriv += g[i] * direction[i];
                }

                if (trialObj <= objValue + Lambda1 * ComputeL1NormFromFlat(w, d) + 1e-4 * stepSize * directionalDeriv)
                {
                    w = trialW;
                    break;
                }

                stepSize *= 0.5;
                if (ls == 19)
                {
                    // Line search failed — take small step anyway
                    for (int i = 0; i < vecLen; i++)
                    {
                        w[i] = w[i] + 1e-4 * direction[i];
                    }

                    // Zero diagonal
                    for (int i = 0; i < d; i++)
                    {
                        w[i * d + i] = 0;
                    }
                }
            }

            // Update L-BFGS memory
            var s = new double[vecLen];
            var y = new double[vecLen];
            var newGrad = FlattenMatrix(ComputeAugmentedLagrangianGradient(X, UnflattenMatrix(w, d), alpha, rho, d).Gradient, d);

            for (int i = 0; i < vecLen; i++)
            {
                s[i] = w[i] - prevW[i];
                y[i] = newGrad[i] - prevGrad[i];
            }

            double sy = 0;
            for (int i = 0; i < vecLen; i++)
            {
                sy += s[i] * y[i];
            }

            if (sy > 1e-10) // Curvature condition
            {
                sVectors.Add(s);
                yVectors.Add(y);
                if (sVectors.Count > LBFGS_MEMORY)
                {
                    sVectors.RemoveAt(0);
                    yVectors.RemoveAt(0);
                }
            }

            prevW = (double[])w.Clone();
            prevGrad = newGrad;
        }

        return UnflattenMatrix(w, d);
    }

    /// <summary>
    /// Computes the gradient of the augmented Lagrangian (without L1 term, which is handled separately).
    /// L(W) = loss(W) + alpha*h(W) + (rho/2)*h(W)^2
    /// </summary>
    private (double[,] Gradient, double Objective) ComputeAugmentedLagrangianGradient(
        double[,] X, double[,] W, double alpha, double rho, int d)
    {
        var (loss, lossGrad) = ComputeL2Loss(X, W);
        var (h, hGrad) = ComputeNOTEARSConstraint(W);

        double objective = loss + alpha * h + 0.5 * rho * h * h;

        var grad = new double[d, d];
        double augFactor = alpha + rho * h;

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                grad[i, j] = lossGrad[i, j] + augFactor * hGrad[i, j];
            }
        }

        return (grad, objective);
    }

    /// <summary>
    /// Computes the augmented Lagrangian objective value (without L1).
    /// </summary>
    private double ComputeAugmentedLagrangianObjective(double[,] X, double[,] W, double alpha, double rho)
    {
        var (loss, _) = ComputeL2Loss(X, W);
        var (h, _) = ComputeNOTEARSConstraint(W);
        return loss + alpha * h + 0.5 * rho * h * h;
    }

    /// <summary>
    /// L-BFGS two-loop recursion to compute search direction.
    /// </summary>
    private static double[] ComputeLBFGSDirection(double[] gradient, List<double[]> sVectors, List<double[]> yVectors)
    {
        int n = gradient.Length;
        int m = sVectors.Count;

        double[] q = (double[])gradient.Clone();

        if (m == 0)
        {
            // No history — use steepest descent
            var dir = new double[n];
            for (int i = 0; i < n; i++)
            {
                dir[i] = -q[i];
            }

            return dir;
        }

        double[] alphas = new double[m];

        // First loop (backward)
        for (int i = m - 1; i >= 0; i--)
        {
            double sy = DotProduct(sVectors[i], yVectors[i]);
            if (sy < 1e-15) sy = 1e-15;
            double rhoI = 1.0 / sy;
            alphas[i] = rhoI * DotProduct(sVectors[i], q);

            for (int j = 0; j < n; j++)
            {
                q[j] -= alphas[i] * yVectors[i][j];
            }
        }

        // Scale by gamma = (s^T y) / (y^T y) from most recent pair
        double gamma;
        {
            double sy = DotProduct(sVectors[m - 1], yVectors[m - 1]);
            double yy = DotProduct(yVectors[m - 1], yVectors[m - 1]);
            gamma = (yy > 1e-15) ? sy / yy : 1.0;
        }

        var r = new double[n];
        for (int i = 0; i < n; i++)
        {
            r[i] = gamma * q[i];
        }

        // Second loop (forward)
        for (int i = 0; i < m; i++)
        {
            double sy = DotProduct(sVectors[i], yVectors[i]);
            if (sy < 1e-15) sy = 1e-15;
            double rhoI = 1.0 / sy;
            double beta = rhoI * DotProduct(yVectors[i], r);

            for (int j = 0; j < n; j++)
            {
                r[j] += (alphas[i] - beta) * sVectors[i][j];
            }
        }

        // Negate for descent direction
        for (int i = 0; i < n; i++)
        {
            r[i] = -r[i];
        }

        return r;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Gets the iteration count and convergence info from the last run.
    /// </summary>
    public (int Iterations, double FinalH, double FinalLoss) GetLastRunInfo()
    {
        return (_lastIterations, _lastH, _lastLoss);
    }

    private static double DotProduct(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    private static double[] FlattenMatrix(double[,] matrix, int d)
    {
        var result = new double[d * d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                result[i * d + j] = matrix[i, j];
            }
        }

        return result;
    }

    private static double[,] UnflattenMatrix(double[] vector, int d)
    {
        var result = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                result[i, j] = vector[i * d + j];
            }
        }

        return result;
    }

    private static double ComputeL1NormFromFlat(double[] w, int d)
    {
        double sum = 0;
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (i != j)
                {
                    sum += Math.Abs(w[i * d + j]);
                }
            }
        }

        return sum;
    }

    #endregion
}
