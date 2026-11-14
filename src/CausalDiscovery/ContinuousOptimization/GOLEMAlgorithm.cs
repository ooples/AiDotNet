using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// GOLEM — Gradient-based Optimization with Likelihood for structure learning of linear DAGs.
/// </summary>
/// <remarks>
/// <para>
/// GOLEM uses a likelihood-based score function with a soft DAG penalty. Unlike NOTEARS which
/// uses a hard acyclicity constraint via augmented Lagrangian, GOLEM optimizes a penalized
/// likelihood objective directly, avoiding the expensive inner-outer loop structure.
/// </para>
/// <para>
/// <b>Two variants:</b>
/// <list type="bullet">
/// <item><b>GOLEM-EV (Equal Variance):</b> Assumes equal noise variance across all variables.
/// Score = n*d/2 * log(||X - XW||²_F) - log|det(I - W)| + lambda1*||W||_1</item>
/// <item><b>GOLEM-NV (Non-equal Variance):</b> Allows different noise variances.
/// Score = sum_j [n/2 * log(||X_j - XW_j||²) - log|det(I - W)|/d] + lambda1*||W||_1</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> GOLEM is an alternative to NOTEARS that's simpler to implement and
/// can be more efficient. Instead of using a special "acyclicity constraint," it bakes the
/// constraint into the objective function itself. It directly measures how likely the data is
/// given a particular causal graph, plus a penalty for cycles.
/// </para>
/// <para>
/// <b>Key advantage:</b> Single-loop optimization (no inner/outer loops), uses standard
/// gradient descent optimizers (Adam), and the log-determinant term naturally penalizes cycles.
/// </para>
/// <para>
/// Reference: Ng et al. (2020), "On the Role of Sparsity and DAG Constraints for Learning
/// Linear DAGs", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("On the Role of Sparsity and DAG Constraints for Learning Linear DAGs", "https://proceedings.neurips.cc/paper/2020/hash/d04d42cdf14579cd294e5079e0745411-Abstract.html", Year = 2020, Authors = "Ignavier Ng, AmirEmad Ghassami, Kun Zhang")]
public class GOLEMAlgorithm<T> : ContinuousOptimizationBase<T>
{
    #region Constants

    private const double DEFAULT_LEARNING_RATE = 0.001;
    private const double ADAM_BETA1 = 0.9;
    private const double ADAM_BETA2 = 0.999;
    private const int DEFAULT_NUM_ITERATIONS = 50000;
    private const double DEFAULT_LAMBDA2 = 5.0;
    private const double CONVERGENCE_TOL = 1e-6;
    private const int CHECKPOINT_INTERVAL = 1000;

    #endregion

    #region Fields

    private double _lambda2 = DEFAULT_LAMBDA2;
    private bool _equalVariance = true;
    private int _lastIterations;
    private double _lastH;
    private double _lastLoss;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override string Name => "GOLEM";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes GOLEM with optional configuration.
    /// </summary>
    public GOLEMAlgorithm(CausalDiscoveryOptions? options = null, bool equalVariance = true)
    {
        ApplyOptions(options);
        // Preserve GOLEM's published 50,000-step default while honoring the
        // common MaxIterations option when callers explicitly provide it.
        if (options?.MaxIterations is null)
            MaxIterations = DEFAULT_NUM_ITERATIONS;
        _equalVariance = equalVariance;
    }

    #endregion

    #region Core Algorithm

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        // Initialize W = 0
        var weights = new Tensor<T>(new[] { d, d });

        // Adam state
        var m = new Tensor<T>(new[] { d, d });
        var v = new Tensor<T>(new[] { d, d });
        var offDiagonal = OffDiagonalMask(d);

        double prevObj = double.MaxValue;
        _lastIterations = 0;

        for (int iter = 1; iter <= MaxIterations; iter++)
        {
            _lastIterations = iter;

            // Objective and gradient, then one Adam step (diagonal re-zeroed)
            double obj;
            (weights, m, v, obj) = GolemAdamStep(data, weights, m, v, offDiagonal, iter, n, d);

            // Convergence check
            if (iter % CHECKPOINT_INTERVAL == 0)
            {
                if (Math.Abs(obj - prevObj) / (Math.Abs(prevObj) + 1e-15) < CONVERGENCE_TOL)
                    break;
                prevObj = obj;
            }
        }

        // Compute final metrics
        var W = weights.ToMatrix();
        var (finalLoss, _) = ComputeL2Loss(data, W);
        _lastLoss = finalLoss;

        var (hFinal, _) = ComputeNOTEARSConstraint(W);
        _lastH = hFinal;

        return ThresholdAndClean(W, WThreshold);
    }

    #endregion

    /// <summary>
    /// One GOLEM step: score gradient plus the DAG-penalty gradient and the L1 subgradient
    /// (off-diagonal), followed by an Adam update and a zeroed diagonal.
    /// </summary>
    private (Tensor<T> W, Tensor<T> M, Tensor<T> V, double Objective) GolemAdamStep(
        Matrix<T> data, Tensor<T> W, Tensor<T> m, Tensor<T> v, Tensor<T> offDiagonal, int iter, int n, int d)
    {
        var current = W.ToMatrix();

        // Compute objective and gradient
        var (obj, grad) = _equalVariance
            ? ComputeGOLEMEV(data, current, n, d)
            : ComputeGOLEMNV(data, current, n, d);

        // Add L1 subgradient and DAG penalty gradient
        var (dagPenalty, dagGrad) = ComputeDAGPenalty(current, d);
        obj += _lambda2 * dagPenalty;

        var zeros = Engine.TensorZerosLike(W);
        var l1 = Engine.TensorWhere(offDiagonal,
            Engine.TensorMultiplyScalar(Engine.TensorSign(W), NumOps.FromDouble(Lambda1)), zeros);
        var g = Engine.TensorAdd(
            Engine.TensorAdd(Tensor<T>.FromMatrix(grad),
                Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(dagGrad), NumOps.FromDouble(_lambda2))),
            l1);

        // Adam update
        var (updated, newM, newV) = AdamUpdate(W, g, m, v, ADAM_BETA1, ADAM_BETA2, 1e-8, DEFAULT_LEARNING_RATE,
            1 - Math.Pow(ADAM_BETA1, iter), 1 - Math.Pow(ADAM_BETA2, iter));

        // Zero diagonal
        return (Engine.TensorWhere(offDiagonal, updated, zeros), newM, newV, obj);
    }

    #region GOLEM Score Functions

    /// <summary>
    /// GOLEM-EV score: equal variance assumption.
    /// Score = n*d/2 * log(||X - XW||²_F / n) - log|det(I - W)|
    /// Gradient computed analytically.
    /// </summary>
    private (double Score, Matrix<T> Gradient) ComputeGOLEMEV(Matrix<T> X, Matrix<T> W, int n, int d)
    {
        // Compute residual R = X - XW
        var x = Tensor<T>.FromMatrix(X);
        var w = Tensor<T>.FromMatrix(W);
        var R = Engine.TensorSubtract(x, Engine.TensorMatMul(x, w));
        double rss = NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(R)));

        // (double) on one operand FIRST: n * d is an int multiply that overflows
        // before the division promotes it, so a large graph silently divides by a
        // negative or wrapped count.
        double sigmaSquared = rss / ((double)n * d);
        if (sigmaSquared < 1e-15) sigmaSquared = 1e-15;

        // I - W
        var ImW = Engine.TensorSubtract(Engine.TensorEye<T>(d), w);

        double logDetImW = LogAbsDeterminant(ImW);

        // Score = (n*d/2) * log(sigma^2) - n * log|det(I-W)|
        // Same overflow: n * d is evaluated in int before /2.0 promotes the result.
        double score = ((double)n * d / 2.0) * Math.Log(sigmaSquared) - n * logDetImW;

        // X^T R term: d/dW [(n*d/2)*log(sigma^2)] = -X^T R / sigma^2
        var xtr = Engine.TensorMatMul(Engine.TensorTranspose(x), R);
        var grad = Engine.TensorDivideScalar(Engine.TensorNegate(xtr), NumOps.FromDouble(sigmaSquared));

        // Log-det term: d/dW [-n * log|det(I-W)|] = n * (I-W)^{-T}
        grad = AddLogDetGradient(grad, ImW, n);

        return (score, grad.ToMatrix());
    }

    /// <summary>
    /// Adds n * (I - W)^{-T} to the gradient when I - W is invertible.
    /// </summary>
    private Tensor<T> AddLogDetGradient(Tensor<T> grad, Tensor<T> ImW, int n)
    {
        var invImW = InvertSmallMatrix(ImW.ToMatrix());
        if (invImW is null)
            return grad;

        return Engine.TensorAdd(grad, Engine.TensorMultiplyScalar(
            Engine.TensorTranspose(Tensor<T>.FromMatrix(invImW)), NumOps.FromDouble(n)));
    }

    /// <summary>
    /// GOLEM-NV score: non-equal variance assumption.
    /// Score = sum_j [n/2 * log(||R_j||^2 / n)] - n * log|det(I - W)|
    /// </summary>
    private (double Score, Matrix<T> Gradient) ComputeGOLEMNV(Matrix<T> X, Matrix<T> W, int n, int d)
    {
        var x = Tensor<T>.FromMatrix(X);
        var w = Tensor<T>.FromMatrix(W);
        var R = Engine.TensorSubtract(x, Engine.TensorMatMul(x, w));
        var rssPerVar = Engine.ReduceSum(Engine.TensorSquare(R), new[] { 0 }, true); // [1, d]

        // I - W
        var ImW = Engine.TensorSubtract(Engine.TensorEye<T>(d), w);

        double logDetImW = LogAbsDeterminant(ImW);

        // Guard against singular I-W: return large penalty to steer optimization away
        if (double.IsNegativeInfinity(logDetImW) || double.IsNaN(logDetImW))
        {
            var zeroGrad = new Matrix<T>(d, d);
            return (double.MaxValue / 2, zeroGrad);
        }

        // Per-variable residual variance, floored at 1e-15
        var sigma = Engine.TensorMax(Engine.TensorDivideScalar(rssPerVar, NumOps.FromDouble(n)), NumOps.FromDouble(1e-15));
        double score = -n * logDetImW
            + (n / 2.0) * NumOps.ToDouble(Engine.TensorSum(Engine.TensorLog(sigma)));

        // Gradient: -X^T R / rss_j per column
        var xtr = Engine.TensorMatMul(Engine.TensorTranspose(x), R);
        var grad = Engine.TensorDivide(Engine.TensorNegate(xtr), Engine.TensorBroadcastTo(rssPerVar, new[] { d, d }));

        grad = AddLogDetGradient(grad, ImW, n);

        return (score, grad.ToMatrix());
    }

    /// <summary>
    /// Computes a soft DAG penalty: h(W) = tr(e^(W∘W)) - d.
    /// </summary>
    private (double Penalty, Matrix<T> Gradient) ComputeDAGPenalty(Matrix<T> W, int d)
    {
        return ComputeNOTEARSConstraint(W);
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

    #endregion
}
