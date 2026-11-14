using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// DAGMA Linear — DAG learning via M-matrices and a log-determinant acyclicity characterization.
/// </summary>
/// <remarks>
/// <para>
/// DAGMA replaces the NOTEARS matrix exponential constraint with a log-determinant constraint:
/// h(W, s) = -log det(sI - W∘W) + d*log(s), which has better gradient behavior and is ~10x faster.
/// </para>
/// <para>
/// <b>Optimization:</b> Uses a central path / barrier method instead of augmented Lagrangian.
/// At each outer iteration, the domain parameter s decreases, tightening the DAG constraint.
/// Inner optimization uses gradient descent with Adam optimizer.
/// </para>
/// <para>
/// <b>For Beginners:</b> DAGMA does the same thing as NOTEARS (finding causal relationships)
/// but uses a different math trick that's faster and more numerically stable. Instead of
/// using the matrix exponential, it uses the log-determinant, which has nicer gradients
/// and converges faster — especially for larger problems.
/// </para>
/// <para>
/// <b>Default hyperparameters (from original paper):</b>
/// lambda1 = 0.03, T = 5, mu_init = 1.0, mu_factor = 0.1,
/// s = [1.0, 0.9, 0.8, 0.7, 0.6], lr = 0.0003, w_threshold = 0.3
/// </para>
/// <para>
/// Reference: Bello et al. (2022), "DAGMA: Learning DAGs via M-matrices and a Log-Determinant
/// Acyclicity Characterization", NeurIPS.
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
[ResearchPaper("DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization", "https://proceedings.neurips.cc/paper_files/paper/2022/hash/2e6a3b0e4e43db1cbba0c5e5b7ee8a41-Abstract-Conference.html", Year = 2022, Authors = "Kevin Bello, Bryon Aragam, Pradeep Ravikumar")]
public class DAGMALinear<T> : ContinuousOptimizationBase<T>
{
    #region Constants

    /// <summary>
    /// Default L1 penalty for DAGMA (different from NOTEARS default).
    /// </summary>
    /// <remarks>The DAGMA paper uses a smaller lambda1 than NOTEARS.</remarks>
    private const double DAGMA_DEFAULT_LAMBDA1 = 0.03;

    /// <summary>
    /// Number of outer iterations (central path steps).
    /// </summary>
    /// <remarks>Each step tightens the DAG constraint by reducing s.</remarks>
    private const int DEFAULT_T = 5;

    /// <summary>
    /// Initial penalty weight for the log-det constraint.
    /// </summary>
    private const double DEFAULT_MU_INIT = 1.0;

    /// <summary>
    /// Multiplicative decay factor for mu between outer iterations.
    /// </summary>
    private const double DEFAULT_MU_FACTOR = 0.1;

    /// <summary>
    /// Learning rate for the Adam optimizer in the inner loop.
    /// </summary>
    private const double DEFAULT_LEARNING_RATE = 0.0003;

    /// <summary>
    /// The step size below which the inner solve gives up rather than halving further.
    /// </summary>
    /// <remarks>
    /// A step this small moves the iterate by less than double precision can represent against the
    /// weights themselves, so continuing to halve only spins. Reaching it means no valid M-matrix
    /// step exists from the current point, and the solve stops there as unconverged.
    /// </remarks>
    private const double MIN_LEARNING_RATE = 1e-16;

    /// <summary>
    /// Adam optimizer beta1 (first moment decay).
    /// </summary>
    private const double ADAM_BETA1 = 0.99;

    /// <summary>
    /// Adam optimizer beta2 (second moment decay).
    /// </summary>
    private const double ADAM_BETA2 = 0.999;

    /// <summary>
    /// Number of warm-up inner iterations for non-final outer steps.
    /// </summary>
    private const int DEFAULT_WARM_ITER = 30000;

    /// <summary>
    /// Number of inner iterations for the final outer step.
    /// </summary>
    private const int DEFAULT_MAX_ITER = 60000;

    /// <summary>
    /// Convergence tolerance for the inner loop objective change.
    /// </summary>
    private const double INNER_CONVERGENCE_TOL = 1e-6;

    /// <summary>
    /// Checkpoint interval for convergence checking.
    /// </summary>
    private const int CHECKPOINT_INTERVAL = 1000;

    #endregion

    #region Fields

    private readonly double[] _sValues;
    private readonly int? _configuredMaxIterations;

    /// <summary>
    /// The caller's learning rate, never written after construction.
    /// </summary>
    /// <remarks>
    /// The step size DECAYS during a run: every M-matrix domain violation halves it. That decay has
    /// to live in a per-run local, not in this field. Held here, a second DiscoverStructure call on
    /// the same instance started from whatever the first call decayed it to -- possibly 1e-16 -- so
    /// the same model object returned a different graph for identical data.
    /// </remarks>
    private readonly double _configuredLearningRate;

    private int _lastIterations;
    private double _lastH;
    private double _lastLoss;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override string Name => "DAGMA Linear";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes DAGMA Linear with optional configuration.
    /// </summary>
    public DAGMALinear(CausalDiscoveryOptions? options = null)
    {
        Lambda1 = DAGMA_DEFAULT_LAMBDA1;
        ApplyOptions(options);

        _sValues = [1.0, 0.9, 0.8, 0.7, 0.6];
        _configuredMaxIterations = options?.MaxIterations;
        _configuredLearningRate = options?.LearningRate ?? DEFAULT_LEARNING_RATE;
        if (double.IsNaN(_configuredLearningRate) || double.IsInfinity(_configuredLearningRate)
            || _configuredLearningRate <= 0)
            throw new ArgumentException("LearningRate must be a positive finite value.", nameof(options));
    }

    #endregion

    #region Core Algorithm

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int d = data.Columns;

        // Initialize W = 0
        var W = new Matrix<T>(d, d);
        double mu = DEFAULT_MU_INIT;
        _lastIterations = 0;

        int T = Math.Min(DEFAULT_T, _sValues.Length);

        // Per-RUN step size. It decays across the outer loop as inner solves hit the M-matrix
        // boundary, which is intended within one run, and it starts from the configured value on
        // every call, which is what makes repeated calls on one instance reproducible.
        double learningRate = _configuredLearningRate;

        for (int t = 0; t < T; t++)
        {
            double s = _sValues[t];
            // DEVIATION, stated explicitly: CausalDiscoveryOptions.MaxIterations documents an OUTER
            // iteration budget, and the other continuous-optimization algorithms honour that. DAGMA
            // fixes its outer loop at DEFAULT_T = 5 central-path steps -- that count is the schedule
            // in _sValues, not a budget a caller can spend -- so the option is applied to the inner
            // Adam loop instead, as a CAP on the reference's own inner counts rather than a target.
            // Consequence to be aware of: MaxIterations = 100 permits up to 5 x 100 = 500 Adam steps
            // in total, not 100.
            int defaultInner = (t < T - 1) ? DEFAULT_WARM_ITER : DEFAULT_MAX_ITER;
            int maxInner = _configuredMaxIterations.HasValue
                ? Math.Min(defaultInner, _configuredMaxIterations.Value)
                : defaultInner;

            int actualIter;
            (W, actualIter) = SolveInnerProblem(data, W, mu, s, d, maxInner, ref learningRate);

            mu *= DEFAULT_MU_FACTOR;
            _lastIterations += actualIter;
        }

        // Compute final metrics
        var (finalLoss, _) = ComputeL2Loss(data, W);
        _lastLoss = finalLoss + Lambda1 * ComputeL1Norm(W);

        var (h, _) = ComputeLogDetConstraint(W, _sValues[^1], d);
        _lastH = h;

        // Threshold and clean
        return EnforceAcyclic(ThresholdAndClean(W, WThreshold));
    }

    /// <summary>
    /// Solves the inner optimization problem using Adam optimizer.
    /// Minimizes: score(W) + mu * h(W, s)
    /// </summary>
    private (Matrix<T> W, int ActualIterations) SolveInnerProblem(Matrix<T> X, Matrix<T> W, double mu, double s, int d, int maxIter, ref double learningRate)
    {
        var w = Tensor<T>.FromMatrix(W);

        // Adam state
        var m = new Tensor<T>(new[] { d, d }); // First moment
        var v = new Tensor<T>(new[] { d, d }); // Second moment
        var offDiagonal = OffDiagonalMask(d);

        double prevObj = double.MaxValue;
        int actualIter = 0;

        for (int iter = 1; iter <= maxIter; iter++)
        {
            // Objective, gradient and Adam moments at the current weights
            var (obj, direction) = ComputeDagmaAdamDirection(X, w, offDiagonal, mu, s, d, iter, ref m, ref v);

            // Adam update, then zero diagonal
            w = ApplyDagmaStep(w, direction, learningRate, offDiagonal);

            // M-matrix domain check (per DAGMA reference implementation):
            // Verify sI - W⊙W is still an M-matrix (inverse has no negative entries).
            // If violated, halve the learning rate and retry the step.
            if (!IsInMMatrixDomain(w, s, d))
            {
                // Stepped outside the M-matrix domain. RETRY UNTIL THE STEP IS VALID, rather than
                // halving once and hoping: the single replacement step was never itself checked, so
                // if it also landed outside the domain the next iteration computed a log-determinant
                // on an invalid matrix and fell through to the fallback gradient.
                //
                // Each attempt restarts from the last VALID weights, which is why the undo is inside
                // the loop -- undoing a step taken at one rate and redoing it at another only lands
                // back on the same point if the rate used for the undo is the rate that took it.
                bool recovered = false;
                double appliedRate = learningRate;

                while (learningRate >= MIN_LEARNING_RATE)
                {
                    // Undo whatever rate produced the current (invalid) position.
                    w = UndoDagmaStep(w, direction, appliedRate);

                    learningRate *= 0.5;
                    if (learningRate < MIN_LEARNING_RATE) break;

                    w = ApplyDagmaStep(w, direction, learningRate, offDiagonal);
                    appliedRate = learningRate;

                    if (IsInMMatrixDomain(w, s, d))
                    {
                        recovered = true;
                        break;
                    }
                }

                // The rate floor was reached with no valid step available. The last undo left w on
                // the previous valid iterate, so the solve stops there as unconverged rather than
                // continuing from a point outside the domain.
                if (!recovered) break;
            }

            // Convergence check at checkpoint intervals
            actualIter = iter;

            if (iter % CHECKPOINT_INTERVAL == 0)
            {
                if (Math.Abs(obj - prevObj) / (Math.Abs(prevObj) + 1e-15) < INNER_CONVERGENCE_TOL)
                {
                    break;
                }

                prevObj = obj;
            }
        }

        return (w.ToMatrix(), actualIter);
    }

    /// <summary>
    /// Evaluates the DAGMA objective and gradient at <paramref name="w"/>, advances the Adam moments, and
    /// returns the bias-corrected Adam direction mHat / (sqrt(vHat) + 1e-8).
    /// </summary>
    private (double Objective, (Tensor<T> MHat, Tensor<T> Denominator) Direction) ComputeDagmaAdamDirection(
        Matrix<T> X, Tensor<T> w, Tensor<T> offDiagonal, double mu, double s, int d, int iter,
        ref Tensor<T> m, ref Tensor<T> v)
    {
        var currentW = w.ToMatrix();

        // Compute gradients
        var (loss, lossGrad) = ComputeL2Loss(X, currentW);
        var (h, hGrad) = ComputeLogDetConstraint(currentW, s, d);

        // Per DAGMA reference: obj = mu*score + h (score weighted by mu, NOT h)
        double obj = mu * (loss + Lambda1 * ComputeL1Norm(currentW)) + h;

        // Per reference: grad = mu*G_score + mu*lambda1*sign(W) + G_h
        // (mu weights the score gradient, h gradient is unweighted; no L1 term on the diagonal)
        var grad = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(lossGrad), NumOps.FromDouble(mu)),
            Tensor<T>.FromMatrix(hGrad));
        var l1 = Engine.TensorMultiplyScalar(Engine.TensorSign(w), NumOps.FromDouble(mu * Lambda1));
        grad = Engine.TensorAdd(grad, Engine.TensorWhere(offDiagonal, l1, Engine.TensorZerosLike(w)));

        // Adam moments
        m = Engine.TensorAdd(Engine.TensorMultiplyScalar(m, NumOps.FromDouble(ADAM_BETA1)),
            Engine.TensorMultiplyScalar(grad, NumOps.FromDouble(1 - ADAM_BETA1)));
        v = Engine.TensorAdd(Engine.TensorMultiplyScalar(v, NumOps.FromDouble(ADAM_BETA2)),
            Engine.TensorMultiplyScalar(Engine.TensorSquare(grad), NumOps.FromDouble(1 - ADAM_BETA2)));

        var mHat = Engine.TensorDivideScalar(m, NumOps.FromDouble(1 - Math.Pow(ADAM_BETA1, iter)));
        var vHat = Engine.TensorDivideScalar(v, NumOps.FromDouble(1 - Math.Pow(ADAM_BETA2, iter)));
        var denominator = Engine.TensorAddScalar(Engine.TensorSqrt(vHat), NumOps.FromDouble(1e-8));
        return (obj, (mHat, denominator));
    }

    /// <summary>w - rate * mHat / (sqrt(vHat) + eps), with the diagonal set to zero.</summary>
    private Tensor<T> ApplyDagmaStep(Tensor<T> w, (Tensor<T> MHat, Tensor<T> Denominator) direction, double rate,
        Tensor<T> offDiagonal)
    {
        var step = Engine.TensorDivide(Engine.TensorMultiplyScalar(direction.MHat, NumOps.FromDouble(rate)),
            direction.Denominator);
        var moved = Engine.TensorSubtract(w, step);
        return Engine.TensorWhere(offDiagonal, moved, Engine.TensorZerosLike(moved));
    }

    /// <summary>w + rate * mHat / (sqrt(vHat) + eps) on every entry (the inverse of a step taken at that rate).</summary>
    private Tensor<T> UndoDagmaStep(Tensor<T> w, (Tensor<T> MHat, Tensor<T> Denominator) direction, double rate)
    {
        var step = Engine.TensorDivide(Engine.TensorMultiplyScalar(direction.MHat, NumOps.FromDouble(rate)),
            direction.Denominator);
        return Engine.TensorAdd(w, step);
    }

    /// <summary>
    /// Whether sI - W⊙W is an M-matrix: invertible with no inverse entry below -1e-10.
    /// </summary>
    private bool IsInMMatrixDomain(Tensor<T> w, double s, int d)
    {
        var inverse = InvertSmallMatrix(ShiftedSquareComplement(w, s, d).ToMatrix());
        return inverse is not null && !HasNegativeEntry(inverse);
    }

    /// <summary>sI - W∘W.</summary>
    private Tensor<T> ShiftedSquareComplement(Tensor<T> w, double s, int d)
        => Engine.TensorSubtract(Engine.TensorMultiplyScalar(Engine.TensorEye<T>(d), NumOps.FromDouble(s)),
            Engine.TensorMultiply(w, w));

    #endregion

    #region DAGMA Constraint

    /// <summary>
    /// Computes the DAGMA log-det acyclicity constraint and gradient.
    /// h(W, s) = -log det(sI - W∘W) + d*log(s)
    /// Gradient: dh/dW = 2 * W ∘ (sI - W∘W)^{-T}
    /// </summary>
    private (double H, Matrix<T> Gradient) ComputeLogDetConstraint(Matrix<T> W, double s, int d)
    {
        // M = sI - W∘W
        var w = Tensor<T>.FromMatrix(W);
        var M = ShiftedSquareComplement(w, s, d);

        // Log-determinant via LU decomposition
        double logDet = LogAbsDeterminant(M);

        // h = -logDet + d * log(s)
        double h = -logDet + d * Math.Log(s);

        // Gradient: 2 * W ∘ inv(M)^T
        var invM = InvertSmallMatrix(M.ToMatrix());
        T two = NumOps.FromDouble(2.0);
        Tensor<T> gradient;

        if (invM != null)
        {
            // Normal case: inversion succeeded
            // Note: for symmetric M, inv(M)^T = inv(M)
            gradient = Engine.TensorMultiplyScalar(
                Engine.TensorMultiply(w, Engine.TensorTranspose(Tensor<T>.FromMatrix(invM))), two);
        }
        else
        {
            // Singular matrix: use identity-scaled fallback gradient to avoid stalling
            System.Diagnostics.Trace.TraceWarning("DAGMA: M matrix is singular; using fallback gradient.");
            gradient = Engine.TensorMultiplyScalar(w, two);
        }

        return (h, gradient.ToMatrix());
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

    /// <summary>
    /// Checks if any entry of the matrix is negative (M-matrix violation).
    /// </summary>
    private bool HasNegativeEntry(Matrix<T> matrix)
    {
        var values = Tensor<T>.FromMatrix(matrix);
        return Engine.TensorCountNonzero(Engine.TensorLessThan(values, NumOps.FromDouble(-1e-10))) > 0;
    }

    #endregion
}
