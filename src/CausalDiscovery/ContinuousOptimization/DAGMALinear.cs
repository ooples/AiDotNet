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
    private double _learningRate;
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
        _learningRate = DEFAULT_LEARNING_RATE;
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

        for (int t = 0; t < T; t++)
        {
            double s = _sValues[t];
            int maxInner = (t < T - 1) ? DEFAULT_WARM_ITER : DEFAULT_MAX_ITER;

            int actualIter;
            (W, actualIter) = SolveInnerProblem(data, W, mu, s, d, maxInner);

            mu *= DEFAULT_MU_FACTOR;
            _lastIterations += actualIter;
        }

        // Compute final metrics
        var (finalLoss, _) = ComputeL2Loss(data, W);
        _lastLoss = finalLoss + Lambda1 * ComputeL1Norm(W);

        var (h, _) = ComputeLogDetConstraint(W, _sValues[^1], d);
        _lastH = h;

        // Threshold and clean
        return ThresholdAndClean(W, WThreshold);
    }

    /// <summary>
    /// Solves the inner optimization problem using Adam optimizer.
    /// Minimizes: score(W) + mu * h(W, s)
    /// </summary>
    private (Matrix<T> W, int ActualIterations) SolveInnerProblem(Matrix<T> X, Matrix<T> W, double mu, double s, int d, int maxIter)
    {
        // Flatten W to double[] for Adam state management
        int vecLen = d * d;
        double[] w = FlattenMatrix(W, d);

        // Adam state
        double[] m = new double[vecLen]; // First moment
        double[] v = new double[vecLen]; // Second moment

        double prevObj = double.MaxValue;
        int actualIter = 0;

        for (int iter = 1; iter <= maxIter; iter++)
        {
            var currentW = UnflattenMatrix(w, d);

            // Compute gradients
            var (loss, lossGrad) = ComputeL2Loss(X, currentW);
            var (h, hGrad) = ComputeLogDetConstraint(currentW, s, d);

            double obj = loss + Lambda1 * ComputeL1Norm(currentW) + mu * h;

            // Full gradient = loss_grad + mu * h_grad + L1 subgradient
            double[] grad = new double[vecLen];
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    int idx = i * d + j;
                    grad[idx] = NumOps.ToDouble(lossGrad[i, j]) + mu * NumOps.ToDouble(hGrad[i, j]);

                    if (i != j)
                    {
                        grad[idx] += Lambda1 * Math.Sign(w[idx]);
                    }
                }
            }

            // Adam update
            for (int i = 0; i < vecLen; i++)
            {
                m[i] = ADAM_BETA1 * m[i] + (1 - ADAM_BETA1) * grad[i];
                v[i] = ADAM_BETA2 * v[i] + (1 - ADAM_BETA2) * grad[i] * grad[i];

                double mHat = m[i] / (1 - Math.Pow(ADAM_BETA1, iter));
                double vHat = v[i] / (1 - Math.Pow(ADAM_BETA2, iter));

                w[i] -= _learningRate * mHat / (Math.Sqrt(vHat) + 1e-8);
            }

            // Zero diagonal
            for (int i = 0; i < d; i++)
            {
                w[i * d + i] = 0;
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

        return (UnflattenMatrix(w, d), actualIter);
    }

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
        var M = new Matrix<T>(d, d);
        T sT = NumOps.FromDouble(s);
        for (int i = 0; i < d; i++)
        {
            M[i, i] = sT;
            for (int j = 0; j < d; j++)
                M[i, j] = NumOps.Subtract(M[i, j], NumOps.Multiply(W[i, j], W[i, j]));
        }

        // Log-determinant via LU decomposition
        double logDet = ComputeLogDeterminant(M, d);

        // h = -logDet + d * log(s)
        double h = -logDet + d * Math.Log(s);

        // Gradient: 2 * W ∘ inv(M)^T
        var invM = InvertMatrix(M, d);
        var gradient = new Matrix<T>(d, d);

        if (invM != null)
        {
            // Normal case: inversion succeeded
            T two = NumOps.FromDouble(2.0);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    // Note: for symmetric M, inv(M)^T = inv(M)
                    gradient[i, j] = NumOps.Multiply(two, NumOps.Multiply(W[i, j], invM[j, i]));
        }
        else
        {
            // Singular matrix: use identity-scaled fallback gradient to avoid stalling
            System.Diagnostics.Trace.TraceWarning("DAGMA: M matrix is singular; using fallback gradient.");
            T two = NumOps.FromDouble(2.0);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    gradient[i, j] = NumOps.Multiply(two, W[i, j]);
        }

        return (h, gradient);
    }

    /// <summary>
    /// Computes log-determinant using LU decomposition.
    /// </summary>
    private double ComputeLogDeterminant(Matrix<T> matrix, int d)
    {
        // LU decomposition (partial pivoting) on a copy
        var LU = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                LU[i, j] = matrix[i, j];

        for (int k = 0; k < d; k++)
        {
            // Find pivot
            int maxRow = k;
            for (int i = k + 1; i < d; i++)
                if (Math.Abs(NumOps.ToDouble(LU[i, k])) > Math.Abs(NumOps.ToDouble(LU[maxRow, k])))
                    maxRow = i;

            if (maxRow != k)
                for (int j = 0; j < d; j++)
                    (LU[k, j], LU[maxRow, j]) = (LU[maxRow, j], LU[k, j]);

            if (Math.Abs(NumOps.ToDouble(LU[k, k])) < 1e-15)
                return double.NegativeInfinity; // Singular

            for (int i = k + 1; i < d; i++)
            {
                LU[i, k] = NumOps.Divide(LU[i, k], LU[k, k]);
                for (int j = k + 1; j < d; j++)
                    LU[i, j] = NumOps.Subtract(LU[i, j], NumOps.Multiply(LU[i, k], LU[k, j]));
            }
        }

        double logDet = 0;
        for (int i = 0; i < d; i++)
            logDet += Math.Log(Math.Abs(NumOps.ToDouble(LU[i, i])));
        return logDet;
    }

    /// <summary>
    /// Inverts a matrix using Gauss-Jordan elimination. Returns null if singular.
    /// </summary>
    private Matrix<T>? InvertMatrix(Matrix<T> matrix, int d)
    {
        var aug = new Matrix<T>(d, 2 * d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
                aug[i, j] = matrix[i, j];
            aug[i, i + d] = NumOps.One;
        }

        for (int col = 0; col < d; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < d; row++)
                if (Math.Abs(NumOps.ToDouble(aug[row, col])) > Math.Abs(NumOps.ToDouble(aug[maxRow, col])))
                    maxRow = row;

            if (Math.Abs(NumOps.ToDouble(aug[maxRow, col])) < 1e-12)
                return null;

            if (maxRow != col)
                for (int j = 0; j < 2 * d; j++)
                    (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            T pivot = aug[col, col];
            for (int j = 0; j < 2 * d; j++)
                aug[col, j] = NumOps.Divide(aug[col, j], pivot);

            for (int row = 0; row < d; row++)
            {
                if (row != col)
                {
                    T factor = aug[row, col];
                    for (int j = 0; j < 2 * d; j++)
                        aug[row, j] = NumOps.Subtract(aug[row, j], NumOps.Multiply(factor, aug[col, j]));
                }
            }
        }

        var result = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                result[i, j] = aug[i, j + d];
        return result;
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

    private double[] FlattenMatrix(Matrix<T> matrix, int d)
    {
        var result = new double[d * d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                result[i * d + j] = NumOps.ToDouble(matrix[i, j]);
        return result;
    }

    private Matrix<T> UnflattenMatrix(double[] vector, int d)
    {
        var result = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                result[i, j] = NumOps.FromDouble(vector[i * d + j]);
        return result;
    }

    #endregion
}
