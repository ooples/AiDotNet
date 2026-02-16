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
        var W = new Matrix<T>(d, d);

        // Adam state (flat double[] for optimizer internals)
        int vecLen = d * d;
        var m = new double[vecLen];
        var v = new double[vecLen];

        double prevObj = double.MaxValue;
        _lastIterations = 0;

        for (int iter = 1; iter <= DEFAULT_NUM_ITERATIONS; iter++)
        {
            _lastIterations = iter;

            // Compute objective and gradient
            var (obj, grad) = _equalVariance
                ? ComputeGOLEMEV(data, W, n, d)
                : ComputeGOLEMNV(data, W, n, d);

            // Add L1 subgradient and DAG penalty gradient
            var (dagPenalty, dagGrad) = ComputeDAGPenalty(W, d);
            obj += _lambda2 * dagPenalty;

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    int idx = i * d + j;
                    double g = NumOps.ToDouble(grad[i, j]) + _lambda2 * NumOps.ToDouble(dagGrad[i, j]);
                    if (i != j) g += Lambda1 * Math.Sign(NumOps.ToDouble(W[i, j]));

                    // Adam update
                    m[idx] = ADAM_BETA1 * m[idx] + (1 - ADAM_BETA1) * g;
                    v[idx] = ADAM_BETA2 * v[idx] + (1 - ADAM_BETA2) * g * g;

                    double mHat = m[idx] / (1 - Math.Pow(ADAM_BETA1, iter));
                    double vHat = v[idx] / (1 - Math.Pow(ADAM_BETA2, iter));

                    double wij = NumOps.ToDouble(W[i, j]);
                    wij -= DEFAULT_LEARNING_RATE * mHat / (Math.Sqrt(vHat) + 1e-8);
                    W[i, j] = NumOps.FromDouble(wij);
                }
            }

            // Zero diagonal
            for (int i = 0; i < d; i++)
                W[i, i] = NumOps.Zero;

            // Convergence check
            if (iter % CHECKPOINT_INTERVAL == 0)
            {
                if (Math.Abs(obj - prevObj) / (Math.Abs(prevObj) + 1e-15) < CONVERGENCE_TOL)
                    break;
                prevObj = obj;
            }
        }

        // Compute final metrics
        var (finalLoss, _) = ComputeL2Loss(data, W);
        _lastLoss = finalLoss;

        var (hFinal, _) = ComputeNOTEARSConstraint(W);
        _lastH = hFinal;

        return ThresholdAndClean(W, WThreshold);
    }

    #endregion

    #region GOLEM Score Functions

    /// <summary>
    /// GOLEM-EV score: equal variance assumption.
    /// Score = n*d/2 * log(||X - XW||²_F / n) - log|det(I - W)|
    /// Gradient computed analytically.
    /// </summary>
    private (double Score, Matrix<T> Gradient) ComputeGOLEMEV(Matrix<T> X, Matrix<T> W, int n, int d)
    {
        // Compute residual R = X - XW
        var R = new Matrix<T>(n, d);
        double rss = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                T xw = NumOps.Zero;
                for (int k = 0; k < d; k++)
                    xw = NumOps.Add(xw, NumOps.Multiply(X[i, k], W[k, j]));
                R[i, j] = NumOps.Subtract(X[i, j], xw);
                double r = NumOps.ToDouble(R[i, j]);
                rss += r * r;
            }
        }

        double sigmaSquared = rss / (n * d);
        if (sigmaSquared < 1e-15) sigmaSquared = 1e-15;

        // I - W
        var ImW = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            ImW[i, i] = NumOps.One;
            for (int j = 0; j < d; j++)
                ImW[i, j] = NumOps.Subtract(ImW[i, j], W[i, j]);
        }

        double logDetImW = ComputeLogAbsDeterminant(ImW, d);

        // Score = (n*d/2) * log(sigma^2) - n * log|det(I-W)|
        double score = (n * d / 2.0) * Math.Log(sigmaSquared) - n * logDetImW;

        // Gradient
        var grad = new Matrix<T>(d, d);

        // X^T R term: d/dW [(n*d/2)*log(sigma^2)] = -X^T R / sigma^2
        for (int k = 0; k < d; k++)
        {
            for (int j = 0; j < d; j++)
            {
                double xtr = 0;
                for (int i = 0; i < n; i++)
                    xtr += NumOps.ToDouble(X[i, k]) * NumOps.ToDouble(R[i, j]);
                grad[k, j] = NumOps.FromDouble(-xtr / sigmaSquared);
            }
        }

        // Log-det term: d/dW [-n * log|det(I-W)|] = n * (I-W)^{-T}
        var invImW = InvertMatrix(ImW, d);
        if (invImW != null)
        {
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    grad[i, j] = NumOps.Add(grad[i, j], NumOps.FromDouble(n * NumOps.ToDouble(invImW[j, i])));
        }

        return (score, grad);
    }

    /// <summary>
    /// GOLEM-NV score: non-equal variance assumption.
    /// Score = sum_j [n/2 * log(||R_j||^2 / n)] - n * log|det(I - W)|
    /// </summary>
    private (double Score, Matrix<T> Gradient) ComputeGOLEMNV(Matrix<T> X, Matrix<T> W, int n, int d)
    {
        var R = new Matrix<T>(n, d);
        var rssPerVar = new double[d];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                T xw = NumOps.Zero;
                for (int k = 0; k < d; k++)
                    xw = NumOps.Add(xw, NumOps.Multiply(X[i, k], W[k, j]));
                R[i, j] = NumOps.Subtract(X[i, j], xw);
                double r = NumOps.ToDouble(R[i, j]);
                rssPerVar[j] += r * r;
            }
        }

        // I - W
        var ImW = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            ImW[i, i] = NumOps.One;
            for (int j = 0; j < d; j++)
                ImW[i, j] = NumOps.Subtract(ImW[i, j], W[i, j]);
        }

        double logDetImW = ComputeLogAbsDeterminant(ImW, d);

        // Guard against singular I-W: return large penalty to steer optimization away
        if (double.IsNegativeInfinity(logDetImW) || double.IsNaN(logDetImW))
        {
            var zeroGrad = new Matrix<T>(d, d);
            return (double.MaxValue / 2, zeroGrad);
        }

        double score = -n * logDetImW;
        for (int j = 0; j < d; j++)
        {
            double sigmaJ = rssPerVar[j] / n;
            if (sigmaJ < 1e-15) sigmaJ = 1e-15;
            score += (n / 2.0) * Math.Log(sigmaJ);
        }

        // Gradient
        var grad = new Matrix<T>(d, d);
        for (int k = 0; k < d; k++)
        {
            for (int j = 0; j < d; j++)
            {
                double sigmaJ = rssPerVar[j] / n;
                if (sigmaJ < 1e-15) sigmaJ = 1e-15;

                double xtr = 0;
                for (int i = 0; i < n; i++)
                    xtr += NumOps.ToDouble(X[i, k]) * NumOps.ToDouble(R[i, j]);
                grad[k, j] = NumOps.FromDouble(-xtr / rssPerVar[j]);
            }
        }

        var invImW = InvertMatrix(ImW, d);
        if (invImW != null)
        {
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    grad[i, j] = NumOps.Add(grad[i, j], NumOps.FromDouble(n * NumOps.ToDouble(invImW[j, i])));
        }

        return (score, grad);
    }

    /// <summary>
    /// Computes a soft DAG penalty: h(W) = tr(e^(W∘W)) - d.
    /// </summary>
    private (double Penalty, Matrix<T> Gradient) ComputeDAGPenalty(Matrix<T> W, int d)
    {
        return ComputeNOTEARSConstraint(W);
    }

    #endregion

    #region Matrix Utilities

    private double ComputeLogAbsDeterminant(Matrix<T> matrix, int d)
    {
        // LU decomposition (partial pivoting) on a copy
        var LU = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                LU[i, j] = matrix[i, j];

        for (int k = 0; k < d; k++)
        {
            int maxRow = k;
            for (int i = k + 1; i < d; i++)
                if (Math.Abs(NumOps.ToDouble(LU[i, k])) > Math.Abs(NumOps.ToDouble(LU[maxRow, k])))
                    maxRow = i;

            if (maxRow != k)
                for (int j = 0; j < d; j++)
                    (LU[k, j], LU[maxRow, j]) = (LU[maxRow, j], LU[k, j]);

            if (Math.Abs(NumOps.ToDouble(LU[k, k])) < 1e-15)
                return double.NegativeInfinity;

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

    private Matrix<T>? InvertMatrix(Matrix<T> matrix, int d)
    {
        var aug = new Matrix<T>(d, 2 * d);
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++) aug[i, j] = matrix[i, j];
            aug[i, i + d] = NumOps.One;
        }

        for (int col = 0; col < d; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < d; row++)
                if (Math.Abs(NumOps.ToDouble(aug[row, col])) > Math.Abs(NumOps.ToDouble(aug[maxRow, col])))
                    maxRow = row;

            if (Math.Abs(NumOps.ToDouble(aug[maxRow, col])) < 1e-12) return null;

            if (maxRow != col)
                for (int j = 0; j < 2 * d; j++)
                    (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            T pivot = aug[col, col];
            for (int j = 0; j < 2 * d; j++) aug[col, j] = NumOps.Divide(aug[col, j], pivot);

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

    #endregion
}
