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
        double[,] X = MatrixToDoubleArray(data);
        int n = data.Rows;
        int d = data.Columns;

        // Initialize W = 0
        var W = new double[d, d];

        // Adam state
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
                ? ComputeGOLEMEV(X, W, n, d)
                : ComputeGOLEMNV(X, W, n, d);

            // Add L1 subgradient and DAG penalty gradient
            var (dagPenalty, dagGrad) = ComputeDAGPenalty(W, d);
            obj += _lambda2 * dagPenalty;

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    int idx = i * d + j;
                    double g = grad[i, j] + _lambda2 * dagGrad[i, j];
                    if (i != j) g += Lambda1 * Math.Sign(W[i, j]);

                    // Adam update
                    m[idx] = ADAM_BETA1 * m[idx] + (1 - ADAM_BETA1) * g;
                    v[idx] = ADAM_BETA2 * v[idx] + (1 - ADAM_BETA2) * g * g;

                    double mHat = m[idx] / (1 - Math.Pow(ADAM_BETA1, iter));
                    double vHat = v[idx] / (1 - Math.Pow(ADAM_BETA2, iter));

                    W[i, j] -= DEFAULT_LEARNING_RATE * mHat / (Math.Sqrt(vHat) + 1e-8);
                }
            }

            // Zero diagonal
            for (int i = 0; i < d; i++)
                W[i, i] = 0;

            // Convergence check
            if (iter % CHECKPOINT_INTERVAL == 0)
            {
                if (Math.Abs(obj - prevObj) / (Math.Abs(prevObj) + 1e-15) < CONVERGENCE_TOL)
                    break;
                prevObj = obj;
            }
        }

        // Compute final metrics
        var (finalLoss, _) = ComputeL2Loss(X, W);
        _lastLoss = finalLoss;

        var (hFinal, _) = ComputeNOTEARSConstraint(W);
        _lastH = hFinal;

        var WThresholded = ThresholdAndClean(W, WThreshold);
        return DoubleArrayToMatrix(WThresholded);
    }

    #endregion

    #region GOLEM Score Functions

    /// <summary>
    /// GOLEM-EV score: equal variance assumption.
    /// Score = n*d/2 * log(||X - XW||²_F / n) - log|det(I - W)|
    /// Gradient computed analytically.
    /// </summary>
    private (double Score, double[,] Gradient) ComputeGOLEMEV(double[,] X, double[,] W, int n, int d)
    {
        // Compute residual R = X - XW
        var R = new double[n, d];
        double rss = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double xw = 0;
                for (int k = 0; k < d; k++)
                    xw += X[i, k] * W[k, j];
                R[i, j] = X[i, j] - xw;
                rss += R[i, j] * R[i, j];
            }
        }

        double sigmaSquared = rss / (n * d);
        if (sigmaSquared < 1e-15) sigmaSquared = 1e-15;

        // I - W
        var ImW = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            ImW[i, i] = 1.0;
            for (int j = 0; j < d; j++)
                ImW[i, j] -= W[i, j];
        }

        double logDetImW = ComputeLogAbsDeterminant(ImW, d);

        // Score = (n*d/2) * log(sigma^2) - n * log|det(I-W)|
        double score = (n * d / 2.0) * Math.Log(sigmaSquared) - n * logDetImW;

        // Gradient of score w.r.t. W
        // d/dW [log(sigma^2)] part: -2 * X^T R / (n * sigma^2 * d)  * (n*d/2)
        //                          = -X^T R / (sigma^2 * n)  * d/2... simplified:
        // d/dW [(n*d/2)*log(sigma^2)] = (n*d/2) * (1/sigma^2) * d(sigma^2)/dW
        //   where d(sigma^2)/dW = -2/(n*d) * X^T R
        // = (n*d/2) * (1/sigma^2) * (-2/(n*d)) * X^T R = -X^T R / sigma^2

        var grad = new double[d, d];

        // X^T R term
        for (int k = 0; k < d; k++)
        {
            for (int j = 0; j < d; j++)
            {
                double xtr = 0;
                for (int i = 0; i < n; i++)
                    xtr += X[i, k] * R[i, j];
                grad[k, j] = -xtr / sigmaSquared;
            }
        }

        // Log-det term: d/dW [-n * log|det(I-W)|] = n * (I-W)^{-T}
        var invImW = InvertMatrix(ImW, d);
        if (invImW != null)
        {
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    grad[i, j] += n * invImW[j, i]; // transpose of inverse
        }

        return (score, grad);
    }

    /// <summary>
    /// GOLEM-NV score: non-equal variance assumption.
    /// Score = sum_j [n/2 * log(||R_j||^2 / n)] - n * log|det(I - W)|
    /// </summary>
    private (double Score, double[,] Gradient) ComputeGOLEMNV(double[,] X, double[,] W, int n, int d)
    {
        var R = new double[n, d];
        var rssPerVar = new double[d];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double xw = 0;
                for (int k = 0; k < d; k++)
                    xw += X[i, k] * W[k, j];
                R[i, j] = X[i, j] - xw;
                rssPerVar[j] += R[i, j] * R[i, j];
            }
        }

        // I - W
        var ImW = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            ImW[i, i] = 1.0;
            for (int j = 0; j < d; j++)
                ImW[i, j] -= W[i, j];
        }

        double logDetImW = ComputeLogAbsDeterminant(ImW, d);

        double score = -n * logDetImW;
        for (int j = 0; j < d; j++)
        {
            double sigmaJ = rssPerVar[j] / n;
            if (sigmaJ < 1e-15) sigmaJ = 1e-15;
            score += (n / 2.0) * Math.Log(sigmaJ);
        }

        // Gradient
        var grad = new double[d, d];
        for (int k = 0; k < d; k++)
        {
            for (int j = 0; j < d; j++)
            {
                double sigmaJ = rssPerVar[j] / n;
                if (sigmaJ < 1e-15) sigmaJ = 1e-15;

                double xtr = 0;
                for (int i = 0; i < n; i++)
                    xtr += X[i, k] * R[i, j];
                grad[k, j] = -xtr / (sigmaJ * n) * (n / 2.0) * (2.0 / n);
                // Simplified: -xtr / (rssPerVar[j])
            }
        }

        var invImW = InvertMatrix(ImW, d);
        if (invImW != null)
        {
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    grad[i, j] += n * invImW[j, i];
        }

        return (score, grad);
    }

    /// <summary>
    /// Computes a soft DAG penalty: h(W) = tr(e^(W∘W)) - d.
    /// </summary>
    private (double Penalty, double[,] Gradient) ComputeDAGPenalty(double[,] W, int d)
    {
        return ComputeNOTEARSConstraint(W);
    }

    #endregion

    #region Matrix Utilities

    private static double ComputeLogAbsDeterminant(double[,] matrix, int d)
    {
        var LU = (double[,])matrix.Clone();
        for (int k = 0; k < d; k++)
        {
            int maxRow = k;
            for (int i = k + 1; i < d; i++)
                if (Math.Abs(LU[i, k]) > Math.Abs(LU[maxRow, k]))
                    maxRow = i;

            if (maxRow != k)
                for (int j = 0; j < d; j++)
                    (LU[k, j], LU[maxRow, j]) = (LU[maxRow, j], LU[k, j]);

            if (Math.Abs(LU[k, k]) < 1e-15)
                return double.NegativeInfinity;

            for (int i = k + 1; i < d; i++)
            {
                LU[i, k] /= LU[k, k];
                for (int j = k + 1; j < d; j++)
                    LU[i, j] -= LU[i, k] * LU[k, j];
            }
        }

        double logDet = 0;
        for (int i = 0; i < d; i++)
            logDet += Math.Log(Math.Abs(LU[i, i]));
        return logDet;
    }

    private static double[,]? InvertMatrix(double[,] matrix, int d)
    {
        var aug = new double[d, 2 * d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++) aug[i, j] = matrix[i, j];
            aug[i, i + d] = 1.0;
        }

        for (int col = 0; col < d; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < d; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;

            if (Math.Abs(aug[maxRow, col]) < 1e-12) return null;

            if (maxRow != col)
                for (int j = 0; j < 2 * d; j++)
                    (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            double pivot = aug[col, col];
            for (int j = 0; j < 2 * d; j++) aug[col, j] /= pivot;

            for (int row = 0; row < d; row++)
            {
                if (row != col)
                {
                    double factor = aug[row, col];
                    for (int j = 0; j < 2 * d; j++)
                        aug[row, j] -= factor * aug[col, j];
                }
            }
        }

        var result = new double[d, d];
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
