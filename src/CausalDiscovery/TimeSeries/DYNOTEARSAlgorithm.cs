using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// DYNOTEARS — Dynamic NOTEARS for time series structure learning.
/// </summary>
/// <remarks>
/// <para>
/// DYNOTEARS extends the NOTEARS continuous optimization framework to time series data.
/// It jointly learns both contemporaneous (W) and lagged (A₁, ..., Aₖ) adjacency matrices
/// using an augmented Lagrangian with the acyclicity constraint only on the contemporaneous matrix W.
/// </para>
/// <para>
/// <b>Model:</b> X(t) = W^T X(t) + Σ_k A_k^T X(t-k) + e(t)
/// <b>Objective:</b> min_{W,A} ½n⁻¹ ||X_t - X_t W - Z A||²_F + λ₁(||W||₁ + ||A||₁)
/// <b>Constraint:</b> h(W) = tr(e^(W∘W)) - d = 0 (acyclicity only on contemporaneous W)
/// </para>
/// <para>
/// <b>For Beginners:</b> DYNOTEARS is like NOTEARS but for time series. It can learn
/// both "X and Y affect each other at the same time" and "yesterday's X affects today's Y"
/// type relationships simultaneously, using the same elegant continuous optimization approach.
/// The key insight is that only the contemporaneous matrix W needs to be acyclic — lagged
/// effects can't create instantaneous cycles.
/// </para>
/// <para>
/// Reference: Pamfil et al. (2020), "DYNOTEARS: Structure Learning from Time-Series Data", AISTATS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DYNOTEARSAlgorithm<T> : TimeSeriesCausalBase<T>
{
    private double _lambda1 = 0.1;
    private double _wThreshold = 0.3;
    private int _maxIterations = 100;
    private double _hTol = 1e-8;

    /// <inheritdoc/>
    public override string Name => "DYNOTEARS";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public DYNOTEARSAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        if (options?.SparsityPenalty.HasValue == true) _lambda1 = options.SparsityPenalty.Value;
        if (options?.EdgeThreshold.HasValue == true) _wThreshold = options.EdgeThreshold.Value;
        if (options?.MaxIterations.HasValue == true) _maxIterations = options.MaxIterations.Value;
        if (options?.AcyclicityTolerance.HasValue == true) _hTol = options.AcyclicityTolerance.Value;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n <= MaxLag + 1) return DoubleArrayToMatrix(new double[d, d]);

        int effectiveN = n - MaxLag;

        // Convert to double array
        var rawData = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                rawData[i, j] = NumOps.ToDouble(data[i, j]);

        // Build X_t: contemporaneous data matrix [effectiveN x d]
        var Xt = new double[effectiveN, d];
        for (int t = 0; t < effectiveN; t++)
            for (int j = 0; j < d; j++)
                Xt[t, j] = rawData[t + MaxLag, j];

        // Build Z: lagged data matrix [effectiveN x (d * MaxLag)]
        // Z = [X(t-1), X(t-2), ..., X(t-MaxLag)]
        int lagDim = d * MaxLag;
        var Z = new double[effectiveN, lagDim];
        for (int t = 0; t < effectiveN; t++)
        {
            for (int lag = 1; lag <= MaxLag; lag++)
            {
                int colOffset = (lag - 1) * d;
                for (int j = 0; j < d; j++)
                    Z[t, colOffset + j] = rawData[t + MaxLag - lag, j];
            }
        }

        // Joint optimization of W (d x d) and A (lagDim x d) using augmented Lagrangian
        var W = new double[d, d]; // contemporaneous — must be DAG
        var A = new double[lagDim, d]; // lagged — no acyclicity constraint

        // Augmented Lagrangian parameters
        double rho = 1.0;
        double alpha = 0.0; // Lagrange multiplier
        double rhoMax = 1e+16;
        double gammaRho = 10.0; // rho increase factor
        double hPrev = double.PositiveInfinity;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Inner optimization: minimize L(W, A) = loss + lambda1*sparsity + alpha*h(W) + rho/2 * h(W)^2
            // Use gradient descent on the joint (W, A) parameter space
            OptimizeInner(Xt, Z, W, A, effectiveN, d, lagDim, rho, alpha);

            // Evaluate acyclicity constraint on W: h(W) = tr(e^{W∘W}) - d
            double h = ComputeAcyclicity(W, d);

            if (h < _hTol)
                break; // converged to a DAG

            // Update augmented Lagrangian parameters
            if (h > 0.25 * hPrev)
            {
                rho = Math.Min(rho * gammaRho, rhoMax);
            }
            alpha += rho * h;
            hPrev = h;
        }

        // Threshold W and combine with A to produce summary adjacency
        var result = new double[d, d];

        // Contemporaneous edges (from W)
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (Math.Abs(W[i, j]) >= _wThreshold)
                    result[i, j] = W[i, j];

        // Lagged edges: aggregate across lags into summary d x d matrix
        for (int lag = 1; lag <= MaxLag; lag++)
        {
            int colOffset = (lag - 1) * d;
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    double lagWeight = Math.Abs(A[colOffset + i, j]);
                    if (lagWeight >= _wThreshold)
                    {
                        // Take the maximum absolute lagged effect across lags
                        double current = Math.Abs(result[i, j]);
                        if (lagWeight > current)
                            result[i, j] = A[colOffset + i, j];
                    }
                }
            }
        }

        return DoubleArrayToMatrix(result);
    }

    /// <summary>
    /// Inner optimization loop: gradient descent on the joint (W, A) objective.
    /// L(W,A) = ½n⁻¹||Xt - Xt*W - Z*A||²_F + λ₁(||W||₁ + ||A||₁) + α*h(W) + ρ/2 * h(W)²
    /// </summary>
    private void OptimizeInner(double[,] Xt, double[,] Z,
        double[,] W, double[,] A,
        int n, int d, int lagDim,
        double rho, double alpha)
    {
        double learningRate = 1e-3;
        int innerSteps = 200;
        double prevLoss = double.PositiveInfinity;

        for (int step = 0; step < innerSteps; step++)
        {
            // Compute residual: R = Xt - Xt*W - Z*A  [n x d]
            var R = new double[n, d];
            for (int t = 0; t < n; t++)
            {
                for (int j = 0; j < d; j++)
                {
                    double val = Xt[t, j];
                    for (int k = 0; k < d; k++)
                        val -= Xt[t, k] * W[k, j];
                    for (int k = 0; k < lagDim; k++)
                        val -= Z[t, k] * A[k, j];
                    R[t, j] = val;
                }
            }

            // Gradient of loss w.r.t. W: -1/n * Xt' * R
            var gradW = new double[d, d];
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    double sum = 0;
                    for (int t = 0; t < n; t++)
                        sum += Xt[t, i] * R[t, j];
                    gradW[i, j] = -sum / n;
                }

            // Gradient of loss w.r.t. A: -1/n * Z' * R
            var gradA = new double[lagDim, d];
            for (int i = 0; i < lagDim; i++)
                for (int j = 0; j < d; j++)
                {
                    double sum = 0;
                    for (int t = 0; t < n; t++)
                        sum += Z[t, i] * R[t, j];
                    gradA[i, j] = -sum / n;
                }

            // Gradient of acyclicity constraint w.r.t. W: 2 * (e^{W∘W}) ∘ W
            double h = ComputeAcyclicity(W, d);
            var expWoW = ComputeMatrixExponentialOfHadamard(W, d);
            var gradH = new double[d, d];
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    gradH[i, j] = 2.0 * expWoW[i, j] * W[i, j];

            // Combined gradient for W: grad_loss + lambda1*sign(W) + (alpha + rho*h) * grad_h
            double constraintMult = alpha + rho * h;
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    double l1Grad = _lambda1 * Math.Sign(W[i, j]);
                    gradW[i, j] += l1Grad + constraintMult * gradH[i, j];
                }

            // Combined gradient for A: grad_loss + lambda1*sign(A)
            for (int i = 0; i < lagDim; i++)
                for (int j = 0; j < d; j++)
                {
                    double l1Grad = _lambda1 * Math.Sign(A[i, j]);
                    gradA[i, j] += l1Grad;
                }

            // Gradient descent update
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    W[i, j] -= learningRate * gradW[i, j];

            for (int i = 0; i < lagDim; i++)
                for (int j = 0; j < d; j++)
                    A[i, j] -= learningRate * gradA[i, j];

            // Zero diagonal of W (no self-loops)
            for (int i = 0; i < d; i++)
                W[i, i] = 0;

            // Convergence check every 20 steps
            if (step % 20 == 0)
            {
                double loss = ComputeLoss(R, n, d);
                if (Math.Abs(loss - prevLoss) < 1e-8 * (1 + Math.Abs(prevLoss)))
                    break;
                prevLoss = loss;
            }
        }
    }

    /// <summary>
    /// Computes h(W) = tr(e^{W∘W}) - d, the NOTEARS acyclicity constraint.
    /// Uses Taylor series approximation: e^M ≈ I + M + M²/2! + M³/3! + ...
    /// </summary>
    private static double ComputeAcyclicity(double[,] W, int d)
    {
        // W∘W (Hadamard square)
        var WoW = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                WoW[i, j] = W[i, j] * W[i, j];

        // Compute trace of matrix exponential via Taylor series
        // tr(e^M) = d + tr(M) + tr(M²)/2! + tr(M³)/3! + ...
        double trace = d; // tr(I)
        var power = new double[d, d]; // M^k
        // Initialize power = I
        for (int i = 0; i < d; i++) power[i, i] = 1.0;

        double factorial = 1.0;
        for (int k = 1; k <= Math.Min(d, 20); k++)
        {
            factorial *= k;
            // power = power * WoW
            var newPower = new double[d, d];
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    double sum = 0;
                    for (int m = 0; m < d; m++)
                        sum += power[i, m] * WoW[m, j];
                    newPower[i, j] = sum;
                }
            power = newPower;

            // Add tr(M^k) / k!
            double tr = 0;
            for (int i = 0; i < d; i++) tr += power[i, i];
            trace += tr / factorial;
        }

        return trace - d;
    }

    /// <summary>
    /// Computes e^{W∘W} via Taylor series (needed for gradient of h(W)).
    /// </summary>
    private static double[,] ComputeMatrixExponentialOfHadamard(double[,] W, int d)
    {
        var WoW = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                WoW[i, j] = W[i, j] * W[i, j];

        // e^M via Taylor series
        var result = new double[d, d];
        var power = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            result[i, i] = 1.0; // I
            power[i, i] = 1.0;
        }

        double factorial = 1.0;
        for (int k = 1; k <= Math.Min(d, 20); k++)
        {
            factorial *= k;
            var newPower = new double[d, d];
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    double sum = 0;
                    for (int m = 0; m < d; m++)
                        sum += power[i, m] * WoW[m, j];
                    newPower[i, j] = sum;
                }
            power = newPower;

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    result[i, j] += power[i, j] / factorial;
        }

        return result;
    }

    private static double ComputeLoss(double[,] R, int n, int d)
    {
        double sum = 0;
        for (int t = 0; t < n; t++)
            for (int j = 0; j < d; j++)
                sum += R[t, j] * R[t, j];
        return sum / (2.0 * n);
    }
}
