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

        if (n <= MaxLag + 1) return new Matrix<T>(d, d);

        int effectiveN = n - MaxLag;

        // Build X_t: contemporaneous data matrix [effectiveN x d]
        var Xt = new Matrix<T>(effectiveN, d);
        for (int t = 0; t < effectiveN; t++)
            for (int j = 0; j < d; j++)
                Xt[t, j] = data[t + MaxLag, j];

        // Build Z: lagged data matrix [effectiveN x (d * MaxLag)]
        // Z = [X(t-1), X(t-2), ..., X(t-MaxLag)]
        int lagDim = d * MaxLag;
        var Z = new Matrix<T>(effectiveN, lagDim);
        for (int t = 0; t < effectiveN; t++)
        {
            for (int lag = 1; lag <= MaxLag; lag++)
            {
                int colOffset = (lag - 1) * d;
                for (int j = 0; j < d; j++)
                    Z[t, colOffset + j] = data[t + MaxLag - lag, j];
            }
        }

        // Joint optimization of W (d x d) and A (lagDim x d) using augmented Lagrangian
        var W = new Matrix<T>(d, d); // contemporaneous — must be DAG
        var A = new Matrix<T>(lagDim, d); // lagged — no acyclicity constraint

        // Augmented Lagrangian parameters
        double rho = 1.0;
        double alpha = 0.0; // Lagrange multiplier
        double rhoMax = 1e+16;
        double gammaRho = 10.0; // rho increase factor
        double hPrev = double.PositiveInfinity;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Inner optimization: minimize L(W, A) = loss + lambda1*sparsity + alpha*h(W) + rho/2 * h(W)^2
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
        var result = new Matrix<T>(d, d);

        // Contemporaneous edges (from W)
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (Math.Abs(NumOps.ToDouble(W[i, j])) >= _wThreshold)
                    result[i, j] = W[i, j];

        // Lagged edges: aggregate across lags into summary d x d matrix
        for (int lag = 1; lag <= MaxLag; lag++)
        {
            int colOffset = (lag - 1) * d;
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    double lagWeight = Math.Abs(NumOps.ToDouble(A[colOffset + i, j]));
                    if (lagWeight >= _wThreshold)
                    {
                        // Take the maximum absolute lagged effect across lags
                        double current = Math.Abs(NumOps.ToDouble(result[i, j]));
                        if (lagWeight > current)
                            result[i, j] = A[colOffset + i, j];
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Inner optimization loop: gradient descent on the joint (W, A) objective.
    /// L(W,A) = ½n⁻¹||Xt - Xt*W - Z*A||²_F + λ₁(||W||₁ + ||A||₁) + α*h(W) + ρ/2 * h(W)²
    /// </summary>
    private void OptimizeInner(Matrix<T> Xt, Matrix<T> Z,
        Matrix<T> W, Matrix<T> A,
        int n, int d, int lagDim,
        double rho, double alpha)
    {
        double learningRate = 1e-3;
        int innerSteps = 200;
        double prevLoss = double.PositiveInfinity;

        for (int step = 0; step < innerSteps; step++)
        {
            // Compute residual: R = Xt - Xt*W - Z*A  [n x d]
            var R = new Matrix<T>(n, d);
            for (int t = 0; t < n; t++)
            {
                for (int j = 0; j < d; j++)
                {
                    T val = Xt[t, j];
                    for (int k = 0; k < d; k++)
                        val = NumOps.Subtract(val, NumOps.Multiply(Xt[t, k], W[k, j]));
                    for (int k = 0; k < lagDim; k++)
                        val = NumOps.Subtract(val, NumOps.Multiply(Z[t, k], A[k, j]));
                    R[t, j] = val;
                }
            }

            // Gradient of loss w.r.t. W: -1/n * Xt' * R
            T nT = NumOps.FromDouble(n);
            var gradW = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    T sum = NumOps.Zero;
                    for (int t = 0; t < n; t++)
                        sum = NumOps.Add(sum, NumOps.Multiply(Xt[t, i], R[t, j]));
                    gradW[i, j] = NumOps.Negate(NumOps.Divide(sum, nT));
                }

            // Gradient of loss w.r.t. A: -1/n * Z' * R
            var gradA = new Matrix<T>(lagDim, d);
            for (int i = 0; i < lagDim; i++)
                for (int j = 0; j < d; j++)
                {
                    T sum = NumOps.Zero;
                    for (int t = 0; t < n; t++)
                        sum = NumOps.Add(sum, NumOps.Multiply(Z[t, i], R[t, j]));
                    gradA[i, j] = NumOps.Negate(NumOps.Divide(sum, nT));
                }

            // Gradient of acyclicity constraint w.r.t. W: 2 * (e^{W∘W}) ∘ W
            double h = ComputeAcyclicity(W, d);
            var expWoW = ComputeMatrixExponentialOfHadamard(W, d);
            T two = NumOps.FromDouble(2.0);
            var gradH = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    gradH[i, j] = NumOps.Multiply(two, NumOps.Multiply(expWoW[i, j], W[i, j]));

            // Combined gradient for W: grad_loss + lambda1*sign(W) + (alpha + rho*h) * grad_h
            T constraintMult = NumOps.FromDouble(alpha + rho * h);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    T l1Grad = NumOps.FromDouble(_lambda1 * Math.Sign(NumOps.ToDouble(W[i, j])));
                    gradW[i, j] = NumOps.Add(gradW[i, j],
                        NumOps.Add(l1Grad, NumOps.Multiply(constraintMult, gradH[i, j])));
                }

            // Combined gradient for A: grad_loss + lambda1*sign(A)
            for (int i = 0; i < lagDim; i++)
                for (int j = 0; j < d; j++)
                {
                    T l1Grad = NumOps.FromDouble(_lambda1 * Math.Sign(NumOps.ToDouble(A[i, j])));
                    gradA[i, j] = NumOps.Add(gradA[i, j], l1Grad);
                }

            // Gradient descent update
            T lr = NumOps.FromDouble(learningRate);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    W[i, j] = NumOps.Subtract(W[i, j], NumOps.Multiply(lr, gradW[i, j]));

            for (int i = 0; i < lagDim; i++)
                for (int j = 0; j < d; j++)
                    A[i, j] = NumOps.Subtract(A[i, j], NumOps.Multiply(lr, gradA[i, j]));

            // Zero diagonal of W (no self-loops)
            for (int i = 0; i < d; i++)
                W[i, i] = NumOps.Zero;

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
    private double ComputeAcyclicity(Matrix<T> W, int d)
    {
        // W∘W (Hadamard square)
        var WoW = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                WoW[i, j] = NumOps.Multiply(W[i, j], W[i, j]);

        // Compute trace of matrix exponential via Taylor series
        double trace = d; // tr(I)
        var power = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++) power[i, i] = NumOps.One;

        double factorial = 1.0;
        for (int k = 1; k <= Math.Min(d, 20); k++)
        {
            factorial *= k;
            // power = power * WoW
            var newPower = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    T sum = NumOps.Zero;
                    for (int m = 0; m < d; m++)
                        sum = NumOps.Add(sum, NumOps.Multiply(power[i, m], WoW[m, j]));
                    newPower[i, j] = sum;
                }
            power = newPower;

            // Add tr(M^k) / k!
            double tr = 0;
            for (int i = 0; i < d; i++) tr += NumOps.ToDouble(power[i, i]);
            trace += tr / factorial;
        }

        return trace - d;
    }

    /// <summary>
    /// Computes e^{W∘W} via Taylor series (needed for gradient of h(W)).
    /// </summary>
    private Matrix<T> ComputeMatrixExponentialOfHadamard(Matrix<T> W, int d)
    {
        var WoW = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                WoW[i, j] = NumOps.Multiply(W[i, j], W[i, j]);

        // e^M via Taylor series
        var result = new Matrix<T>(d, d);
        var power = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            result[i, i] = NumOps.One; // I
            power[i, i] = NumOps.One;
        }

        double factorial = 1.0;
        for (int k = 1; k <= Math.Min(d, 20); k++)
        {
            factorial *= k;
            T invFactorial = NumOps.FromDouble(1.0 / factorial);
            var newPower = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    T sum = NumOps.Zero;
                    for (int m = 0; m < d; m++)
                        sum = NumOps.Add(sum, NumOps.Multiply(power[i, m], WoW[m, j]));
                    newPower[i, j] = sum;
                }
            power = newPower;

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    result[i, j] = NumOps.Add(result[i, j], NumOps.Multiply(power[i, j], invFactorial));
        }

        return result;
    }

    private double ComputeLoss(Matrix<T> R, int n, int d)
    {
        double sum = 0;
        for (int t = 0; t < n; t++)
            for (int j = 0; j < d; j++)
            {
                double r = NumOps.ToDouble(R[t, j]);
                sum += r * r;
            }
        return sum / (2.0 * n);
    }
}
