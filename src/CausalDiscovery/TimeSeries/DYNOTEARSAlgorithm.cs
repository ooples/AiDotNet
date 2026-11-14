using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("DYNOTEARS: Structure Learning from Time-Series Data", "https://proceedings.mlr.press/v108/pamfil20a.html", Year = 2020, Authors = "Roxana Pamfil, Nisara Sriwattanaworachai, Shaan Desai, Philip Pilgerstorfer, Konstantinos Georgatzis, Paul Maygsidt, Jesse M. Sheridan")]
public class DYNOTEARSAlgorithm<T> : TimeSeriesCausalBase<T>
{
    private double _lambda1 = 0.1;
    private double _wThreshold = 0.3;
    private int _maxIterations = 100;
    private int _innerIterations = 200;
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
        if (options?.InnerIterations.HasValue == true) _innerIterations = options.InnerIterations.Value;
        if (options?.AcyclicityTolerance.HasValue == true) _hTol = options.AcyclicityTolerance.Value;

        if (_maxIterations <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxIterations must be positive.");
        if (_innerIterations <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "InnerIterations must be positive.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n <= MaxLag + 1) return new Matrix<T>(d, d);

        // Standardize data to zero mean, unit variance for numerical stability.
        // Without standardization, raw data values cause the matrix exponential
        // in the acyclicity constraint to explode, preventing convergence.
        var x = StandardizeData(Tensor<T>.FromMatrix(data), n, d);

        int effectiveN = n - MaxLag;

        // Build X_t: contemporaneous data matrix [effectiveN x d]
        var Xt = Engine.TensorSlice(x, new[] { MaxLag, 0 }, new[] { effectiveN, d });

        // Build Z: lagged data matrix [effectiveN x (d * MaxLag)]
        // Z = [X(t-1), X(t-2), ..., X(t-MaxLag)]
        int lagDim = d * MaxLag;
        var Z = Engine.TensorConcatenate(Enumerable.Range(1, MaxLag)
            .Select(lag => Engine.TensorSlice(x, new[] { MaxLag - lag, 0 }, new[] { effectiveN, d }))
            .ToArray(), 1);

        // Joint optimization of W (d x d) and A (lagDim x d) using augmented Lagrangian
        var W = new Tensor<T>(new[] { d, d }); // contemporaneous — must be DAG
        var A = new Tensor<T>(new[] { lagDim, d }); // lagged — no acyclicity constraint

        // Augmented Lagrangian parameters
        double rho = 1.0;
        double alpha = 0.0; // Lagrange multiplier
        double rhoMax = 1e+16;
        double gammaRho = 10.0; // rho increase factor
        double hPrev = double.PositiveInfinity;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Inner optimization: minimize L(W, A) = loss + lambda1*sparsity + alpha*h(W) + rho/2 * h(W)^2
            (W, A) = OptimizeInner(Xt, Z, W, A, effectiveN, d, lagDim, rho, alpha);

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

        return BuildSummary(Xt, W, A, d).ToMatrix();
    }

    /// <summary>
    /// Thresholded summary adjacency: contemporaneous W entries with |w| ≥ threshold, then for each lag in
    /// order the off-diagonal lagged entries with |a| ≥ threshold that exceed the magnitude already there;
    /// when nothing off the diagonal survives, the pairwise |correlation| of the contemporaneous data
    /// (i &lt; j, both sums of squares above 1e-10, |corr| ≥ threshold).
    /// </summary>
    private Tensor<T> BuildSummary(Tensor<T> Xt, Tensor<T> W, Tensor<T> A, int d)
    {
        T threshold = NumOps.FromDouble(_wThreshold);
        var zeros = new Tensor<T>(new[] { d, d });
        var offDiagonal = OffDiagonalMask(d);

        // Contemporaneous edges (from W)
        var result = Engine.TensorWhere(AtLeast(Engine.TensorAbs(W), threshold), W, zeros);

        // Lagged edges: aggregate across lags into summary d x d matrix.
        // Skip the diagonal (i == j): a lagged self-effect A[i,i] is autoregression
        // (X_{t-p} -> X_t), not a structural self-edge in the causal graph. The
        // contemporaneous matrix W is acyclic with a zero diagonal by construction
        // (NOTEARS/DYNOTEARS, Pamfil et al. 2020), so the summary adjacency the
        // algorithm reports as the causal STRUCTURE must also have a zero diagonal —
        // a variable does not "cause itself". Aggregating lagged self-effects onto the
        // diagonal previously produced spurious self-loops (e.g. result[0,0]=0.357).
        for (int lag = 1; lag <= MaxLag; lag++)
        {
            var lagged = Engine.TensorSlice(A, new[] { (lag - 1) * d, 0 }, new[] { d, d });
            var lagWeight = Engine.TensorAbs(lagged);
            // Take the maximum absolute lagged effect across lags
            var replace = Engine.TensorMultiply(Engine.TensorMultiply(offDiagonal, AtLeast(lagWeight, threshold)),
                Engine.TensorGreaterThan(lagWeight, Engine.TensorAbs(result)));
            result = Engine.TensorWhere(replace, lagged, result);
        }

        // Fallback: if thresholding removed all edges (e.g., near-degenerate data
        // where gradient descent produces symmetric small weights), use correlation.
        var offDiagonalMagnitude = Engine.TensorMultiply(offDiagonal, Engine.TensorGreaterThan(Engine.TensorAbs(result), NumOps.Zero));
        if (Engine.TensorCountNonzero(offDiagonalMagnitude) > 0)
            return result;

        // Use pairwise correlation on the standardized contemporaneous data
        var centered = CenterColumns(Xt);
        var cross = Engine.TensorMatMul(Engine.TensorTranspose(centered), centered);
        var ss = Engine.TensorDiagonal(cross);
        var corr = Engine.TensorAbs(Engine.TensorDivide(cross, Engine.TensorSqrt(Engine.TensorOuter(ss, ss))));
        var shape = new[] { d, d };
        T ssFloor = NumOps.FromDouble(1e-10);
        var ssOk = Engine.TensorMultiply(
            Engine.TensorGreaterThan(Engine.TensorBroadcastTo(ss.Reshape(new[] { d, 1 }), shape), ssFloor),
            Engine.TensorGreaterThan(Engine.TensorBroadcastTo(ss.Reshape(new[] { 1, d }), shape), ssFloor));
        var keep = Engine.TensorMultiply(Engine.TensorMultiply(Engine.TensorTriu(Engine.TensorOnesLike(corr), 1), ssOk),
            AtLeast(corr, threshold));
        return Engine.TensorWhere(keep, corr, result);
    }

    /// <summary>
    /// 1 where value ≥ bound (false for NaN), else 0.
    /// </summary>
    private Tensor<T> AtLeast(Tensor<T> values, T bound)
        => Engine.TensorAdd(Engine.TensorGreaterThan(values, bound), Engine.TensorEquals(values, bound));

    /// <summary>
    /// Inner optimization loop: gradient descent on the joint (W, A) objective.
    /// L(W,A) = ½n⁻¹||Xt - Xt*W - Z*A||²_F + λ₁(||W||₁ + ||A||₁) + α*h(W) + ρ/2 * h(W)²
    /// </summary>
    private (Tensor<T> W, Tensor<T> A) OptimizeInner(Tensor<T> Xt, Tensor<T> Z,
        Tensor<T> W, Tensor<T> A,
        int n, int d, int lagDim,
        double rho, double alpha)
    {
        double learningRate = 1e-3;
        double prevLoss = double.PositiveInfinity;

        var XtT = Engine.TensorTranspose(Xt);
        var ZT = Engine.TensorTranspose(Z);
        T nT = NumOps.FromDouble(n);
        T two = NumOps.FromDouble(2.0);
        T lambda = NumOps.FromDouble(_lambda1);
        T lr = NumOps.FromDouble(learningRate);
        T rhoT = NumOps.FromDouble(rho);
        T alphaT = NumOps.FromDouble(alpha);
        T notNaNBound = NumOps.FromDouble(-1.0);
        var zeros = new Tensor<T>(new[] { d, d });
        var offDiagonal = OffDiagonalMask(d);

        for (int step = 0; step < _innerIterations; step++)
        {
            // Compute residual: R = Xt - Xt*W - Z*A  [n x d]
            var R = Engine.TensorSubtract(Engine.TensorSubtract(Xt, Engine.TensorMatMul(Xt, W)), Engine.TensorMatMul(Z, A));

            // Gradients of the loss: -1/n * Xt' * R and -1/n * Z' * R
            var gradW = Engine.TensorNegate(Engine.TensorDivideScalar(Engine.TensorMatMul(XtT, R), nT));
            var gradA = Engine.TensorNegate(Engine.TensorDivideScalar(Engine.TensorMatMul(ZT, R), nT));

            // sign() is undefined for NaN (Math.Sign throws), so a diverged W or A fails loudly.
            if (Engine.TensorCountNonzero(Engine.TensorGreaterThan(Engine.TensorAbs(W), notNaNBound)) < d * d ||
                Engine.TensorCountNonzero(Engine.TensorGreaterThan(Engine.TensorAbs(A), notNaNBound)) < lagDim * d)
                throw new ArithmeticException("Function does not accept floating point Not-a-Number values.");

            // Gradient of acyclicity constraint w.r.t. W: 2 * (e^{W∘W}) ∘ W
            var (expWoW, h) = ExpHadamardSquare(W, d);
            var gradH = Engine.TensorMultiplyScalar(Engine.TensorMultiply(expWoW, W), two);

            // Combined gradient for W: grad_loss + lambda1*sign(W) + (alpha + rho*h) * grad_h
            var constraintMult = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(h, rhoT), alphaT).Reshape(new[] { 1, 1 });
            gradW = Engine.TensorAdd(gradW, Engine.TensorAdd(Engine.TensorMultiplyScalar(Engine.TensorSign(W), lambda),
                Engine.TensorMultiply(Engine.TensorBroadcastTo(constraintMult, new[] { d, d }), gradH)));

            // Combined gradient for A: grad_loss + lambda1*sign(A)
            gradA = Engine.TensorAdd(gradA, Engine.TensorMultiplyScalar(Engine.TensorSign(A), lambda));

            // Gradient descent update, then zero diagonal of W (no self-loops)
            W = Engine.TensorSubtract(W, Engine.TensorMultiplyScalar(gradW, lr));
            A = Engine.TensorSubtract(A, Engine.TensorMultiplyScalar(gradA, lr));
            W = Engine.TensorWhere(offDiagonal, W, zeros);

            // Convergence check every 20 steps
            if (step % 20 == 0)
            {
                double loss = ComputeLoss(R, n);
                if (Math.Abs(loss - prevLoss) < 1e-8 * (1 + Math.Abs(prevLoss)))
                    break;
                prevLoss = loss;
            }
        }

        return (W, A);
    }

    /// <summary>
    /// Computes h(W) = tr(e^{W∘W}) - d, the NOTEARS acyclicity constraint.
    /// Uses Taylor series approximation: e^M ≈ I + M + M²/2! + M³/3! + ...
    /// </summary>
    private double ComputeAcyclicity(Tensor<T> W, int d)
        => NumOps.ToDouble(ExpHadamardSquare(W, d).Trace.ToArray()[0]);

    /// <summary>
    /// e^{W∘W} by its Taylor series through min(d, 20) terms (needed for the gradient of h(W)), and
    /// h(W) = tr(e^{W∘W}) − d (the trace of every term after the identity) as a [1] tensor.
    /// </summary>
    private (Tensor<T> Exp, Tensor<T> Trace) ExpHadamardSquare(Tensor<T> w, int d)
    {
        var wow = Engine.TensorMultiply(w, w);
        var identity = Engine.TensorEye<T>(d);
        var series = new Tensor<T>(new[] { d, d });
        var power = identity;

        // runningFactorial holds k! as a [1, 1] tensor, broadcast over the power matrix.
        var one = Engine.TensorOnesLike(new Tensor<T>(new[] { 1, 1 }));
        var counter = Engine.TensorZerosLike(one);
        var runningFactorial = one;
        for (int k = 1; k <= Math.Min(d, 20); k++)
        {
            power = Engine.TensorMatMul(power, wow);
            counter = Engine.TensorAdd(counter, one);
            runningFactorial = Engine.TensorMultiply(runningFactorial, counter);
            series = Engine.TensorAdd(series, Engine.TensorDivide(power, Engine.TensorBroadcastTo(runningFactorial, new[] { d, d })));
        }

        var trace = Engine.ReduceSum(Engine.TensorDiagonal(series), new[] { 0 }, true);
        return (Engine.TensorAdd(identity, series), trace);
    }

    private double ComputeLoss(Tensor<T> R, int n)
        => NumOps.ToDouble(Engine.TensorSum(Engine.TensorMultiply(R, R))) / (2.0 * n);

    private Tensor<T> StandardizeData(Tensor<T> data, int n, int d)
    {
        // (x - mean) / sqrt(var + 1e-15) with the population variance
        var centered = CenterColumns(data);
        var std = Engine.TensorSqrt(Engine.TensorAddScalar(
            Engine.ReduceMean(Engine.TensorMultiply(centered, centered), new[] { 0 }, true), NumOps.FromDouble(1e-15)));

        // Column-specific perturbation to break exact collinearity.
        // When variables are exact linear transforms, standardization makes columns
        // identical. This small scaling breaks the symmetry so gradient descent
        // can distinguish between causal directions. Column j is scaled by 1 + 1e-4 * (j + 1).
        var columnNumber = Engine.TensorCumSum(Engine.TensorOnesLike(new Tensor<T>(new[] { 1, d })), 1);
        var perturb = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(columnNumber, NumOps.FromDouble(1e-4)), NumOps.One);

        var shape = new[] { n, d };
        return Engine.TensorMultiply(Engine.TensorDivide(centered, Engine.TensorBroadcastTo(std, shape)),
            Engine.TensorBroadcastTo(perturb, shape));
    }
}
