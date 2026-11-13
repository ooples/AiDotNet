using AiDotNet.Attributes;
using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// NOTEARS Low-Rank — DAG learning with low-rank parameterization for scalability.
/// </summary>
/// <remarks>
/// <para>
/// Parameterizes the weighted adjacency matrix W as a product of low-rank factors W = A * B^T,
/// where A and B are d x r matrices with r much less than d. This reduces the number of
/// parameters from O(d^2) to O(dr) and enables scalability to graphs with many variables.
/// The NOTEARS acyclicity constraint h(W) = tr(e^(W * W)) - d is applied to the reconstructed W.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Choose rank r = min(d, 10)</item>
/// <item>Initialize A, B as small random matrices of size d x r</item>
/// <item>Reconstruct W = A * B^T (with diagonal zeroed)</item>
/// <item>Compute L2 loss and NOTEARS acyclicity constraint on W</item>
/// <item>Compute gradients via chain rule: dL/dA = dL/dW * B, dL/dB = dL/dW^T * A</item>
/// <item>Update A, B via gradient descent with augmented Lagrangian for acyclicity</item>
/// <item>Threshold the final W = A * B^T</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> When you have many variables (say hundreds), the standard NOTEARS
/// weight matrix becomes very large. The low-rank trick says "most causal graphs are relatively
/// simple" and represents the matrix using fewer numbers, making the optimization much faster.
/// </para>
/// <para>
/// Reference: Fang et al. (2023), "On Low-Rank Directed Acyclic Graphs and Causal
/// Structure Learning", IEEE Transactions on Signal Processing.
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
[ResearchPaper("On Low-Rank Directed Acyclic Graphs and Causal Structure Learning", "https://arxiv.org/abs/2006.05691", Year = 2023, Authors = "Zhuangyan Fang, Shengyu Zhu, Jiji Zhang, Yue Liu, Zhitang Chen, Yangbo He")]
public class NOTEARSLowRank<T> : ContinuousOptimizationBase<T>
{
    /// <summary>
    /// How many dual-ascent updates must complete before <c>HTolerance</c> is allowed to stop the
    /// outer loop.
    /// </summary>
    /// <remarks>
    /// Fang et al. 2023 (arXiv:2006.05691), Algorithm 1: the augmented-Lagrangian loop runs four
    /// dual-ascent updates of (rho, alpha) before the acyclicity residual is trusted as a stopping
    /// signal. h starts small for any near-zero initialization, so testing it earlier stops on an
    /// initialization artefact rather than on a converged data fit. The gate is expressed as
    /// <c>outerIter &gt;= MinimumDualAscentUpdates - 1</c> because outerIter is zero-based.
    ///
    /// A caller who sets MaxIterations below this value gets a solver whose tolerance check never
    /// runs and which stops only on the rho ceiling.
    /// </remarks>
    private const int MinimumDualAscentUpdates = 4;

    private readonly int _maxRank;
    private readonly int _innerIterations;
    private readonly int? _seed;
    private readonly double? _initScale;
    private readonly double _rhoMax;

    /// <inheritdoc/>
    public override string Name => "NOTEARS Low-Rank";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes NOTEARS Low-Rank with optional configuration.
    /// </summary>
    public NOTEARSLowRank(CausalDiscoveryOptions? options = null)
    {
        ApplyOptions(options);

        // NOTEARS-Low-Rank is unregularized by default and uses the shorter
        // dual-ascent schedule published with the reference implementation.
        // Every value remains externally configurable through the shared options.
        Lambda1 = options?.SparsityPenalty ?? 0.0;
        MaxIterations = options?.MaxIterations ?? 15;
        HTolerance = options?.AcyclicityTolerance ?? 1e-6;
        _maxRank = options?.MaxRank ?? 10;
        _innerIterations = options?.InnerIterations ?? 100;
        _initScale = options?.InitScale;
        _seed = options?.Seed;
        _rhoMax = options?.MaxPenalty ?? 1e+20;
        if (_maxRank <= 0)
            throw new ArgumentException("MaxRank must be greater than 0.");
        if (_innerIterations < 1)
            throw new ArgumentException("InnerIterations must be at least 1.");
        if (_initScale is { } initScale
            && (initScale <= 0 || double.IsNaN(initScale) || double.IsInfinity(initScale)))
            throw new ArgumentException("InitScale must be positive and finite.");
        if (_rhoMax <= 0 || double.IsNaN(_rhoMax) || double.IsInfinity(_rhoMax))
            throw new ArgumentException("MaxPenalty must be positive and finite.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        // The reference evaluates standardized SEM observations. Normalize at
        // the public API boundary so callers get the same numerical regime.
        var X = StandardizeData(data);

        // Choose rank
        int rank = Math.Min(d, _maxRank);

        var A = new Matrix<T>(d, rank);
        var B = new Matrix<T>(d, rank);
        if (_initScale is null)
        {
            // Fang et al., Appendix C.2: U0 and V0 are the first r columns
            // of the d-by-d identity matrices.
            var identityColumns = Engine.TensorSlice(Engine.TensorEye<T>(d), new[] { 0, 0 }, new[] { d, rank });
            A = identityColumns.ToMatrix();
            B = identityColumns.ToMatrix();
        }
        else
        {
            // Explicit InitScale opts into a randomized alternative while retaining
            // the same low-rank parameterization. Seed makes that customization
            // reproducible; the default remains the paper's identity factors.
            var rng = _seed.HasValue
                ? Tensors.Helpers.RandomHelper.CreateSeededRandom(_seed.Value)
                : Tensors.Helpers.RandomHelper.CreateSecureRandom();
            var draws = Tensor<T>.CreateRandom(rng, d, d);
            var initialW = Engine.TensorMultiplyScalar(
                Engine.TensorSubtract(Engine.TensorMultiplyScalar(draws, NumOps.FromDouble(2.0)), Engine.TensorOnesLike(draws)),
                NumOps.FromDouble(_initScale.Value)).ToMatrix();

            var svd = new SvdDecomposition<T>(initialW);
            var leading = Engine.TensorSlice(Tensor<T>.FromMatrix(svd.U), new[] { 0, 0 }, new[] { d, rank });
            var singular = Engine.TensorSlice(Tensor<T>.FromVector(svd.S), new[] { 0 }, new[] { rank });
            A = Engine.TensorMultiply(leading,
                Engine.TensorBroadcastTo(singular.Reshape(new[] { 1, rank }), new[] { d, rank })).ToMatrix();
            B = Engine.TensorTranspose(
                Engine.TensorSlice(Tensor<T>.FromMatrix(svd.Vt), new[] { 0, 0 }, new[] { rank, d })).ToMatrix();
        }

        // Augmented Lagrangian with L-BFGS inner solver (per NOTEARS reference)
        T rhoT = NumOps.One;
        T alphaLag = NumOps.Zero;
        T prevH = NumOps.FromDouble(double.MaxValue);
        T innerTolerance = NumOps.FromDouble(1e-5);
        // The library's L-BFGS optimizer, used here purely as a function minimizer (no model), so
        // the inner solver and the model-training path share one implementation of the two-loop
        // recursion and the Armijo line search.
        var optimizer = Optimizers.LBFGSOptimizer<T, Tensor<T>, Tensor<T>>.CreateForFunction();

        for (int outerIter = 0; outerIter < MaxIterations; outerIter++)
        {
            // Flatten [A; B] into a single parameter vector
            var initialParams = FlattenABToVector(A, B, d, rank);

            // Capture current Lagrangian coefficients for the closure
            T currentAlpha = alphaLag;
            T currentRho = rhoT;

            // Define the objective + gradient function for L-BFGS
            var optimized = optimizer.Minimize(
                initialParams,
                (Vector<T> p) => EvaluateAugmentedLagrangian(p, X, d, rank, currentAlpha, currentRho),
                _innerIterations,
                innerTolerance);

            // A rejected L-BFGS line-search point can still contain finite
            // parameters whose product overflows, or non-finite parameters if
            // the search direction itself overflowed. Keep the last accepted
            // factors unless both the vector and reconstructed adjacency are
            // finite; otherwise the final threshold pass would surface Inf.
            if (IsFiniteVector(optimized))
            {
                var (candidateA, candidateB) = UnflattenVectorToAB(optimized, d, rank);
                var candidateW = ReconstructW(candidateA, candidateB, d, rank);
                if (IsFiniteMatrix(candidateW))
                {
                    A = candidateA;
                    B = candidateB;
                }
            }

            // Outer: evaluate and update augmented Lagrangian
            var outerW = ReconstructW(A, B, d, rank);
            var (hVal, _) = ComputeNOTEARSConstraint(outerW);
            if (double.IsNaN(hVal) || double.IsInfinity(hVal))
                break;

            bool rhoCapped = UpdateLagrangian(hVal, ref alphaLag, ref rhoT, ref prevH);
            // Do not trust the acyclicity residual until the reference's dual-ascent updates have
            // run; see MinimumDualAscentUpdates.
            if (outerIter >= MinimumDualAscentUpdates - 1 && hVal < HTolerance) break;
            if (rhoCapped) break;
        }

        // Final reconstruction and thresholding (with covariance fallback)
        var result = ReconstructW(A, B, d, rank);
        return ThresholdWithFallback(result, WThreshold, data);
    }

    /// <summary>
    /// Dual ascent on alpha, and rho grows tenfold (capped at the maximum penalty) when h did not
    /// fall below a quarter of its previous value.
    /// </summary>
    /// <returns><c>true</c> when rho has reached the maximum penalty.</returns>
    private bool UpdateLagrangian(double hVal, ref T alphaLag, ref T rhoT, ref T prevH)
    {
        T hValT = NumOps.FromDouble(hVal);

        alphaLag = NumOps.Add(alphaLag, NumOps.Multiply(rhoT, hValT));
        if (NumOps.GreaterThan(hValT, NumOps.Multiply(NumOps.FromDouble(0.25), prevH)))
        {
            T newRho = NumOps.Multiply(rhoT, NumOps.FromDouble(10));
            T rhoMaxT = NumOps.FromDouble(_rhoMax);
            rhoT = NumOps.GreaterThan(newRho, rhoMaxT) ? rhoMaxT : newRho;
        }
        prevH = hValT;
        return NumOps.ToDouble(rhoT) >= _rhoMax;
    }

    /// <summary>
    /// Augmented Lagrangian objective and its gradient with respect to the flattened factors [A; B].
    /// </summary>
    private (T Objective, Vector<T> Gradient) EvaluateAugmentedLagrangian(
        Vector<T> p, Matrix<T> X, int d, int rank, T currentAlpha, T currentRho)
    {
        int paramLen = 2 * d * rank;

        // Unflatten parameters
        var (localA, localB) = UnflattenVectorToAB(p, d, rank);
        var W = ReconstructW(localA, localB, d, rank);

        // Guard against non-finite W from an overshooting line-search
        // step: tr(exp(W∘W)) overflows for large |W|, the Inf/NaN then
        // propagates through the gradient into A/B, and Math.Sign(NaN)
        // below throws ArithmeticException. Returning a large finite
        // objective with a zero gradient makes the line search treat
        // the step as a failure and backtrack — the standard handling
        // in NOTEARS-style augmented-Lagrangian solvers.
        if (!IsFiniteMatrix(W))
            return (NumOps.FromDouble(1e+30), new Vector<T>(paramLen));

        // Compute augmented Lagrangian objective
        var (loss, lossGrad) = ComputeL2Loss(X, W);
        var (h, hGrad) = ComputeNOTEARSConstraint(W);
        if (double.IsNaN(loss) || double.IsInfinity(loss)
            || double.IsNaN(h) || double.IsInfinity(h))
            return (NumOps.FromDouble(1e+30), new Vector<T>(paramLen));
        T augCoeff = NumOps.Add(currentAlpha, NumOps.Multiply(currentRho, NumOps.FromDouble(h)));
        T obj = NumOps.FromDouble(loss
            + Lambda1 * ComputeL1Norm(W)
            + NumOps.ToDouble(currentAlpha) * h
            + 0.5 * NumOps.ToDouble(currentRho) * h * h);

        // ∂L/∂W = lossGrad + augCoeff * hGrad + lambda1 * sign(W), where sign(NaN) = 0.
        //
        // SafeSign: the augmented Lagrangian under aggressive rho schedules
        // (rho *= 10 every outer iteration that fails the constraint-decrease
        // gate) can drive `rho * h` to overflow, propagating NaN into W. The
        // raw `Math.Sign(double.NaN)` throws ArithmeticException ("Function
        // does not accept floating point Not-a-Number values"), tearing down
        // L-BFGS mid-step with no chance for the outer Lagrangian schedule
        // to recover. Treating sign(NaN) = 0 matches the convention in
        // sklearn/scipy subgradient code: a NaN coordinate contributes no
        // L1 subgradient, the optimizer keeps stepping, and the outer
        // schedule's rho ceiling clamp prevents unbounded blow-up so the
        // run eventually settles into a meaningful structure (closes the
        // NOTEARSLowRank_FindsCausalStructure Integration C failure).
        var w = Tensor<T>.FromMatrix(W);
        var zeros = Engine.TensorZerosLike(w);
        var safeSign = Engine.TensorWhere(Engine.TensorNotEquals(w, w), zeros, Engine.TensorSign(w));
        var totalGW = Engine.TensorAdd(
            Engine.TensorAdd(Tensor<T>.FromMatrix(lossGrad),
                Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(hGrad), augCoeff)),
            Engine.TensorMultiplyScalar(safeSign, NumOps.FromDouble(Lambda1)));

        // ReconstructW fixes W[j,j] to zero, so diagonal entries are constants and have no
        // factor derivative. Including their loss gradients here makes the supplied gradient
        // inconsistent with the objective.
        totalGW = Engine.TensorWhere(OffDiagonalMask(d), totalGW, zeros);

        // Chain rule to parameter space: ∂L/∂A = G B, ∂L/∂B = Gᵀ A
        var gA = Engine.TensorMatMul(totalGW, Tensor<T>.FromMatrix(localB));
        var gB = Engine.TensorMatMul(Engine.TensorTranspose(totalGW), Tensor<T>.FromMatrix(localA));
        var grad = Engine.TensorConcatenate(
            new[] { gA.Reshape(new[] { d * rank }), gB.Reshape(new[] { d * rank }) }, 0).ToVector();

        return (obj, grad);
    }

    /// <summary>
    /// Flattens low-rank factors [A; B] into a single Vector for the optimizer.
    /// </summary>
    private static Vector<T> FlattenABToVector(Matrix<T> A, Matrix<T> B, int d, int rank)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorConcatenate(new[]
        {
            Tensor<T>.FromMatrix(A).Reshape(new[] { d * rank }),
            Tensor<T>.FromMatrix(B).Reshape(new[] { d * rank })
        }, 0).ToVector();
    }

    /// <summary>
    /// Unflattens a parameter Vector back into A, B matrices.
    /// </summary>
    private static (Matrix<T> A, Matrix<T> B) UnflattenVectorToAB(Vector<T> vec, int d, int rank)
    {
        var engine = AiDotNetEngine.Current;
        var flat = vec.AsTensor();
        var A = engine.TensorSlice(flat, new[] { 0 }, new[] { d * rank }).Reshape(new[] { d, rank });
        var B = engine.TensorSlice(flat, new[] { d * rank }, new[] { d * rank }).Reshape(new[] { d, rank });
        return (A.ToMatrix(), B.ToMatrix());
    }

    /// <summary>
    /// Reconstructs W = A * B^T with diagonal zeroed.
    /// </summary>
    private Matrix<T> ReconstructW(Matrix<T> A, Matrix<T> B, int d, int rank)
    {
        var product = Engine.TensorMatMul(Tensor<T>.FromMatrix(A), Engine.TensorTranspose(Tensor<T>.FromMatrix(B)));
        return Engine.TensorWhere(OffDiagonalMask(d), product, Engine.TensorZerosLike(product)).ToMatrix();
    }

    private bool IsFiniteVector(Vector<T> values) => CountNonFinite(values.AsTensor()) == 0;

    private bool IsFiniteMatrix(Matrix<T> values) => CountNonFinite(Tensor<T>.FromMatrix(values)) == 0;

    /// <summary>Number of NaN or infinite entries.</summary>
    private int CountNonFinite(Tensor<T> values)
    {
        var nonFinite = Engine.TensorWhere(Engine.TensorIsFinite(values),
            Engine.TensorZerosLike(values), Engine.TensorOnesLike(values));
        return Engine.TensorCountNonzero(nonFinite);
    }
}
