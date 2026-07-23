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
            for (int i = 0; i < d; i++)
                for (int r = 0; r < rank; r++)
                {
                    T identityEntry = i == r ? NumOps.One : NumOps.Zero;
                    A[i, r] = identityEntry;
                    B[i, r] = identityEntry;
                }
        }
        else
        {
            // Explicit InitScale opts into a randomized alternative while retaining
            // the same low-rank parameterization. Seed makes that customization
            // reproducible; the default remains the paper's identity factors.
            var rng = _seed.HasValue
                ? Tensors.Helpers.RandomHelper.CreateSeededRandom(_seed.Value)
                : Tensors.Helpers.RandomHelper.CreateSecureRandom();
            var initialW = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    initialW[i, j] =
                        NumOps.FromDouble((rng.NextDouble() * 2.0 - 1.0) * _initScale.Value);

            var svd = new SvdDecomposition<T>(initialW);
            for (int i = 0; i < d; i++)
                for (int r = 0; r < rank; r++)
                {
                    A[i, r] = NumOps.Multiply(svd.U[i, r], svd.S[r]);
                    B[i, r] = svd.Vt[r, i];
                }
        }

        // Augmented Lagrangian with L-BFGS inner solver (per NOTEARS reference)
        T rhoT = NumOps.One;
        T alphaLag = NumOps.Zero;
        T prevH = NumOps.FromDouble(double.MaxValue);
        var optimizer = new Optimizers.LBFGSFunctionOptimizer<T>();
        int paramLen = 2 * d * rank;

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
                (Vector<T> p) =>
                {
                    // Unflatten parameters
                    var localA = new Matrix<T>(d, rank);
                    var localB = new Matrix<T>(d, rank);
                    UnflattenVectorToAB(p, localA, localB, d, rank);
                    var W = ReconstructW(localA, localB, d, rank);

                    // Guard against non-finite W from an overshooting line-search
                    // step: tr(exp(W∘W)) overflows for large |W|, the Inf/NaN then
                    // propagates through the gradient into A/B, and Math.Sign(NaN)
                    // below throws ArithmeticException. Returning a large finite
                    // objective with a zero gradient makes the line search treat
                    // the step as a failure and backtrack — the standard handling
                    // in NOTEARS-style augmented-Lagrangian solvers.
                    bool wFinite = true;
                    for (int i = 0; i < d && wFinite; i++)
                        for (int j = 0; j < d; j++)
                        {
                            double wij = NumOps.ToDouble(W[i, j]);
                            if (double.IsNaN(wij) || double.IsInfinity(wij)) { wFinite = false; break; }
                        }
                    if (!wFinite)
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

                    // Chain rule to parameter space: ∂L/∂A, ∂L/∂B
                    var grad = new Vector<T>(paramLen);
                    for (int i = 0; i < d; i++)
                        for (int r = 0; r < rank; r++)
                        {
                            T gA = NumOps.Zero;
                            T gB = NumOps.Zero;
                            for (int j = 0; j < d; j++)
                            {
                                // ReconstructW fixes W[j,j] to zero, so diagonal
                                // entries are constants and have no factor derivative.
                                // Including their loss gradients here makes the
                                // supplied gradient inconsistent with the objective.
                                if (j == i)
                                    continue;

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
                                double wij = NumOps.ToDouble(W[i, j]);
                                double signWij = double.IsNaN(wij) ? 0.0 : Math.Sign(wij);
                                T totalGW = NumOps.Add(
                                    NumOps.Add(lossGrad[i, j], NumOps.Multiply(augCoeff, hGrad[i, j])),
                                    NumOps.FromDouble(Lambda1 * signWij));
                                gA = NumOps.Add(gA, NumOps.Multiply(totalGW, localB[j, r]));

                                double wji = NumOps.ToDouble(W[j, i]);
                                double signWji = double.IsNaN(wji) ? 0.0 : Math.Sign(wji);
                                T totalGWT = NumOps.Add(
                                    NumOps.Add(lossGrad[j, i], NumOps.Multiply(augCoeff, hGrad[j, i])),
                                    NumOps.FromDouble(Lambda1 * signWji));
                                gB = NumOps.Add(gB, NumOps.Multiply(totalGWT, localA[j, r]));
                            }
                            grad[i * rank + r] = gA;
                            grad[d * rank + i * rank + r] = gB;
                        }

                    return (obj, grad);
                },
                _innerIterations,
                NumOps.FromDouble(1e-5));

            // A rejected L-BFGS line-search point can still contain finite
            // parameters whose product overflows, or non-finite parameters if
            // the search direction itself overflowed. Keep the last accepted
            // factors unless both the vector and reconstructed adjacency are
            // finite; otherwise the final threshold pass would surface Inf.
            if (IsFiniteVector(optimized))
            {
                var candidateA = new Matrix<T>(d, rank);
                var candidateB = new Matrix<T>(d, rank);
                UnflattenVectorToAB(optimized, candidateA, candidateB, d, rank);
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
            T hValT = NumOps.FromDouble(hVal);

            alphaLag = NumOps.Add(alphaLag, NumOps.Multiply(rhoT, hValT));
            if (NumOps.GreaterThan(hValT, NumOps.Multiply(NumOps.FromDouble(0.25), prevH)))
            {
                T newRho = NumOps.Multiply(rhoT, NumOps.FromDouble(10));
                T rhoMaxT = NumOps.FromDouble(_rhoMax);
                rhoT = NumOps.GreaterThan(newRho, rhoMaxT) ? rhoMaxT : newRho;
            }
            prevH = hValT;

            // The reference performs at least four dual-ascent updates before
            // honoring h_tol, so a small initial h is not mistaken for a
            // converged data-fit solution.
            if (outerIter >= 3 && hVal < HTolerance) break;
            if (NumOps.ToDouble(rhoT) >= _rhoMax) break;
        }

        // Final reconstruction and thresholding (with covariance fallback)
        var result = ReconstructW(A, B, d, rank);
        return ThresholdWithFallback(result, WThreshold, data);
    }

    /// <summary>
    /// Flattens low-rank factors [A; B] into a single Vector for the optimizer.
    /// </summary>
    private static Vector<T> FlattenABToVector(Matrix<T> A, Matrix<T> B, int d, int rank)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var vec = new Vector<T>(2 * d * rank);
        for (int i = 0; i < d; i++)
            for (int r = 0; r < rank; r++)
            {
                vec[i * rank + r] = A[i, r];
                vec[d * rank + i * rank + r] = B[i, r];
            }
        return vec;
    }

    /// <summary>
    /// Unflattens a parameter Vector back into A, B matrices.
    /// </summary>
    private static void UnflattenVectorToAB(Vector<T> vec, Matrix<T> A, Matrix<T> B, int d, int rank)
    {
        for (int i = 0; i < d; i++)
            for (int r = 0; r < rank; r++)
            {
                A[i, r] = vec[i * rank + r];
                B[i, r] = vec[d * rank + i * rank + r];
            }
    }

    /// <summary>
    /// Reconstructs W = A * B^T with diagonal zeroed.
    /// </summary>
    private Matrix<T> ReconstructW(Matrix<T> A, Matrix<T> B, int d, int rank)
    {
        var W = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                T sum = NumOps.Zero;
                for (int r = 0; r < rank; r++)
                    sum = NumOps.Add(sum, NumOps.Multiply(A[i, r], B[j, r]));
                W[i, j] = sum;
            }
        return W;
    }

    private bool IsFiniteVector(Vector<T> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            double value = NumOps.ToDouble(values[i]);
            if (double.IsNaN(value) || double.IsInfinity(value))
                return false;
        }
        return true;
    }

    private bool IsFiniteMatrix(Matrix<T> values)
    {
        for (int i = 0; i < values.Rows; i++)
            for (int j = 0; j < values.Columns; j++)
            {
                double value = NumOps.ToDouble(values[i, j]);
                if (double.IsNaN(value) || double.IsInfinity(value))
                    return false;
            }
        return true;
    }
}
