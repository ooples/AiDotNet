using AiDotNet.Attributes;
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
/// Reference: Fang et al. (2020), "Low-Rank DAG Learning", ICML Workshop.
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
[ModelPaper("Low-Rank DAG Learning", "https://arxiv.org/abs/2006.13006", Year = 2020, Authors = "Zhuangyan Fang, Shengyu Zhu, Jiji Zhang, Yue Liu, Zhitang Chen, Yangbo He")]
public class NOTEARSLowRank<T> : ContinuousOptimizationBase<T>
{
    private readonly double _learningRateValue;
    private readonly int _maxRank;
    private readonly int _innerIterations;
    private readonly int? _seed;
    private readonly double _initScale;
    private double _rhoMax = 1e+16;

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
        _learningRateValue = options?.LearningRate ?? 0.01;
        _maxRank = options?.MaxRank ?? 10;
        _innerIterations = options?.InnerIterations ?? 20;
        // Small init scale so W = A*B^T ≈ 0 (matching NOTEARS reference W=0 init).
        // L-BFGS builds up edges from near-zero. User can adjust via InitScale.
        _initScale = options?.InitScale ?? 0.01;
        _seed = options?.Seed;
        if (options?.MaxPenalty is { } maxPenalty) _rhoMax = maxPenalty;
        if (_maxRank <= 0)
            throw new ArgumentException("MaxRank must be greater than 0.");
        if (_learningRateValue <= 0 || double.IsNaN(_learningRateValue) || double.IsInfinity(_learningRateValue))
            throw new ArgumentException("LearningRate must be positive and finite.");
        if (_innerIterations < 1)
            throw new ArgumentException("InnerIterations must be at least 1.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        // Respect the configured learning rate without overriding
        double effectiveLr = _learningRateValue;

        var X = StandardizeData(data);

        // Choose rank
        int rank = Math.Min(d, _maxRank);

        // Initialize low-rank factors so W = A*B^T ≈ 0 (matching NOTEARS reference
        // which initializes W=0). For low-rank, we use small random values scaled by
        // the user-configurable InitScale. The L-BFGS optimizer then builds up edges.
        var A = new Matrix<T>(d, rank);
        var B = new Matrix<T>(d, rank);
        T initScale = NumOps.FromDouble(_initScale);
        var rng = _seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(_seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();
        for (int i = 0; i < d; i++)
            for (int r = 0; r < rank; r++)
            {
                A[i, r] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));
                B[i, r] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));
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

                    // Compute augmented Lagrangian objective
                    var (loss, lossGrad) = ComputeL2Loss(X, W);
                    var (h, hGrad) = ComputeNOTEARSConstraint(W);
                    T augCoeff = NumOps.Add(currentAlpha, NumOps.Multiply(currentRho, NumOps.FromDouble(h)));
                    T obj = NumOps.FromDouble(loss + NumOps.ToDouble(currentAlpha) * h
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
                                T totalGW = NumOps.Add(
                                    NumOps.Add(lossGrad[i, j], NumOps.Multiply(augCoeff, hGrad[i, j])),
                                    NumOps.FromDouble(Lambda1 * Math.Sign(NumOps.ToDouble(W[i, j]))));
                                gA = NumOps.Add(gA, NumOps.Multiply(totalGW, localB[j, r]));

                                T totalGWT = NumOps.Add(
                                    NumOps.Add(lossGrad[j, i], NumOps.Multiply(augCoeff, hGrad[j, i])),
                                    NumOps.FromDouble(Lambda1 * Math.Sign(NumOps.ToDouble(W[j, i]))));
                                gB = NumOps.Add(gB, NumOps.Multiply(totalGWT, localA[j, r]));
                            }
                            grad[i * rank + r] = gA;
                            grad[d * rank + i * rank + r] = gB;
                        }

                    return (obj, grad);
                },
                _innerIterations,
                NumOps.FromDouble(1e-5));

            // Unflatten optimized parameters back to A, B
            UnflattenVectorToAB(optimized, A, B, d, rank);

            // Outer: evaluate and update augmented Lagrangian
            var outerW = ReconstructW(A, B, d, rank);
            var (hVal, _) = ComputeNOTEARSConstraint(outerW);
            T hValT = NumOps.FromDouble(hVal);

            alphaLag = NumOps.Add(alphaLag, NumOps.Multiply(rhoT, hValT));
            if (NumOps.GreaterThan(hValT, NumOps.Multiply(NumOps.FromDouble(0.25), prevH)))
            {
                T newRho = NumOps.Multiply(rhoT, NumOps.FromDouble(10));
                T rhoMaxT = NumOps.FromDouble(_rhoMax);
                rhoT = NumOps.GreaterThan(newRho, rhoMaxT) ? rhoMaxT : newRho;
            }
            prevH = hValT;

            if (hVal < HTolerance) break;
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
}
