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
        double rho = 1.0;
        double alphaLag = 0.0;
        double prevH = double.MaxValue;
        const int lbfgsMemory = 10;
        const double innerTol = 1e-5;

        for (int outerIter = 0; outerIter < MaxIterations; outerIter++)
        {
            // L-BFGS inner optimization over flattened [A; B] parameters
            int paramLen = 2 * d * rank;
            var param = new double[paramLen];
            FlattenAB(A, B, param, d, rank);

            var sVectors = new List<double[]>();
            var yVectors = new List<double[]>();
            double[]? prevGrad = null;
            var prevParam = (double[])param.Clone();

            for (int innerIter = 0; innerIter < _innerIterations; innerIter++)
            {
                UnflattenAB(param, A, B, d, rank);
                var W = ReconstructW(A, B, d, rank);

                // Compute objective and gradient w.r.t. W
                var (loss, lossGrad) = ComputeL2Loss(X, W);
                var (h, hGrad) = ComputeNOTEARSConstraint(W);
                double objective = loss + alphaLag * h + 0.5 * rho * h * h;
                double augCoeff = alphaLag + rho * h;

                // Chain rule: ∂L/∂A[i,r] = Σ_j (∂L/∂W[i,j] + augCoeff*∂h/∂W[i,j]) * B[j,r]
                var g = new double[paramLen];
                for (int i = 0; i < d; i++)
                    for (int r = 0; r < rank; r++)
                    {
                        double gA = 0, gB = 0;
                        for (int j = 0; j < d; j++)
                        {
                            double totalGradIJ = NumOps.ToDouble(lossGrad[i, j])
                                + augCoeff * NumOps.ToDouble(hGrad[i, j])
                                + Lambda1 * Math.Sign(NumOps.ToDouble(W[i, j]));
                            gA += totalGradIJ * NumOps.ToDouble(B[j, r]);

                            double totalGradJI = NumOps.ToDouble(lossGrad[j, i])
                                + augCoeff * NumOps.ToDouble(hGrad[j, i])
                                + Lambda1 * Math.Sign(NumOps.ToDouble(W[j, i]));
                            gB += totalGradJI * NumOps.ToDouble(A[j, r]);
                        }
                        g[i * rank + r] = gA;
                        g[d * rank + i * rank + r] = gB;
                    }

                // Check convergence
                double maxGrad = 0;
                for (int i = 0; i < paramLen; i++)
                    maxGrad = Math.Max(maxGrad, Math.Abs(g[i]));
                if (maxGrad < innerTol) break;

                // L-BFGS two-loop recursion for search direction
                var direction = ComputeLBFGSDirection(g, sVectors, yVectors);

                // Backtracking line search with Armijo condition
                double step = 1.0;
                for (int ls = 0; ls < 20; ls++)
                {
                    var trial = new double[paramLen];
                    for (int i = 0; i < paramLen; i++)
                        trial[i] = param[i] + step * direction[i];

                    UnflattenAB(trial, A, B, d, rank);
                    var trialW = ReconstructW(A, B, d, rank);
                    var (trialLoss, _) = ComputeL2Loss(X, trialW);
                    var (trialH, _) = ComputeNOTEARSConstraint(trialW);
                    double trialObj = trialLoss + alphaLag * trialH + 0.5 * rho * trialH * trialH;

                    double dirDeriv = 0;
                    for (int i = 0; i < paramLen; i++)
                        dirDeriv += g[i] * direction[i];

                    if (trialObj <= objective + 1e-4 * step * dirDeriv)
                    {
                        param = trial;
                        break;
                    }
                    step *= 0.5;
                    if (ls == 19) // Fallback: tiny step
                    {
                        for (int i = 0; i < paramLen; i++)
                            param[i] = prevParam[i] + 1e-4 * direction[i];
                    }
                }

                // Update L-BFGS memory
                UnflattenAB(param, A, B, d, rank);
                var newW = ReconstructW(A, B, d, rank);
                var (newLoss2, newLossGrad2) = ComputeL2Loss(X, newW);
                var (newH2, newHGrad2) = ComputeNOTEARSConstraint(newW);
                double newAugCoeff = alphaLag + rho * newH2;

                var newG = new double[paramLen];
                for (int i = 0; i < d; i++)
                    for (int r = 0; r < rank; r++)
                    {
                        double ngA = 0, ngB = 0;
                        for (int j = 0; j < d; j++)
                        {
                            double tg = NumOps.ToDouble(newLossGrad2[i, j])
                                + newAugCoeff * NumOps.ToDouble(newHGrad2[i, j]);
                            ngA += tg * NumOps.ToDouble(B[j, r]);
                            double tgT = NumOps.ToDouble(newLossGrad2[j, i])
                                + newAugCoeff * NumOps.ToDouble(newHGrad2[j, i]);
                            ngB += tgT * NumOps.ToDouble(A[j, r]);
                        }
                        newG[i * rank + r] = ngA;
                        newG[d * rank + i * rank + r] = ngB;
                    }

                if (prevGrad is not null)
                {
                    var s = new double[paramLen];
                    var y = new double[paramLen];
                    double sy = 0;
                    for (int i = 0; i < paramLen; i++)
                    {
                        s[i] = param[i] - prevParam[i];
                        y[i] = newG[i] - prevGrad[i];
                        sy += s[i] * y[i];
                    }
                    if (sy > 1e-10)
                    {
                        sVectors.Add(s);
                        yVectors.Add(y);
                        if (sVectors.Count > lbfgsMemory)
                        {
                            sVectors.RemoveAt(0);
                            yVectors.RemoveAt(0);
                        }
                    }
                }

                prevParam = (double[])param.Clone();
                prevGrad = newG;
            }

            UnflattenAB(param, A, B, d, rank);

            // Outer: evaluate and update Lagrangian
            var outerW = ReconstructW(A, B, d, rank);
            var (hVal, _) = ComputeNOTEARSConstraint(outerW);

            alphaLag += rho * hVal;
            if (hVal > 0.25 * prevH)
                rho = Math.Min(rho * 10, _rhoMax);
            prevH = hVal;

            if (hVal < HTolerance) break;
            if (rho >= _rhoMax) break;
        }

        // Final reconstruction and thresholding (with covariance fallback)
        var result = ReconstructW(A, B, d, rank);
        return ThresholdWithFallback(result, WThreshold, data);
    }

    private static void FlattenAB(Matrix<T> A, Matrix<T> B, double[] param, int d, int rank)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < d; i++)
            for (int r = 0; r < rank; r++)
            {
                param[i * rank + r] = numOps.ToDouble(A[i, r]);
                param[d * rank + i * rank + r] = numOps.ToDouble(B[i, r]);
            }
    }

    private static void UnflattenAB(double[] param, Matrix<T> A, Matrix<T> B, int d, int rank)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < d; i++)
            for (int r = 0; r < rank; r++)
            {
                A[i, r] = numOps.FromDouble(param[i * rank + r]);
                B[i, r] = numOps.FromDouble(param[d * rank + i * rank + r]);
            }
    }

    /// <summary>
    /// L-BFGS two-loop recursion to compute search direction.
    /// Per Nocedal and Wright, "Numerical Optimization" Algorithm 7.4.
    /// </summary>
    private static double[] ComputeLBFGSDirection(double[] gradient, List<double[]> sVectors, List<double[]> yVectors)
    {
        int n = gradient.Length;
        int m = sVectors.Count;

        var q = (double[])gradient.Clone();

        if (m == 0)
        {
            // Steepest descent (negated)
            for (int i = 0; i < n; i++) q[i] = -q[i];
            return q;
        }

        var alphas = new double[m];
        var rhos = new double[m];

        for (int i = 0; i < m; i++)
        {
            double dot = 0;
            for (int j = 0; j < n; j++) dot += sVectors[i][j] * yVectors[i][j];
            rhos[i] = dot > 1e-10 ? 1.0 / dot : 0;
        }

        // Backward pass
        for (int i = m - 1; i >= 0; i--)
        {
            double dot = 0;
            for (int j = 0; j < n; j++) dot += sVectors[i][j] * q[j];
            alphas[i] = rhos[i] * dot;
            for (int j = 0; j < n; j++) q[j] -= alphas[i] * yVectors[i][j];
        }

        // Initial Hessian approximation: H0 = (s^T y) / (y^T y) * I
        double sTy = 0, yTy = 0;
        for (int j = 0; j < n; j++)
        {
            sTy += sVectors[m - 1][j] * yVectors[m - 1][j];
            yTy += yVectors[m - 1][j] * yVectors[m - 1][j];
        }
        double gamma = yTy > 1e-10 ? sTy / yTy : 1.0;
        for (int j = 0; j < n; j++) q[j] *= gamma;

        // Forward pass
        for (int i = 0; i < m; i++)
        {
            double dot = 0;
            for (int j = 0; j < n; j++) dot += yVectors[i][j] * q[j];
            double beta = rhos[i] * dot;
            for (int j = 0; j < n; j++) q[j] += (alphas[i] - beta) * sVectors[i][j];
        }

        // Negate for descent direction
        for (int i = 0; i < n; i++) q[i] = -q[i];
        return q;
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
