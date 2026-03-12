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
        _learningRateValue = options?.LearningRate ?? 0.001;
        _maxRank = options?.MaxRank ?? 10;
        _innerIterations = options?.InnerIterations ?? 20;
        _seed = options?.Seed;
        if (options?.MaxPenalty is { } maxPenalty) _rhoMax = maxPenalty;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        var X = StandardizeData(data);

        // Choose rank
        int rank = Math.Min(d, _maxRank);

        // Initialize low-rank factors A, B (d x rank)
        var A = new Matrix<T>(d, rank);
        var B = new Matrix<T>(d, rank);
        T initScale = NumOps.FromDouble(0.01);
        var rng = _seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(_seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();
        for (int i = 0; i < d; i++)
            for (int r = 0; r < rank; r++)
            {
                A[i, r] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));
                B[i, r] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));
            }

        T lr = NumOps.FromDouble(_learningRateValue);

        // Augmented Lagrangian
        double rho = 1.0;
        double alpha = 0.0;
        double prevH = double.MaxValue;

        for (int outerIter = 0; outerIter < MaxIterations; outerIter++)
        {
            // Inner optimization
            for (int innerIter = 0; innerIter < _innerIterations; innerIter++)
            {
                // Reconstruct W = A * B^T with diagonal zeroed
                var W = ReconstructW(A, B, d, rank);

                // Compute L2 loss and gradient
                var (loss, lossGrad) = ComputeL2Loss(X, W);

                // Compute acyclicity constraint and gradient
                var (h, hGrad) = ComputeNOTEARSConstraint(W);

                // Combined gradient: dL/dW_total = lossGrad + L1 sign + (alpha + rho*h) * hGrad
                var totalGrad = new Matrix<T>(d, d);
                T augCoeff = NumOps.FromDouble(alpha + rho * h);

                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                    {
                        T g = NumOps.Add(lossGrad[i, j],
                              NumOps.Multiply(augCoeff, hGrad[i, j]));
                        // Add L1 subgradient
                        g = NumOps.Add(g, NumOps.FromDouble(Lambda1 * Math.Sign(NumOps.ToDouble(W[i, j]))));
                        totalGrad[i, j] = g;
                    }

                // Chain rule: dL/dA = totalGrad * B, dL/dB = totalGrad^T * A
                for (int i = 0; i < d; i++)
                    for (int r = 0; r < rank; r++)
                    {
                        T gradA = NumOps.Zero;
                        T gradB = NumOps.Zero;
                        for (int j = 0; j < d; j++)
                        {
                            gradA = NumOps.Add(gradA, NumOps.Multiply(totalGrad[i, j], B[j, r]));
                            gradB = NumOps.Add(gradB, NumOps.Multiply(totalGrad[j, i], A[j, r]));
                        }

                        A[i, r] = NumOps.Subtract(A[i, r], NumOps.Multiply(lr, gradA));
                        B[i, r] = NumOps.Subtract(B[i, r], NumOps.Multiply(lr, gradB));
                    }
            }

            // Outer: evaluate and update Lagrangian
            var finalW = ReconstructW(A, B, d, rank);
            var (hVal, _) = ComputeNOTEARSConstraint(finalW);

            alpha += rho * hVal;
            if (hVal > 0.25 * prevH)
                rho = Math.Min(rho * 10, _rhoMax);
            prevH = hVal;

            if (hVal < HTolerance) break;
            if (rho >= _rhoMax) break;
        }

        // Final reconstruction and thresholding
        var result = ReconstructW(A, B, d, rank);
        return ThresholdAndClean(result, WThreshold);
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
