using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// Metrics for monitoring and evaluating self-supervised learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> These metrics help track the quality of SSL training
/// and detect potential issues like representation collapse. Monitoring these
/// during training helps ensure the model is learning useful representations.</para>
///
/// <para><b>Key metrics:</b></para>
/// <list type="bullet">
/// <item><b>Representation collapse:</b> All embeddings become identical (very bad)</item>
/// <item><b>Feature std:</b> Standard deviation of features (should be positive)</item>
/// <item><b>Alignment:</b> Similarity between positive pairs (should be high)</item>
/// <item><b>Uniformity:</b> Distribution of features on hypersphere (should be uniform)</item>
/// </list>
/// </remarks>
public class SelfSupervisedLearningMetrics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Computes the standard deviation of representations (collapse detection).
    /// </summary>
    /// <param name="representations">Representations [batch_size, dim].</param>
    /// <returns>Standard deviation per dimension, averaged.</returns>
    /// <remarks>
    /// A low standard deviation indicates potential collapse - all representations
    /// are becoming similar. Good representations should have reasonable variance.
    /// </remarks>
    public T ComputeRepresentationStd(Tensor<T> representations)
    {
        // Population std per dimension, averaged over dimensions
        var centered = Engine.TensorSubtract(representations,
            Engine.ReduceMean(representations, new[] { 0 }, keepDims: true));
        var variance = Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 0 }, keepDims: false);
        return Engine.TensorMean(Engine.TensorSqrt(variance));
    }

    /// <summary>
    /// Computes alignment loss between positive pairs.
    /// </summary>
    /// <param name="z1">First view representations [batch_size, dim].</param>
    /// <param name="z2">Second view representations [batch_size, dim].</param>
    /// <returns>Average squared distance between positive pairs.</returns>
    /// <remarks>
    /// Lower alignment loss means positive pairs are closer together in
    /// representation space, which is desirable.
    /// </remarks>
    public T ComputeAlignment(Tensor<T> z1, Tensor<T> z2)
    {
        var batchSize = z1.Shape[0];

        // Mean over the batch of the squared distance between normalized pairs
        var diff = Engine.TensorSubtract(L2Normalize(z1), L2Normalize(z2));
        return NumOps.Divide(Engine.TensorSumOfSquares(diff), NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes uniformity loss (how uniformly distributed embeddings are).
    /// </summary>
    /// <param name="representations">Representations [batch_size, dim].</param>
    /// <param name="t">Temperature parameter (default: 2).</param>
    /// <returns>Uniformity metric (lower is more uniform).</returns>
    /// <remarks>
    /// Measures how uniformly representations are distributed on the hypersphere.
    /// More uniform distributions indicate better representations that capture
    /// diverse information.
    /// </remarks>
    public T ComputeUniformity(Tensor<T> representations, double t = 2.0)
    {
        var batchSize = representations.Shape[0];
        var dim = representations.Shape[1];

        // Normalize representations
        var zNorm = L2Normalize(representations);

        // Pairwise squared distances from explicit differences: [n, n]
        var diff = Engine.TensorSubtract(zNorm.Reshape(batchSize, 1, dim), zNorm.Reshape(1, batchSize, dim));
        var dist = Engine.ReduceSum(Engine.TensorSquare(diff), new[] { 2 }, keepDims: false);

        // exp(-t * dist) summed over the strict upper triangle (pairs i < j)
        var kernel = Engine.TensorExp(Engine.TensorMultiplyScalar(dist, NumOps.FromDouble(-t)));
        T sum = Engine.TensorSum(Engine.TensorTriu(kernel, 1));
        int count = batchSize * (batchSize - 1) / 2;

        // Return log of average
        var avg = NumOps.Divide(sum, NumOps.FromDouble(Math.Max(count, 1)));
        return NumOps.Log(NumOps.Add(avg, NumOps.FromDouble(1e-8)));
    }

    /// <summary>
    /// Computes the effective rank of the representation matrix.
    /// </summary>
    /// <param name="representations">Representations [batch_size, dim].</param>
    /// <returns>Effective rank (normalized entropy of singular values).</returns>
    /// <remarks>
    /// Effective rank measures the "dimensionality" of the learned representations.
    /// Collapsed representations have low effective rank. Good representations
    /// should use many dimensions effectively.
    /// </remarks>
    public T ComputeEffectiveRank(Tensor<T> representations)
    {
        // Compute covariance matrix
        var cov = ComputeCovarianceMatrix(representations);

        // Eigenvalues of the symmetric covariance matrix
        var eigenvalues = ComputeEigenvalues(cov);

        // |lambda| + 1e-8, normalized to a probability distribution
        var magnitudes = Engine.TensorAddScalar(Engine.TensorAbs(eigenvalues), NumOps.FromDouble(1e-8));
        var p = Engine.TensorDivideScalar(magnitudes, Engine.TensorSum(magnitudes));

        // Entropy over p > 1e-10
        var keep = Engine.TensorGreaterThan(p, NumOps.FromDouble(1e-10));
        var terms = Engine.TensorMultiply(keep, Engine.TensorMultiply(p, Engine.TensorLog(p)));
        T entropy = NumOps.Negate(Engine.TensorSum(terms));

        // Effective rank = exp(entropy)
        return NumOps.Exp(entropy);
    }

    /// <summary>
    /// Computes the eigenvalues of a symmetric matrix with a symmetric eigensolver.
    /// </summary>
    /// <remarks>
    /// This replaces a hand-rolled Jacobi iteration whose rotation angle had the wrong sign, so it
    /// never zeroed the pivot and returned values that were not eigenvalues (for
    /// [[3, 1.5], [1.5, 1]] it gave 1.31 and 2.69 instead of 2 +/- sqrt(13)/2).
    /// </remarks>
    private static Tensor<T> ComputeEigenvalues(Tensor<T> matrix)
    {
        return LinalgDispatch<T>.Eigh(matrix).Values;
    }

    /// <summary>
    /// Detects if representations are collapsing.
    /// </summary>
    /// <param name="representations">Representations to check.</param>
    /// <param name="threshold">Threshold for collapse detection (default: 0.01).</param>
    /// <returns>True if collapse is detected.</returns>
    public bool DetectCollapse(Tensor<T> representations, double threshold = 0.01)
    {
        var std = ComputeRepresentationStd(representations);
        return NumOps.ToDouble(std) < threshold;
    }

    /// <summary>
    /// Computes a full set of SSL metrics.
    /// </summary>
    public SelfSupervisedLearningMetricReport<T> ComputeFullReport(
        Tensor<T> z1, Tensor<T> z2)
    {
        return new SelfSupervisedLearningMetricReport<T>
        {
            Std1 = ComputeRepresentationStd(z1),
            Std2 = ComputeRepresentationStd(z2),
            Alignment = ComputeAlignment(z1, z2),
            Uniformity1 = ComputeUniformity(z1),
            Uniformity2 = ComputeUniformity(z2),
            EffectiveRank1 = ComputeEffectiveRank(z1),
            EffectiveRank2 = ComputeEffectiveRank(z2),
            CollapseDetected = DetectCollapse(z1) || DetectCollapse(z2)
        };
    }

    /// <summary>
    /// Computes cosine similarity between corresponding pairs.
    /// </summary>
    public T ComputeCosineSimilarity(Tensor<T> z1, Tensor<T> z2)
    {
        var batchSize = z1.Shape[0];

        // Mean over the batch of the dot product of normalized pairs
        var products = Engine.TensorMultiply(L2Normalize(z1), L2Normalize(z2));
        return NumOps.Divide(Engine.TensorSum(products), NumOps.FromDouble(batchSize));
    }

    private Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        // x / sqrt(sum(x^2) + 1e-8) per row
        var sumSquared = Engine.ReduceSum(Engine.TensorSquare(tensor), new[] { 1 }, keepDims: true);
        var norm = Engine.TensorSqrt(Engine.TensorAddScalar(sumSquared, NumOps.FromDouble(1e-8)));
        return Engine.TensorDivide(tensor, norm);
    }

    private Tensor<T> ComputeCovarianceMatrix(Tensor<T> representations)
    {
        var batchSize = representations.Shape[0];

        // (X - mean)^T (X - mean) / (batchSize - 1)
        var centered = Engine.TensorSubtract(representations,
            Engine.ReduceMean(representations, new[] { 0 }, keepDims: true));
        var scatter = Engine.TensorMatMul(Engine.TensorTranspose(centered), centered);
        return Engine.TensorDivideScalar(scatter, NumOps.FromDouble(batchSize - 1));
    }
}
