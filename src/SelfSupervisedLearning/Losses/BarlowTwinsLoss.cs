using AiDotNet.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// Barlow Twins Loss - redundancy reduction loss for self-supervised learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Barlow Twins loss encourages the cross-correlation matrix
/// between embeddings of two augmented views to be close to the identity matrix.
/// This avoids both collapse and redundant features.</para>
///
/// <para><b>Key insight:</b> Instead of contrastive learning, Barlow Twins achieves
/// representation quality through decorrelation - making different feature dimensions
/// capture different information.</para>
///
/// <para><b>Loss components:</b></para>
/// <list type="bullet">
/// <item><b>Invariance term:</b> Diagonal elements should be 1 (same features match)</item>
/// <item><b>Redundancy reduction:</b> Off-diagonal elements should be 0 (no redundancy)</item>
/// </list>
///
/// <para><b>Loss formula:</b></para>
/// <code>
/// L = Σ_i (1 - C_ii)² + λ * Σ_i Σ_{j≠i} C_ij²
/// </code>
/// <para>where C is the cross-correlation matrix and λ is the redundancy weight.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Barlow Twins: Self-Supervised Learning via Redundancy Reduction", "https://arxiv.org/abs/2103.03230", Year = 2021, Authors = "Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny")]
public class BarlowTwinsLoss<T> : ContrastiveLossBase<T>
{
    // NumOps and Engine are inherited from ContrastiveLossBase; redeclaring them here would shadow
    // the base members rather than add anything.
    /// <summary>
    /// Added inside the square root when standardizing over the batch, so a constant feature scales
    /// by a finite value instead of dividing by zero.
    /// </summary>
    private const double BatchVarianceEpsilon = 1e-12;

    private readonly double _lambda;
    private readonly bool _normalize;

    /// <summary>
    /// Gets the lambda (redundancy reduction weight) parameter.
    /// </summary>
    public double Lambda => _lambda;

    /// <summary>
    /// Initializes a new instance of the BarlowTwinsLoss class.
    /// </summary>
    /// <param name="lambda">Weight for off-diagonal (redundancy) terms (default: 0.0051).</param>
    /// <param name="normalize">Whether to normalize along batch dimension (default: true).</param>
    public BarlowTwinsLoss(double lambda = 0.0051, bool normalize = true)
    {
        if (lambda < 0)
            throw new ArgumentOutOfRangeException(nameof(lambda), "Lambda must be non-negative");

        _lambda = lambda;
        _normalize = normalize;
    }

    /// <summary>
    /// Computes the Barlow Twins loss between two views.
    /// </summary>
    /// <param name="z1">Embeddings from view 1 [batch_size, dim].</param>
    /// <param name="z2">Embeddings from view 2 [batch_size, dim].</param>
    /// <returns>The computed loss value.</returns>
    /// <remarks>
    /// Differentiable Barlow Twins (Zbontar et al. 2021): sum_i (1 - C_ii)^2 + lambda * sum_{i!=j}
    /// C_ij^2 over the cross-correlation C. Diagonal and off-diagonal terms are selected with
    /// constant masks and reduced, rather than read out by indexing, so the tape survives.
    /// </remarks>
    public override Tensor<T> ComputeLoss(Tensor<T> z1, Tensor<T> z2)
    {
        // Shapes, not just ranks: the engine broadcasts, so a [8, 1] against a [8, 768] returns a
        // finite loss computed against the wrong target instead of failing.
        ContrastiveTapeOps<T>.RequireMatchingRank2(
            z1, z2, "BarlowTwins", nameof(z1), nameof(z2));

        if (z1 is null) throw new ArgumentNullException(nameof(z1));
        if (z2 is null) throw new ArgumentNullException(nameof(z2));

        int batchSize = z1.Shape[0];
        int dim = z1.Shape[1];

        var a = _normalize ? BatchNormalizeDifferentiable(z1) : z1;
        var b = _normalize ? BatchNormalizeDifferentiable(z2) : z2;

        // C = (a^T @ b) / batchSize
        var cross = Engine.TensorMatMul(Engine.TensorTranspose(a), b);
        var c = Engine.TensorDivideScalar(cross, NumOps.FromDouble(batchSize));

        var eye = ObjectiveOps.Identity<T>(dim);
        var offMask = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(eye, NumOps.FromDouble(-1.0)), NumOps.One);

        // invariance: sum over the diagonal of (1 - C)^2
        var oneMinusC = Engine.TensorNegate(Engine.TensorAddScalar(c, NumOps.FromDouble(-1.0)));
        var diagPart = Engine.TensorMultiply(oneMinusC, eye);
        var invariance = Engine.ReduceSum(Engine.TensorMultiply(diagPart, diagPart), null, keepDims: false);

        // redundancy: sum over the off-diagonal of C^2
        var offPart = Engine.TensorMultiply(c, offMask);
        var redundancy = Engine.ReduceSum(Engine.TensorMultiply(offPart, offPart), null, keepDims: false);

        return Engine.TensorAdd(
            invariance, Engine.TensorMultiplyScalar(redundancy, NumOps.FromDouble(_lambda)));
    }

    /// <summary>Per-feature standardization across the batch, built from IEngine ops.</summary>
    private Tensor<T> BatchNormalizeDifferentiable(Tensor<T> z)
    {
        int batchSize = z.Shape[0];
        var mean = Engine.TensorDivideScalar(
            Engine.ReduceSum(z, new[] { 0 }, keepDims: true), NumOps.FromDouble(batchSize));
        var centered = Engine.TensorSubtract(z, mean);
        var variance = Engine.TensorDivideScalar(
            Engine.ReduceSum(Engine.TensorMultiply(centered, centered), new[] { 0 }, keepDims: true),
            NumOps.FromDouble(batchSize));
        var std = Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-12)));
        return Engine.TensorDivide(centered, std);
    }

    private T ComputeLossScalarLegacy(Tensor<T> z1, Tensor<T> z2)
    {
        if (z1 is null) throw new ArgumentNullException(nameof(z1));
        if (z2 is null) throw new ArgumentNullException(nameof(z2));

        var batchSize = z1.Shape[0];

        // Normalize along batch dimension
        var z1Norm = _normalize ? BatchNormalize(z1) : z1;
        var z2Norm = _normalize ? BatchNormalize(z2) : z2;

        // Compute cross-correlation matrix C = (z1^T @ z2) / batch_size
        var crossCorr = ComputeCrossCorrelation(z1Norm, z2Norm, batchSize);

        // Compute loss: invariance (diagonal) + redundancy (off-diagonal)
        // Invariance term: sum_i (1 - C_ii)^2
        T invarianceLoss = Engine.TensorSumOfSquares(
            Engine.ScalarMinusTensor(NumOps.One, Engine.TensorDiagonal(crossCorr)));

        // Redundancy term: sum_{i != j} C_ij^2
        T redundancyLoss = Engine.TensorSumOfSquares(OffDiagonal(crossCorr));

        // Total loss = invariance + lambda * redundancy
        var lambdaT = NumOps.FromDouble(_lambda);
        var totalLoss = NumOps.Add(invarianceLoss, NumOps.Multiply(lambdaT, redundancyLoss));

        return totalLoss;
    }

    /// <summary>
    /// Computes the Barlow Twins loss with gradients for backpropagation.
    /// </summary>
    public (T loss, Tensor<T> gradZ1, Tensor<T> gradZ2) ComputeLossWithGradients(
        Tensor<T> z1, Tensor<T> z2)
    {
        var batchSize = z1.Shape[0];
        var dim = z1.Shape[1];

        var z1Norm = _normalize ? BatchNormalize(z1) : z1;
        var z2Norm = _normalize ? BatchNormalize(z2) : z2;

        var crossCorr = ComputeCrossCorrelation(z1Norm, z2Norm, batchSize);

        // Compute loss components
        T invarianceLoss = Engine.TensorSumOfSquares(
            Engine.ScalarMinusTensor(NumOps.One, Engine.TensorDiagonal(crossCorr)));
        T redundancyLoss = Engine.TensorSumOfSquares(OffDiagonal(crossCorr));

        // Gradient of cross-correlation w.r.t. loss:
        // diagonal -2 * (1 - C_ii), off-diagonal 2 * lambda * C_ij
        var diagonalGrad = Engine.TensorMultiplyScalar(
            Engine.ScalarMinusTensor(NumOps.One, crossCorr), NumOps.FromDouble(-2.0));
        var offDiagonalGrad = Engine.TensorMultiplyScalar(crossCorr, NumOps.FromDouble(2.0 * _lambda));
        var gradC = Engine.TensorWhere(Engine.TensorEye<T>(dim), diagonalGrad, offDiagonalGrad);

        var lambdaT = NumOps.FromDouble(_lambda);
        var totalLoss = NumOps.Add(invarianceLoss, NumOps.Multiply(lambdaT, redundancyLoss));

        // Backpropagate through cross-correlation to get gradients w.r.t. z1, z2
        var invBatch = NumOps.FromDouble(1.0 / batchSize);
        var z1Scaled = Engine.TensorMultiplyScalar(z1Norm, invBatch);
        var z2Scaled = Engine.TensorMultiplyScalar(z2Norm, invBatch);

        // gradZ1[b,i] = sum_j gradC[i,j] * z2Norm[b,j] / N
        var gradZ1 = Engine.TensorMatMul(z2Scaled, Engine.TensorTranspose(gradC));

        // gradZ2[b,i] = sum_j gradC[j,i] * z1Norm[b,j] / N
        var gradZ2 = Engine.TensorMatMul(z1Scaled, gradC);

        return (totalLoss, gradZ1, gradZ2);
    }

    /// <summary>
    /// Computes the cross-correlation matrix between two sets of embeddings.
    /// </summary>
    public Tensor<T> ComputeCrossCorrelation(Tensor<T> z1, Tensor<T> z2, int batchSize)
    {
        var invBatch = NumOps.FromDouble(1.0 / batchSize);

        // C[i,j] = (sum_b z1[b,i] * z2[b,j]) / batchSize
        return Engine.TensorMultiplyScalar(Engine.TensorMatMul(Engine.TensorTranspose(z1), z2), invBatch);
    }

    /// <summary>
    /// Normalizes tensor along batch dimension (mean=0, std=1 for each feature).
    /// </summary>
    private Tensor<T> BatchNormalize(Tensor<T> tensor)
    {
        var batchSize = tensor.Shape[0];
        var batchSizeT = NumOps.FromDouble(batchSize);

        // Per-feature mean and population variance across the batch
        var mean = Engine.TensorDivideScalar(Engine.ReduceSum(tensor, new[] { 0 }, keepDims: true), batchSizeT);
        var centered = Engine.TensorSubtract(tensor, mean);
        var variance = Engine.TensorDivideScalar(
            Engine.ReduceSum(Engine.TensorSquare(centered), new[] { 0 }, keepDims: true), batchSizeT);

        var std = Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-8)));

        // Normalize
        return Engine.TensorDivide(centered, std);
    }

    /// <summary>
    /// Computes the off-diagonal sum of a matrix (useful for monitoring).
    /// </summary>
    public T OffDiagonalSum(Tensor<T> matrix)
    {
        return Engine.TensorSumOfSquares(OffDiagonal(matrix));
    }

    /// <summary>
    /// A copy of a square matrix with its diagonal replaced by zero (selected, so a non-finite
    /// diagonal entry does not leak into off-diagonal sums).
    /// </summary>
    private Tensor<T> OffDiagonal(Tensor<T> matrix)
    {
        var dim = matrix.Shape[0];
        return Engine.TensorWhere(Engine.TensorEye<T>(dim), new Tensor<T>(new[] { dim, dim }), matrix);
    }

    /// <summary>
    /// Centres and scales each feature across the batch, on the tape.
    /// </summary>
    /// <remarks>
    /// The epsilon is inside the square root rather than added to the standard deviation afterwards,
    /// so a constant feature yields a finite scale instead of a division by zero, and the derivative
    /// stays finite there too.
    /// </remarks>
    // Instance, not static: Engine is an instance member of ContrastiveLossBase now that the base
    // owns it, so a static helper cannot reach it.
    private Tensor<T> StandardizeOverBatch(Tensor<T> z)
    {
        var batchAxis = new[] { 0 };
        var mean = Engine.ReduceMean(z, batchAxis, keepDims: true);
        var centred = Engine.TensorSubtract(z, mean);
        var variance = Engine.ReduceMean(Engine.TensorMultiply(centred, centred), batchAxis, keepDims: true);
        var scale = Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(BatchVarianceEpsilon)));

        return Engine.TensorDivide(centred, scale);
    }

    /// <summary>Constant [dim, dim] identity. Constant DATA, so building it by index costs no gradient.</summary>

    /// <summary>
    /// Constant [dim, dim] weights: 1 on the diagonal, <paramref name="lambda"/> off it.
    /// </summary>
}
