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
[ModelPaper("Barlow Twins: Self-Supervised Learning via Redundancy Reduction", "https://arxiv.org/abs/2103.03230", Year = 2021, Authors = "Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny")]
public class BarlowTwinsLoss<T> : IContrastiveLoss<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

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
    public T ComputeLoss(Tensor<T> z1, Tensor<T> z2)
    {
        if (z1 is null) throw new ArgumentNullException(nameof(z1));
        if (z2 is null) throw new ArgumentNullException(nameof(z2));

        var batchSize = z1.Shape[0];
        var dim = z1.Shape[1];

        // Normalize along batch dimension
        var z1Norm = _normalize ? BatchNormalize(z1) : z1;
        var z2Norm = _normalize ? BatchNormalize(z2) : z2;

        // Compute cross-correlation matrix C = (z1^T @ z2) / batch_size
        var crossCorr = ComputeCrossCorrelation(z1Norm, z2Norm, batchSize);

        // Compute loss: invariance (diagonal) + redundancy (off-diagonal)
        T invarianceLoss = NumOps.Zero;
        T redundancyLoss = NumOps.Zero;

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                var cij = crossCorr[i, j];

                if (i == j)
                {
                    // Invariance term: (1 - C_ii)^2
                    var diff = NumOps.Subtract(NumOps.One, cij);
                    invarianceLoss = NumOps.Add(invarianceLoss, NumOps.Multiply(diff, diff));
                }
                else
                {
                    // Redundancy term: C_ij^2
                    redundancyLoss = NumOps.Add(redundancyLoss, NumOps.Multiply(cij, cij));
                }
            }
        }

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
        T invarianceLoss = NumOps.Zero;
        T redundancyLoss = NumOps.Zero;

        // Gradient of cross-correlation w.r.t. loss
        var gradC = new T[dim * dim];

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                var cij = crossCorr[i, j];

                if (i == j)
                {
                    var diff = NumOps.Subtract(NumOps.One, cij);
                    invarianceLoss = NumOps.Add(invarianceLoss, NumOps.Multiply(diff, diff));
                    // Gradient: -2 * (1 - C_ii)
                    gradC[i * dim + j] = NumOps.Multiply(NumOps.FromDouble(-2.0), diff);
                }
                else
                {
                    redundancyLoss = NumOps.Add(redundancyLoss, NumOps.Multiply(cij, cij));
                    // Gradient: 2 * lambda * C_ij
                    gradC[i * dim + j] = NumOps.Multiply(NumOps.FromDouble(2.0 * _lambda), cij);
                }
            }
        }

        var lambdaT = NumOps.FromDouble(_lambda);
        var totalLoss = NumOps.Add(invarianceLoss, NumOps.Multiply(lambdaT, redundancyLoss));

        // Backpropagate through cross-correlation to get gradients w.r.t. z1, z2
        var gradZ1 = new T[batchSize * dim];
        var gradZ2 = new T[batchSize * dim];

        var invBatch = NumOps.FromDouble(1.0 / batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            // Extract z2Norm row and z1Norm row for this batch sample
            var z2Row = new Vector<T>(dim);
            var z1Row = new Vector<T>(dim);
            for (int d = 0; d < dim; d++)
            {
                z2Row[d] = NumOps.Multiply(z2Norm[b, d], invBatch);
                z1Row[d] = NumOps.Multiply(z1Norm[b, d], invBatch);
            }

            for (int i = 0; i < dim; i++)
            {
                // gradZ1[b,i] = Σ_j gradC[i,j] * z2Norm[b,j] / N
                var gradCRow = new Vector<T>(dim);
                for (int j = 0; j < dim; j++)
                {
                    gradCRow[j] = gradC[i * dim + j];
                }
                gradZ1[b * dim + i] = Engine.DotProduct(gradCRow, z2Row);

                // gradZ2[b,i] = Σ_j gradC[j,i] * z1Norm[b,j] / N
                var gradCCol = new Vector<T>(dim);
                for (int j = 0; j < dim; j++)
                {
                    gradCCol[j] = gradC[j * dim + i];
                }
                gradZ2[b * dim + i] = Engine.DotProduct(gradCCol, z1Row);
            }
        }

        return (totalLoss,
                new Tensor<T>(gradZ1, [batchSize, dim]),
                new Tensor<T>(gradZ2, [batchSize, dim]));
    }

    /// <summary>
    /// Computes the cross-correlation matrix between two sets of embeddings.
    /// </summary>
    public Tensor<T> ComputeCrossCorrelation(Tensor<T> z1, Tensor<T> z2, int batchSize)
    {
        var dim = z1.Shape[1];
        var result = new T[dim * dim];
        var invBatch = NumOps.FromDouble(1.0 / batchSize);

        // Pre-extract columns as Vector<T> for Engine.DotProduct
        var z1Cols = new Vector<T>[dim];
        var z2Cols = new Vector<T>[dim];
        for (int d = 0; d < dim; d++)
        {
            z1Cols[d] = new Vector<T>(batchSize);
            z2Cols[d] = new Vector<T>(batchSize);
            for (int b = 0; b < batchSize; b++)
            {
                z1Cols[d][b] = z1[b, d];
                z2Cols[d][b] = z2[b, d];
            }
        }

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                result[i * dim + j] = NumOps.Multiply(Engine.DotProduct(z1Cols[i], z2Cols[j]), invBatch);
            }
        }

        return new Tensor<T>(result, [dim, dim]);
    }

    /// <summary>
    /// Normalizes tensor along batch dimension (mean=0, std=1 for each feature).
    /// </summary>
    private Tensor<T> BatchNormalize(Tensor<T> tensor)
    {
        var batchSize = tensor.Shape[0];
        var dim = tensor.Shape[1];
        var result = new T[batchSize * dim];

        for (int j = 0; j < dim; j++)
        {
            // Compute mean
            T mean = NumOps.Zero;
            for (int i = 0; i < batchSize; i++)
            {
                mean = NumOps.Add(mean, tensor[i, j]);
            }
            mean = NumOps.Divide(mean, NumOps.FromDouble(batchSize));

            // Compute variance
            T variance = NumOps.Zero;
            for (int i = 0; i < batchSize; i++)
            {
                var diff = NumOps.Subtract(tensor[i, j], mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            variance = NumOps.Divide(variance, NumOps.FromDouble(batchSize));

            var std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-8)));

            // Normalize
            for (int i = 0; i < batchSize; i++)
            {
                result[i * dim + j] = NumOps.Divide(
                    NumOps.Subtract(tensor[i, j], mean), std);
            }
        }

        return new Tensor<T>(result, [batchSize, dim]);
    }

    /// <summary>
    /// Computes the off-diagonal sum of a matrix (useful for monitoring).
    /// </summary>
    public T OffDiagonalSum(Tensor<T> matrix)
    {
        var dim = matrix.Shape[0];
        T sum = NumOps.Zero;

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                if (i != j)
                {
                    var val = matrix[i, j];
                    sum = NumOps.Add(sum, NumOps.Multiply(val, val));
                }
            }
        }

        return sum;
    }
}
