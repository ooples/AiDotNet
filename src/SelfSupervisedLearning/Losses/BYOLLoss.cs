using AiDotNet.Helpers;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// BYOL (Bootstrap Your Own Latent) Loss - a simple cosine similarity loss.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BYOL uses a simple Mean Squared Error (MSE) loss between
/// normalized predictions and normalized targets. Unlike contrastive methods, it doesn't
/// require negative samples.</para>
///
/// <para><b>Key insight:</b> BYOL avoids collapse through asymmetric architecture
/// (predictor on one branch) and momentum updates, not through negative samples.</para>
///
/// <para><b>Loss formula:</b></para>
/// <code>
/// L = 2 - 2 * (p · sg(z')) / (||p|| * ||sg(z')||)
///   = MSE(normalize(p), normalize(sg(z')))
/// </code>
/// <para>where p is prediction, z' is target projection, and sg() means stop-gradient.</para>
/// </remarks>
public class BYOLLoss<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly bool _normalize;
    private readonly bool _symmetric;

    /// <summary>
    /// Initializes a new instance of the BYOLLoss class.
    /// </summary>
    /// <param name="normalize">Whether to L2-normalize embeddings (default: true).</param>
    /// <param name="symmetric">Whether to use symmetric loss (default: true for BYOL).</param>
    public BYOLLoss(bool normalize = true, bool symmetric = true)
    {
        _normalize = normalize;
        _symmetric = symmetric;
    }

    /// <summary>
    /// Computes the BYOL loss between online predictions and target projections.
    /// </summary>
    /// <param name="onlinePrediction">Predictions from online network [batch_size, dim].</param>
    /// <param name="targetProjection">Projections from target network [batch_size, dim] (stop-gradient applied).</param>
    /// <returns>The computed loss value.</returns>
    public T ComputeLoss(Tensor<T> onlinePrediction, Tensor<T> targetProjection)
    {
        if (onlinePrediction is null) throw new ArgumentNullException(nameof(onlinePrediction));
        if (targetProjection is null) throw new ArgumentNullException(nameof(targetProjection));

        var batchSize = onlinePrediction.Shape[0];
        var dim = onlinePrediction.Shape[1];

        var p = _normalize ? L2Normalize(onlinePrediction) : onlinePrediction;
        var z = _normalize ? L2Normalize(targetProjection) : targetProjection;

        // Compute negative cosine similarity: 2 - 2 * cos(p, z)
        // This is equivalent to ||p - z||^2 when both are normalized
        T totalLoss = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            T cosineSim = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                cosineSim = NumOps.Add(cosineSim, NumOps.Multiply(p[i, d], z[i, d]));
            }

            // Loss = 2 - 2 * cosine_similarity
            var loss = NumOps.Subtract(
                NumOps.FromDouble(2.0),
                NumOps.Multiply(NumOps.FromDouble(2.0), cosineSim));

            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes symmetric BYOL loss for both views.
    /// </summary>
    /// <param name="pred1">Predictions from view 1 going to view 2.</param>
    /// <param name="proj2">Target projections from view 2.</param>
    /// <param name="pred2">Predictions from view 2 going to view 1.</param>
    /// <param name="proj1">Target projections from view 1.</param>
    /// <returns>The symmetric loss.</returns>
    public T ComputeSymmetricLoss(
        Tensor<T> pred1, Tensor<T> proj2,
        Tensor<T> pred2, Tensor<T> proj1)
    {
        var loss1 = ComputeLoss(pred1, proj2);
        var loss2 = ComputeLoss(pred2, proj1);

        return NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Add(loss1, loss2));
    }

    /// <summary>
    /// Computes BYOL loss with gradients for backpropagation.
    /// </summary>
    public (T loss, Tensor<T> gradPrediction) ComputeLossWithGradients(
        Tensor<T> onlinePrediction, Tensor<T> targetProjection)
    {
        var batchSize = onlinePrediction.Shape[0];
        var dim = onlinePrediction.Shape[1];

        var p = _normalize ? L2Normalize(onlinePrediction) : onlinePrediction;
        var z = _normalize ? L2Normalize(targetProjection) : targetProjection;

        var gradP = new T[batchSize * dim];
        T totalLoss = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            // Compute cosine similarity using normalized vectors
            T cosineSim = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                cosineSim = NumOps.Add(cosineSim, NumOps.Multiply(p[i, d], z[i, d]));
            }

            // Compute norm of the ORIGINAL (unnormalized) input for chain rule
            T origNormSq = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                origNormSq = NumOps.Add(origNormSq, NumOps.Multiply(onlinePrediction[i, d], onlinePrediction[i, d]));
            }
            T origNorm = NumOps.Sqrt(NumOps.Add(origNormSq, NumOps.FromDouble(1e-8)));

            // Loss = 2 - 2 * cos_sim
            var loss = NumOps.Subtract(
                NumOps.FromDouble(2.0),
                NumOps.Multiply(NumOps.FromDouble(2.0), cosineSim));

            totalLoss = NumOps.Add(totalLoss, loss);

            // Gradient w.r.t. unnormalized onlinePrediction:
            // d/da_d [cos(a,b)] = (z_d - cos_sim * p_d) / ||a||
            // where p = a/||a||, z = b/||b||
            // So dL/da_d = -2 * (z_d - cos_sim * p_d) / ||a||
            for (int d = 0; d < dim; d++)
            {
                var grad = NumOps.Divide(
                    NumOps.Multiply(
                        NumOps.FromDouble(-2.0),
                        NumOps.Subtract(z[i, d], NumOps.Multiply(cosineSim, p[i, d]))),
                    origNorm);

                gradP[i * dim + d] = grad;
            }
        }

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        var scale = NumOps.FromDouble(1.0 / batchSize);

        for (int i = 0; i < gradP.Length; i++)
        {
            gradP[i] = NumOps.Multiply(gradP[i], scale);
        }

        return (avgLoss, new Tensor<T>(gradP, [batchSize, dim]));
    }

    /// <summary>
    /// Computes the mean squared error between normalized embeddings.
    /// </summary>
    /// <remarks>
    /// For normalized vectors: MSE = 2 - 2*cos_sim, which is equivalent to the BYOL loss.
    /// </remarks>
    public T ComputeMSELoss(Tensor<T> prediction, Tensor<T> target)
    {
        var batchSize = prediction.Shape[0];
        var dim = prediction.Shape[1];

        var p = _normalize ? L2Normalize(prediction) : prediction;
        var z = _normalize ? L2Normalize(target) : target;

        T totalLoss = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            for (int d = 0; d < dim; d++)
            {
                var diff = NumOps.Subtract(p[i, d], z[i, d]);
                totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(diff, diff));
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize * dim));
    }

    private Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        var batchSize = tensor.Shape[0];
        var dim = tensor.Shape[1];
        var result = new T[batchSize * dim];

        for (int i = 0; i < batchSize; i++)
        {
            T sumSquared = NumOps.Zero;
            for (int j = 0; j < dim; j++)
            {
                var val = tensor[i, j];
                sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(val, val));
            }

            var norm = NumOps.Sqrt(NumOps.Add(sumSquared, NumOps.FromDouble(1e-8)));

            for (int j = 0; j < dim; j++)
            {
                result[i * dim + j] = NumOps.Divide(tensor[i, j], norm);
            }
        }

        return new Tensor<T>(result, [batchSize, dim]);
    }
}
