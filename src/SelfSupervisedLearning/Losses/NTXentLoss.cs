using AiDotNet.Helpers;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for contrastive learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> NT-Xent is the loss function used in SimCLR. It encourages
/// representations of augmented views of the same image to be similar, while pushing apart
/// representations of different images.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item>For each image, create two augmented views (positive pair)</item>
/// <item>Compute similarity between all pairs in the batch</item>
/// <item>For each anchor, its positive pair should have highest similarity</item>
/// <item>All other samples in the batch serve as negatives</item>
/// </list>
///
/// <para><b>Loss formula:</b></para>
/// <code>
/// L = -log( exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
/// </code>
/// <para>Where τ (tau) is the temperature parameter.</para>
/// </remarks>
public class NTXentLoss<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _temperature;
    private readonly bool _normalize;

    /// <summary>
    /// Gets the temperature parameter.
    /// </summary>
    public double Temperature => _temperature;

    /// <summary>
    /// Initializes a new instance of the NTXentLoss class.
    /// </summary>
    /// <param name="temperature">Temperature scaling parameter (default: 0.1).</param>
    /// <param name="normalize">Whether to L2-normalize embeddings (default: true).</param>
    public NTXentLoss(double temperature = 0.1, bool normalize = true)
    {
        if (temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be positive");

        _temperature = temperature;
        _normalize = normalize;
    }

    /// <summary>
    /// Computes the NT-Xent loss for a batch of positive pairs.
    /// </summary>
    /// <param name="z1">First view embeddings [batch_size, embedding_dim].</param>
    /// <param name="z2">Second view embeddings [batch_size, embedding_dim].</param>
    /// <returns>The computed loss value.</returns>
    public T ComputeLoss(Tensor<T> z1, Tensor<T> z2)
    {
        if (z1 is null) throw new ArgumentNullException(nameof(z1));
        if (z2 is null) throw new ArgumentNullException(nameof(z2));

        var batchSize = z1.Shape[0];
        var dim = z1.Shape[1];

        // Normalize embeddings if required
        var z1Norm = _normalize ? L2Normalize(z1) : z1;
        var z2Norm = _normalize ? L2Normalize(z2) : z2;

        // Concatenate z1 and z2: [2*batch_size, dim]
        var combined = Concatenate(z1Norm, z2Norm);

        // Compute similarity matrix: [2*batch_size, 2*batch_size]
        var similarity = ComputeSimilarityMatrix(combined);

        // Apply temperature scaling
        var tempScaled = ScaleByTemperature(similarity);

        // Create mask for positive pairs
        // Positives: (i, i+batch_size) and (i+batch_size, i)
        var loss = ComputeContrastiveLoss(tempScaled, batchSize);

        return loss;
    }

    /// <summary>
    /// Computes the NT-Xent loss and returns gradients.
    /// </summary>
    /// <param name="z1">First view embeddings.</param>
    /// <param name="z2">Second view embeddings.</param>
    /// <returns>Loss value and gradients for both views.</returns>
    public (T loss, Tensor<T> gradZ1, Tensor<T> gradZ2) ComputeLossWithGradients(Tensor<T> z1, Tensor<T> z2)
    {
        var batchSize = z1.Shape[0];
        var dim = z1.Shape[1];

        // Forward pass
        var z1Norm = _normalize ? L2Normalize(z1) : z1;
        var z2Norm = _normalize ? L2Normalize(z2) : z2;

        var combined = Concatenate(z1Norm, z2Norm);
        var similarity = ComputeSimilarityMatrix(combined);
        var tempScaled = ScaleByTemperature(similarity);

        // Compute loss and gradients
        var (loss, gradCombined) = ComputeContrastiveLossWithGrad(tempScaled, batchSize, combined);

        // Split gradients back
        var gradZ1 = new T[batchSize * dim];
        var gradZ2 = new T[batchSize * dim];

        for (int i = 0; i < batchSize * dim; i++)
        {
            gradZ1[i] = gradCombined.Data.Span[i];
            gradZ2[i] = gradCombined.Data.Span[batchSize * dim + i];
        }

        return (loss, new Tensor<T>(gradZ1, [batchSize, dim]), new Tensor<T>(gradZ2, [batchSize, dim]));
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

    private Tensor<T> Concatenate(Tensor<T> a, Tensor<T> b)
    {
        var batchA = a.Shape[0];
        var batchB = b.Shape[0];
        var dim = a.Shape[1];

        var result = new T[(batchA + batchB) * dim];

        Array.Copy(a.Data.ToArray(), 0, result, 0, batchA * dim);
        Array.Copy(b.Data.ToArray(), 0, result, batchA * dim, batchB * dim);

        return new Tensor<T>(result, [batchA + batchB, dim]);
    }

    private Tensor<T> ComputeSimilarityMatrix(Tensor<T> embeddings)
    {
        var n = embeddings.Shape[0];
        var dim = embeddings.Shape[1];
        var sim = new T[n * n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T dot = NumOps.Zero;
                for (int k = 0; k < dim; k++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(embeddings[i, k], embeddings[j, k]));
                }
                sim[i * n + j] = dot;
            }
        }

        return new Tensor<T>(sim, [n, n]);
    }

    private Tensor<T> ScaleByTemperature(Tensor<T> similarity)
    {
        var result = new T[similarity.Length];
        var invTemp = NumOps.FromDouble(1.0 / _temperature);

        for (int i = 0; i < similarity.Length; i++)
        {
            result[i] = NumOps.Multiply(similarity.Data.Span[i], invTemp);
        }

        return new Tensor<T>(result, similarity.Shape);
    }

    private T ComputeContrastiveLoss(Tensor<T> similarity, int batchSize)
    {
        var n = similarity.Shape[0]; // 2 * batchSize
        T totalLoss = NumOps.Zero;
        int validPairs = 0;

        for (int i = 0; i < n; i++)
        {
            // Find positive index
            int positiveIdx = i < batchSize ? i + batchSize : i - batchSize;

            // Compute log-softmax for row i
            T maxVal = NumOps.FromDouble(double.MinValue);
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    var val = similarity[i, j];
                    if (NumOps.GreaterThan(val, maxVal)) maxVal = val;
                }
            }

            T sumExp = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    sumExp = NumOps.Add(sumExp, NumOps.Exp(NumOps.Subtract(similarity[i, j], maxVal)));
                }
            }

            var logSumExp = NumOps.Add(maxVal, NumOps.Log(sumExp));
            var positiveScore = similarity[i, positiveIdx];

            // Loss for this sample: -positive_score + log_sum_exp
            var sampleLoss = NumOps.Subtract(logSumExp, positiveScore);
            totalLoss = NumOps.Add(totalLoss, sampleLoss);
            validPairs++;
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(validPairs));
    }

    private (T loss, Tensor<T> grad) ComputeContrastiveLossWithGrad(
        Tensor<T> similarity, int batchSize, Tensor<T> embeddings)
    {
        var n = similarity.Shape[0];
        var dim = embeddings.Shape[1];
        var grad = new T[n * dim];
        T totalLoss = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            int positiveIdx = i < batchSize ? i + batchSize : i - batchSize;

            // Compute softmax probabilities
            T maxVal = NumOps.FromDouble(double.MinValue);
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    var val = similarity[i, j];
                    if (NumOps.GreaterThan(val, maxVal)) maxVal = val;
                }
            }

            T sumExp = NumOps.Zero;
            var expVals = new T[n];
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    expVals[j] = NumOps.Exp(NumOps.Subtract(similarity[i, j], maxVal));
                    sumExp = NumOps.Add(sumExp, expVals[j]);
                }
            }

            var logSumExp = NumOps.Add(maxVal, NumOps.Log(sumExp));
            totalLoss = NumOps.Add(totalLoss, NumOps.Subtract(logSumExp, similarity[i, positiveIdx]));

            // Compute gradients
            var invTemp = NumOps.FromDouble(1.0 / _temperature);
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    var prob = NumOps.Divide(expVals[j], sumExp);
                    var gradWeight = j == positiveIdx
                        ? NumOps.Subtract(prob, NumOps.One)
                        : prob;

                    gradWeight = NumOps.Multiply(gradWeight, invTemp);

                    for (int k = 0; k < dim; k++)
                    {
                        grad[i * dim + k] = NumOps.Add(
                            grad[i * dim + k],
                            NumOps.Multiply(gradWeight, embeddings[j, k]));
                    }
                }
            }
        }

        return (NumOps.Divide(totalLoss, NumOps.FromDouble(n)), new Tensor<T>(grad, [n, dim]));
    }
}
