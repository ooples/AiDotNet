using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Contrastive Loss function for learning similarity metrics.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Contrastive Loss helps a model learn to identify whether two examples are similar or dissimilar.
/// It works with pairs of examples and their similarity label (1 for similar, 0 for dissimilar).
/// 
/// For similar pairs, the loss penalizes distance between them, encouraging them to be close together.
/// For dissimilar pairs, the loss penalizes proximity below a certain margin, encouraging them to be at least
/// that far apart.
/// 
/// The formula has two components:
/// - For similar pairs (y=1): distance²
/// - For dissimilar pairs (y=0): max(0, margin - distance)²
/// 
/// Contrastive Loss is commonly used in:
/// - Siamese neural networks
/// - Face verification systems (determining if two faces are the same person)
/// - Signature verification
/// - Any situation where you need to learn a similarity metric between pairs
/// 
/// This approach is simpler than Triplet Loss as it only requires pairs of examples rather than triplets.
/// </para>
/// </remarks>
public class ContrastiveLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The margin that enforces separation between dissimilar pairs.
    /// </summary>
    private readonly T _margin;

    /// <summary>
    /// Initializes a new instance of the ContrastiveLoss class.
    /// </summary>
    /// <param name="margin">The minimum desired distance between dissimilar examples. Default is 1.0.</param>
    public ContrastiveLoss(double margin = 1.0)
    {
        _margin = NumOps.FromDouble(margin);
    }

    /// <summary>
    /// Calculates the Contrastive Loss between two output vectors based on their similarity.
    /// </summary>
    /// <param name="output1">The first output vector.</param>
    /// <param name="output2">The second output vector.</param>
    /// <param name="similarityLabel">A value of 1 indicates similar pairs, 0 indicates dissimilar pairs.</param>
    /// <returns>The contrastive loss value.</returns>
    public T CalculateLoss(Vector<T> output1, Vector<T> output2, T similarityLabel)
    {
        // Calculate the Euclidean distance between the vectors
        T distance = EuclideanDistance(output1, output2);

        // Calculate the loss for similar pairs: y * distance²
        T similarTerm = NumOps.Multiply(
            similarityLabel,
            NumOps.Power(distance, NumOps.FromDouble(2))
        );

        // Calculate the loss for dissimilar pairs: (1-y) * max(0, margin - distance)²
        T dissimilarTerm = NumOps.Multiply(
            NumOps.Subtract(NumOps.One, similarityLabel),
            NumOps.Power(
                MathHelper.Max(NumOps.Zero, NumOps.Subtract(_margin, distance)),
                NumOps.FromDouble(2)
            )
        );

        // Total loss is the sum of both terms
        return NumOps.Add(similarTerm, dissimilarTerm);
    }

    /// <summary>
    /// Calculates the gradients of the Contrastive Loss function for both output vectors.
    /// </summary>
    /// <param name="output1">The first output vector.</param>
    /// <param name="output2">The second output vector.</param>
    /// <param name="similarityLabel">A value of 1 indicates similar pairs, 0 indicates dissimilar pairs.</param>
    /// <returns>A tuple containing the gradients for both output vectors.</returns>
    public (Vector<T>, Vector<T>) CalculateDerivative(Vector<T> output1, Vector<T> output2, T similarityLabel)
    {
        T distance = EuclideanDistance(output1, output2);
        Vector<T> grad1 = new Vector<T>(output1.Length);
        Vector<T> grad2 = new Vector<T>(output2.Length);

        for (int i = 0; i < output1.Length; i++)
        {
            T diff = NumOps.Subtract(output1[i], output2[i]);

            if (NumOps.Equals(similarityLabel, NumOps.One))
            {
                // Gradient for similar pairs: 2 * (output1 - output2)
                grad1[i] = NumOps.Multiply(NumOps.FromDouble(2), diff);
                grad2[i] = NumOps.Multiply(NumOps.FromDouble(-2), diff);
            }
            else
            {
                // For dissimilar pairs, only apply gradient if they're closer than the margin
                if (NumOps.LessThan(distance, _margin))
                {
                    // Gradient: -2 * (margin - distance) * (output1 - output2) / distance
                    T scaleFactor = NumOps.Multiply(
                        NumOps.FromDouble(-2),
                        NumOps.Divide(
                            NumOps.Subtract(_margin, distance),
                            distance
                        )
                    );

                    grad1[i] = NumOps.Multiply(scaleFactor, diff);
                    grad2[i] = NumOps.Multiply(NumOps.Negate(scaleFactor), diff);
                }
                else
                {
                    // If distance >= margin, gradient is zero
                    grad1[i] = NumOps.Zero;
                    grad2[i] = NumOps.Zero;
                }
            }
        }

        return (grad1, grad2);
    }

    /// <summary>
    /// This method is not used for Contrastive Loss as it requires two input vectors and a similarity label.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (target) values vector.</param>
    /// <returns>Throws NotSupportedException.</returns>
    /// <exception cref="NotSupportedException">Always thrown as ContrastiveLoss requires two input vectors and a similarity label.</exception>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "ContrastiveLoss requires two input vectors and a similarity label. " +
            "Use the Calculate(Vector<T>, Vector<T>, T) method instead."
        );
    }

    /// <summary>
    /// This method is not used for Contrastive Loss as it requires two input vectors and a similarity label.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (target) values vector.</param>
    /// <returns>Throws NotSupportedException.</returns>
    /// <exception cref="NotSupportedException">Always thrown as ContrastiveLoss requires two input vectors and a similarity label.</exception>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "ContrastiveLoss requires two input vectors and a similarity label. " +
            "Use the CalculateDerivative(Vector<T>, Vector<T>, T) method instead."
        );
    }

    /// <summary>
    /// Calculates the Euclidean distance between two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>A scalar value representing the Euclidean distance.</returns>
    private T EuclideanDistance(Vector<T> v1, Vector<T> v2)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            T diff = NumOps.Subtract(v1[i], v2[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }

    /// <summary>
    /// Calculates Contrastive Loss on GPU for batched input tensors.
    /// </summary>
    /// <param name="output1">The first output GPU tensor.</param>
    /// <param name="output2">The second output GPU tensor.</param>
    /// <param name="labels">The similarity labels GPU tensor (1 for similar, 0 for dissimilar).</param>
    /// <returns>A tuple containing the loss value and gradient tensors for both outputs.</returns>
    public (T Loss, IGpuTensor<T> Gradient1, IGpuTensor<T> Gradient2) CalculateLossAndGradientGpu(
        IGpuTensor<T> output1, IGpuTensor<T> output2, IGpuTensor<T> labels)
    {
        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        var backend = engine?.GetBackend();

        if (backend == null)
        {
            // Fall back to CPU implementation
            throw new NotSupportedException("GPU backend not available for ContrastiveLoss GPU computation.");
        }

        int batchSize = output1.Shape[0];
        int embeddingSize = output1.ElementCount / batchSize;
        float margin = Convert.ToSingle(NumOps.ToDouble(_margin));

        // Compute loss on GPU
        float lossValue = backend.ContrastiveLoss(output1.Buffer, output2.Buffer, labels.Buffer, batchSize, embeddingSize, margin);

        // Allocate gradient buffers
        var grad1Buffer = backend.AllocateBuffer(output1.ElementCount);
        var grad2Buffer = backend.AllocateBuffer(output2.ElementCount);

        // Compute gradients on GPU
        backend.ContrastiveBackward(output1.Buffer, output2.Buffer, labels.Buffer,
            grad1Buffer, grad2Buffer, batchSize, embeddingSize, margin);

        // Create gradient tensors
        var grad1Tensor = new GpuTensor<T>(backend, grad1Buffer, output1.Shape, GpuTensorRole.Gradient);
        var grad2Tensor = new GpuTensor<T>(backend, grad2Buffer, output2.Shape, GpuTensorRole.Gradient);

        return (NumOps.FromDouble(lossValue), grad1Tensor, grad2Tensor);
    }
}
