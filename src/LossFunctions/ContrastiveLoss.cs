using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
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
[LossCategory(LossCategory.Contrastive)]
[LossTask(LossTask.Embedding)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, ApiShape = LossApiShape.PairedEmbedding, TestInputFormat = LossTestInputFormat.SimilarityLabels, ExpectedOutput = OutputType.Distances)]
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
        T distance = VectorHelper.EuclideanDistance(output1, output2);

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
    /// Calculates Contrastive Loss on GPU for batched input tensors.
    /// </summary>
    /// <param name="output1">The first output GPU tensor.</param>
    /// <param name="output2">The second output GPU tensor.</param>
    /// <param name="labels">The similarity labels GPU tensor (1 for similar, 0 for dissimilar).</param>
    /// <returns>A tuple containing the loss value and gradient tensors for both outputs.</returns>
    public (T Loss, Tensor<T> Gradient1, Tensor<T> Gradient2) CalculateLossAndGradientGpu(
        Tensor<T> output1, Tensor<T> output2, Tensor<T> labels)
    {
        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        var backend = engine?.GetBackend();

        if (backend == null)
        {
            // Fall back to CPU implementation
            throw new NotSupportedException("GPU backend not available for ContrastiveLoss GPU computation.");
        }

        int batchSize = output1.Shape[0];
        int embeddingSize = output1.Length / batchSize;
        float margin = Convert.ToSingle(NumOps.ToDouble(_margin));

        // Compute loss on GPU
        float lossValue = backend.ContrastiveLoss(output1.Buffer, output2.Buffer, labels.Buffer, batchSize, embeddingSize, margin);

        // Allocate gradient buffers
        var grad1Buffer = backend.AllocateBuffer(output1.Length);
        var grad2Buffer = backend.AllocateBuffer(output2.Length);

        // Compute gradients on GPU
        backend.ContrastiveBackward(output1.Buffer, output2.Buffer, labels.Buffer,
            grad1Buffer, grad2Buffer, batchSize, embeddingSize, margin);

        // Create gradient tensors
        var grad1Tensor = GpuTensorHelper.UploadToGpu<T>(backend, grad1Buffer, output1._shape, GpuTensorRole.Gradient);
        var grad2Tensor = GpuTensorHelper.UploadToGpu<T>(backend, grad2Buffer, output2._shape, GpuTensorRole.Gradient);

        return (NumOps.FromDouble(lossValue), grad1Tensor, grad2Tensor);
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // Contrastive targets are binary similarity labels (0/1), not class indices.
        // Only align when predicted has a trailing singleton dim (e.g., [B] → [B,1]).
        if (target.Shape.Length == predicted.Shape.Length - 1
            && predicted.Shape[predicted.Shape.Length - 1] == 1
            && target.Length == predicted.Length)
        {
            target = Engine.Reshape(target, predicted.Shape.ToArray());
        }
        // Contrastive = mean(y * d² + (1-y) * max(0, margin - d)²)
        var squared = Engine.TensorMultiply(predicted, predicted);
        var oneMinusY = Engine.ScalarMinusTensor(NumOps.One, target);
        var marginMinusD = Engine.ScalarMinusTensor(_margin, predicted);
        var clampedMargin = Engine.ReLU(marginMinusD);
        var clampedSq = Engine.TensorMultiply(clampedMargin, clampedMargin);
        var positivePart = Engine.TensorMultiply(target, squared);
        var negativePart = Engine.TensorMultiply(oneMinusY, clampedSq);
        var result = Engine.TensorAdd(positivePart, negativePart);
        var allAxes = Enumerable.Range(0, result.Shape.Length).ToArray();
        return Engine.ReduceMean(result, allAxes, keepDims: false);
    }

    /// <summary>
    /// Computes the contrastive loss as a tape-differentiable scalar from an embedding PAIR and
    /// its similarity label.
    /// </summary>
    /// <param name="output1">First embedding of the pair.</param>
    /// <param name="output2">Second embedding of the pair, same shape.</param>
    /// <param name="similarityLabel">1 for a similar pair, 0 for a dissimilar one.</param>
    /// <returns>A rank-0 scalar tensor holding y*d^2 + (1-y)*max(0, margin - d)^2.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either embedding is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the two shapes differ.</exception>
    /// <remarks>
    /// <para>
    /// Only the DISTANCE is computed here; the pair rule itself is the two-argument
    /// <see cref="ComputeTapeLoss(Tensor{T}, Tensor{T})"/>, which already consumes a distance and a
    /// label. Keeping the rule in one place means the pair API and the pointwise API cannot
    /// disagree about what the loss is.
    /// </para>
    /// <para>
    /// The label carries no gradient -- it is data, not a parameter -- so the backward pass reaches
    /// both embeddings and nothing else.
    /// </para>
    /// </remarks>
    public Tensor<T> ComputeTapeLoss(Tensor<T> output1, Tensor<T> output2, T similarityLabel)
    {
        if (output1 is null) throw new ArgumentNullException(nameof(output1));
        if (output2 is null) throw new ArgumentNullException(nameof(output2));

        if (!output1._shape.SequenceEqual(output2._shape))
        {
            throw new ArgumentException(
                "Both embeddings must have the same shape, but were "
                + $"[{string.Join(", ", output1.Shape.ToArray())}] and "
                + $"[{string.Join(", ", output2.Shape.ToArray())}].",
                nameof(output2));
        }

        var delta = Engine.TensorSubtract(output1, output2);
        var squared = Engine.TensorMultiply(delta, delta);
        var featureAxis = new[] { squared.Shape.Length - 1 };
        var summed = Engine.ReduceSum(squared, featureAxis, keepDims: false);

        // A single un-batched pair reduces to rank 0 here, and the pointwise overload below then
        // calls ReduceMean over an EMPTY axis list, whose backward rejects the empty shape. Keeping
        // one sample as [1] makes the un-batched case take the same path as a batch of one.
        if (summed.Shape.Length == 0)
        {
            summed = Engine.Reshape(summed, new[] { 1 });
        }

        // Offset under the root: d(sqrt(x))/dx is unbounded at x = 0, and a similar pair converges
        // to exactly zero distance, so without it the gradient becomes NaN precisely when training
        // is succeeding.
        var distance = Engine.TensorSqrt(Engine.TensorAddScalar(summed, NumOps.FromDouble(1e-12)));

        var label = new Tensor<T>(distance._shape);
        label.Fill(similarityLabel);

        return ComputeTapeLoss(distance, label);
    }
}
