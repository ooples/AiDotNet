using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Triplet Loss function for learning similarity embeddings.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Triplet Loss helps create embeddings (numerical representations) where similar items 
/// are close together and different items are far apart in a vector space.
/// 
/// It works with triplets of data:
/// - Anchor: A reference point (e.g., a person's face)
/// - Positive: An example similar to the anchor (e.g., another image of the same person)
/// - Negative: An example different from the anchor (e.g., an image of a different person)
/// 
/// The loss encourages the model to make the distance between the anchor and positive smaller than
/// the distance between the anchor and negative by at least a specified margin.
/// 
/// This loss function is commonly used in:
/// - Face recognition systems
/// - Image retrieval applications
/// - Recommendation systems
/// - Any task where you need to learn meaningful similarity metrics
/// 
/// By minimizing triplet loss, the model learns to create an embedding space where semantically 
/// similar items cluster together and dissimilar items are pushed apart.
/// </para>
/// </remarks>
[LossCategory(LossCategory.Ranking)]
[LossCategory(LossCategory.Contrastive)]
[LossTask(LossTask.Embedding)]
[LossTask(LossTask.Ranking)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, ApiShape = LossApiShape.TripletMatrix, ExpectedOutput = OutputType.Distances)]
public class TripletLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The margin that enforces separation between positive and negative pairs.
    /// </summary>
    private readonly T _margin;

    /// <summary>
    /// Initializes a new instance of the TripletLoss class.
    /// </summary>
    /// <param name="margin">The minimum desired difference between positive and negative distances. Default is 1.0.</param>
    public TripletLoss(double margin = 1.0)
    {
        _margin = NumOps.FromDouble(margin);
    }

    /// <summary>
    /// Calculates the Triplet Loss for embedding learning.
    /// </summary>
    /// <param name="anchor">The anchor samples (reference points).</param>
    /// <param name="positive">The positive samples (similar to anchors).</param>
    /// <param name="negative">The negative samples (dissimilar to anchors).</param>
    /// <returns>A scalar value representing the triplet loss.</returns>
    /// <exception cref="ArgumentException">Thrown when input matrices have inconsistent dimensions.</exception>
    public T CalculateLoss(Matrix<T> anchor, Matrix<T> positive, Matrix<T> negative)
    {
        // Validate input dimensions
        if (anchor.Rows != positive.Rows || anchor.Rows != negative.Rows ||
            anchor.Columns != positive.Columns || anchor.Columns != negative.Columns)
        {
            throw new ArgumentException("Anchor, positive, and negative matrices must have the same dimensions.");
        }

        var batchSize = anchor.Rows;
        var totalLoss = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            var anchorSample = anchor.GetRow(i);
            var positiveSample = positive.GetRow(i);
            var negativeSample = negative.GetRow(i);

            var positiveDistance = VectorHelper.EuclideanDistance(anchorSample, positiveSample);
            var negativeDistance = VectorHelper.EuclideanDistance(anchorSample, negativeSample);

            // max(0, positive_distance - negative_distance + margin)
            var loss = MathHelper.Max(
                NumOps.Zero,
                NumOps.Add(
                    NumOps.Subtract(positiveDistance, negativeDistance),
                    _margin
                )
            );

            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }


    /// <summary>
    /// This method is not used for Triplet Loss as it requires multiple input vectors.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (target) values vector.</param>
    /// <returns>Throws NotSupportedException.</returns>
    /// <exception cref="NotSupportedException">Always thrown as TripletLoss requires three input matrices.</exception>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "TripletLoss requires three input matrices (anchor, positive, negative). " +
            "Use the Calculate(Matrix<T>, Matrix<T>, Matrix<T>) method instead."
        );
    }

    /// <summary>
    /// Calculates Triplet Loss on GPU for batched input tensors.
    /// </summary>
    /// <param name="anchor">The anchor GPU tensor (batch of embeddings).</param>
    /// <param name="positive">The positive GPU tensor (similar to anchors).</param>
    /// <param name="negative">The negative GPU tensor (dissimilar to anchors).</param>
    /// <returns>A tuple containing the loss value and gradient tensors for anchor, positive, and negative.</returns>
    public (T Loss, Tensor<T> AnchorGradient, Tensor<T> PositiveGradient, Tensor<T> NegativeGradient) CalculateLossAndGradientGpu(
        Tensor<T> anchor, Tensor<T> positive, Tensor<T> negative)
    {
        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        var backend = engine?.GetBackend();

        if (backend == null)
        {
            // Fall back to CPU implementation
            throw new NotSupportedException("GPU backend not available for TripletLoss GPU computation.");
        }

        int batchSize = anchor.Shape[0];
        int embeddingSize = anchor.Length / batchSize;
        float margin = Convert.ToSingle(NumOps.ToDouble(_margin));

        // Compute loss on GPU
        float lossValue = backend.TripletLoss(anchor.Buffer, positive.Buffer, negative.Buffer, batchSize, embeddingSize, margin);

        // Allocate gradient buffers
        var anchorGradBuffer = backend.AllocateBuffer(anchor.Length);
        var positiveGradBuffer = backend.AllocateBuffer(positive.Length);
        var negativeGradBuffer = backend.AllocateBuffer(negative.Length);

        // Compute gradients on GPU
        backend.TripletLossBackward(anchor.Buffer, positive.Buffer, negative.Buffer,
            anchorGradBuffer, positiveGradBuffer, negativeGradBuffer,
            batchSize, embeddingSize, margin);

        // Create gradient tensors
        var anchorGradTensor = GpuTensorHelper.UploadToGpu<T>(backend, anchorGradBuffer, anchor._shape, GpuTensorRole.Gradient);
        var positiveGradTensor = GpuTensorHelper.UploadToGpu<T>(backend, positiveGradBuffer, positive._shape, GpuTensorRole.Gradient);
        var negativeGradTensor = GpuTensorHelper.UploadToGpu<T>(backend, negativeGradBuffer, negative._shape, GpuTensorRole.Gradient);

        return (NumOps.FromDouble(lossValue), anchorGradTensor, positiveGradTensor, negativeGradTensor);
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // Triplet: mean(max(0, d_pos - d_neg + margin))
        // predicted contains distance differences, target unused
        var marginTensor = new Tensor<T>(predicted._shape);
        marginTensor.Fill(_margin);
        var shifted = Engine.TensorAdd(predicted, marginTensor);
        var zeros = new Tensor<T>(shifted._shape);
        var clamped = Engine.TensorMax(shifted, zeros);
        var allAxes = Enumerable.Range(0, clamped.Shape.Length).ToArray();
        return Engine.ReduceMean(clamped, allAxes, keepDims: false);
    }

    /// <summary>
    /// Computes the triplet loss as a tape-differentiable scalar from the three embedding sets.
    /// </summary>
    /// <param name="anchor">Anchor embeddings, shaped [batch, features] (or [features] for one triplet).</param>
    /// <param name="positive">Embeddings that should be close to the anchor, same shape.</param>
    /// <param name="negative">Embeddings that should be far from the anchor, same shape.</param>
    /// <returns>A rank-0 scalar tensor holding mean(max(0, d(a,p) - d(a,n) + margin)).</returns>
    /// <exception cref="ArgumentNullException">Thrown when any argument is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the three shapes differ.</exception>
    /// <remarks>
    /// <para>
    /// This computes only the GEOMETRY -- the two Euclidean distances -- and then hands the
    /// distance difference to the two-argument <see cref="ComputeTapeLoss(Tensor{T}, Tensor{T})"/>,
    /// which owns the hinge. The margin rule therefore exists in exactly one place and the two
    /// entry points cannot drift apart.
    /// </para>
    /// <para>
    /// Gradients flow to all three inputs, which is what a triplet objective needs: the anchor is
    /// pulled toward the positive and pushed from the negative in the same backward pass.
    /// </para>
    /// </remarks>
    public Tensor<T> ComputeTapeLoss(Tensor<T> anchor, Tensor<T> positive, Tensor<T> negative)
    {
        if (anchor is null) throw new ArgumentNullException(nameof(anchor));
        if (positive is null) throw new ArgumentNullException(nameof(positive));
        if (negative is null) throw new ArgumentNullException(nameof(negative));

        if (!anchor._shape.SequenceEqual(positive._shape) || !anchor._shape.SequenceEqual(negative._shape))
        {
            throw new ArgumentException(
                "Anchor, positive and negative must have the same shape, but were "
                + $"[{string.Join(", ", anchor.Shape.ToArray())}], "
                + $"[{string.Join(", ", positive.Shape.ToArray())}] and "
                + $"[{string.Join(", ", negative.Shape.ToArray())}].",
                nameof(positive));
        }

        var difference = Engine.TensorSubtract(
            EuclideanDistance(anchor, positive),
            EuclideanDistance(anchor, negative));

        // The two-argument overload ignores its target; it reads the distance difference only.
        return ComputeTapeLoss(difference, new Tensor<T>(difference._shape));
    }

    /// <summary>
    /// Row-wise Euclidean distance between two equally shaped embedding tensors.
    /// </summary>
    /// <remarks>
    /// The small offset under the root is not cosmetic: d(sqrt(x))/dx is unbounded as x approaches
    /// zero, so two IDENTICAL embeddings -- the exact case a positive pair converges to -- would
    /// otherwise produce a NaN gradient and destroy the whole backward pass.
    /// </remarks>
    private Tensor<T> EuclideanDistance(Tensor<T> left, Tensor<T> right)
    {
        var delta = Engine.TensorSubtract(left, right);
        var squared = Engine.TensorMultiply(delta, delta);

        // Reduce the feature axis, keeping one distance per sample.
        var featureAxis = new[] { squared.Shape.Length - 1 };
        var summed = Engine.ReduceSum(squared, featureAxis, keepDims: false);

        // A single un-batched triplet reduces to rank 0, and the hinge overload then calls
        // ReduceMean over an EMPTY axis list, whose backward rejects the empty shape. Keeping one
        // sample as [1] makes the un-batched case take the same path as a batch of one.
        if (summed.Shape.Length == 0)
        {
            summed = Engine.Reshape(summed, new[] { 1 });
        }

        return Engine.TensorSqrt(Engine.TensorAddScalar(summed, NumOps.FromDouble(1e-12)));
    }
}
