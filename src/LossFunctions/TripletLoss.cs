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

            var positiveDistance = EuclideanDistance(anchorSample, positiveSample);
            var negativeDistance = EuclideanDistance(anchorSample, negativeSample);

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
    /// Calculates the gradients of the Triplet Loss function for anchor, positive, and negative samples.
    /// </summary>
    /// <param name="anchor">The anchor samples matrix.</param>
    /// <param name="positive">The positive samples matrix (similar to anchor).</param>
    /// <param name="negative">The negative samples matrix (dissimilar to anchor).</param>
    /// <returns>A tuple containing the gradients for anchor, positive, and negative samples.</returns>
    /// <exception cref="ArgumentException">Thrown when input matrices have inconsistent dimensions.</exception>
    public (Matrix<T>, Matrix<T>, Matrix<T>) CalculateDerivative(Matrix<T> anchor, Matrix<T> positive, Matrix<T> negative)
    {
        // Validate input dimensions
        if (anchor.Rows != positive.Rows || anchor.Rows != negative.Rows ||
            anchor.Columns != positive.Columns || anchor.Columns != negative.Columns)
        {
            throw new ArgumentException("Anchor, positive, and negative matrices must have the same dimensions.");
        }

        var batchSize = anchor.Rows;
        var featureCount = anchor.Columns;

        var anchorGradient = new Matrix<T>(batchSize, featureCount);
        var positiveGradient = new Matrix<T>(batchSize, featureCount);
        var negativeGradient = new Matrix<T>(batchSize, featureCount);

        for (int i = 0; i < batchSize; i++)
        {
            var anchorSample = anchor.GetRow(i);
            var positiveSample = positive.GetRow(i);
            var negativeSample = negative.GetRow(i);

            var positiveDistance = EuclideanDistance(anchorSample, positiveSample);
            var negativeDistance = EuclideanDistance(anchorSample, negativeSample);

            var loss = NumOps.Subtract(
                NumOps.Add(positiveDistance, _margin),
                negativeDistance
            );

            if (NumOps.GreaterThan(loss, NumOps.Zero))
            {
                // Only compute gradients if loss > 0 (the triplet is active)
                for (int j = 0; j < featureCount; j++)
                {
                    var anchorPositiveDiff = NumOps.Subtract(anchorSample[j], positiveSample[j]);
                    var anchorNegativeDiff = NumOps.Subtract(anchorSample[j], negativeSample[j]);

                    // Gradient for anchor: 2*(anchor - positive) - 2*(anchor - negative)
                    anchorGradient[i, j] = NumOps.Multiply(
                        NumOps.FromDouble(2),
                        NumOps.Subtract(anchorPositiveDiff, anchorNegativeDiff)
                    );

                    // Gradient for positive: -2*(anchor - positive)
                    positiveGradient[i, j] = NumOps.Multiply(
                        NumOps.FromDouble(-2),
                        anchorPositiveDiff
                    );

                    // Gradient for negative: 2*(anchor - negative)
                    negativeGradient[i, j] = NumOps.Multiply(
                        NumOps.FromDouble(2),
                        anchorNegativeDiff
                    );
                }
            }
            else
            {
                // If the triplet loss is zero or negative, the gradients are zero
                for (int j = 0; j < featureCount; j++)
                {
                    anchorGradient[i, j] = NumOps.Zero;
                    positiveGradient[i, j] = NumOps.Zero;
                    negativeGradient[i, j] = NumOps.Zero;
                }
            }
        }

        return (anchorGradient, positiveGradient, negativeGradient);
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
    /// This method is not used for Triplet Loss as it requires multiple input vectors.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (target) values vector.</param>
    /// <returns>Throws NotSupportedException.</returns>
    /// <exception cref="NotSupportedException">Always thrown as TripletLoss requires three input matrices.</exception>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "TripletLoss requires three input matrices (anchor, positive, negative). " +
            "Use the CalculateDerivative(Matrix<T>, Matrix<T>, Matrix<T>) method instead."
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
    /// Calculates Triplet Loss on GPU for batched input tensors.
    /// </summary>
    /// <param name="anchor">The anchor GPU tensor (batch of embeddings).</param>
    /// <param name="positive">The positive GPU tensor (similar to anchors).</param>
    /// <param name="negative">The negative GPU tensor (dissimilar to anchors).</param>
    /// <returns>A tuple containing the loss value and gradient tensors for anchor, positive, and negative.</returns>
    public (T Loss, IGpuTensor<T> AnchorGradient, IGpuTensor<T> PositiveGradient, IGpuTensor<T> NegativeGradient) CalculateLossAndGradientGpu(
        IGpuTensor<T> anchor, IGpuTensor<T> positive, IGpuTensor<T> negative)
    {
        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        var backend = engine?.GetBackend();

        if (backend == null)
        {
            // Fall back to CPU implementation
            throw new NotSupportedException("GPU backend not available for TripletLoss GPU computation.");
        }

        int batchSize = anchor.Shape[0];
        int embeddingSize = anchor.ElementCount / batchSize;
        float margin = Convert.ToSingle(NumOps.ToDouble(_margin));

        // Compute loss on GPU
        float lossValue = backend.TripletLoss(anchor.Buffer, positive.Buffer, negative.Buffer, batchSize, embeddingSize, margin);

        // Allocate gradient buffers
        var anchorGradBuffer = backend.AllocateBuffer(anchor.ElementCount);
        var positiveGradBuffer = backend.AllocateBuffer(positive.ElementCount);
        var negativeGradBuffer = backend.AllocateBuffer(negative.ElementCount);

        // Compute gradients on GPU
        backend.TripletLossBackward(anchor.Buffer, positive.Buffer, negative.Buffer,
            anchorGradBuffer, positiveGradBuffer, negativeGradBuffer,
            batchSize, embeddingSize, margin);

        // Create gradient tensors
        var anchorGradTensor = new GpuTensor<T>(backend, anchorGradBuffer, anchor.Shape, GpuTensorRole.Gradient);
        var positiveGradTensor = new GpuTensor<T>(backend, positiveGradBuffer, positive.Shape, GpuTensorRole.Gradient);
        var negativeGradTensor = new GpuTensor<T>(backend, negativeGradBuffer, negative.Shape, GpuTensorRole.Gradient);

        return (NumOps.FromDouble(lossValue), anchorGradTensor, positiveGradTensor, negativeGradTensor);
    }
}
