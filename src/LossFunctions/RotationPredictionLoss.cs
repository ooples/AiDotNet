using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Self-supervised loss function based on rotation prediction for images.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Rotation prediction is a self-supervised task where:
/// 1. Images are rotated by 0°, 90°, 180°, or 270°
/// 2. Model predicts which rotation was applied (4-class classification)
/// 3. Model learns spatial relationships and features without needing class labels
/// </para>
/// <para><b>For Beginners:</b> This teaches the model to understand image structure without labels.
///
/// Imagine showing someone 100 photos, each rotated randomly:
/// - They learn to recognize: which way is "up", spatial relationships, object orientations
/// - They don't need to know: what the objects are (no labels needed)
///
/// After this training, when you show them 5 labeled cat photos:
/// - They already understand image structure
/// - They just need to learn: "cats look like THIS"
/// - Much faster than learning everything from scratch!
///
/// <b>How it works:</b>
/// 1. Take each unlabeled image
/// 2. Create 4 versions: rotated by 0°, 90°, 180°, 270°
/// 3. Label each version: 0, 1, 2, 3 (which rotation was applied)
/// 4. Train model to predict the rotation
///
/// <b>What the model learns:</b>
/// - Edge orientations
/// - Spatial relationships
/// - Object structure
/// - "Natural" vs "unnatural" orientations
///
/// These features are very useful for actual classification tasks!
/// </para>
/// </remarks>
[LossCategory(LossCategory.Contrastive)]
[LossTask(LossTask.MultiClass)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = true, ApiShape = LossApiShape.SelfSupervised, ExpectedOutput = OutputType.Probabilities)]
public class RotationPredictionLoss<T> : LossFunctionBase<T>, ISelfSupervisedLoss<T>
{

    /// <inheritdoc/>
    public (TInput augmentedX, TOutput augmentedY) CreateTask<TInput, TOutput>(TInput input)
    {
        // Handle Matrix<T> input by treating it as flattened images
        if (input is Matrix<T> matrixInput)
        {
            return CreateTaskFromMatrix<TInput, TOutput>(matrixInput);
        }

        if (input is not Tensor<T> tensorInput)
        {
            throw new NotSupportedException(
                $"RotationPredictionLoss only supports Tensor<T> or Matrix<T> input, but received {typeof(TInput)}");
        }

        // Validate input shape (should be [N, H, W] or [N, H, W, C])
        if (tensorInput.Shape.Length < 3)
        {
            throw new ArgumentException(
                $"Input tensor must have at least 3 dimensions [N, H, W] or [N, H, W, C], " +
                $"but got shape [{string.Join(", ", tensorInput._shape)}]");
        }

        int numImages = tensorInput.Shape[0];
        int height = tensorInput.Shape[1];
        int width = tensorInput.Shape[2];
        int channels = tensorInput.Shape.Length > 3 ? tensorInput.Shape[3] : 1;

        bool copiesAnyElement = numImages > 0 && height > 0 && width > 0 && channels > 0;
        if (tensorInput.Shape.Length > 4 && copiesAnyElement)
        {
            throw new ArgumentException("Number of indices must match the tensor's rank.");
        }

        if (height != width && copiesAnyElement)
        {
            // The 90°/270° rotations keep the [height, width] layout, so they only fit square images.
            throw new ArgumentOutOfRangeException(nameof(input),
                $"Rotation prediction needs square images, but got {height}x{width}.");
        }

        // Create rotated versions (4 rotations per image), always as [N*4, H, W, C]:
        // image-major, rotation-minor, so image n's rotation r lands at row 4n + r.
        int totalRotatedImages = numImages * 4;
        var images = Engine.Reshape(tensorInput, new[] { numImages, height, width, channels });
        var augmentedX = Engine.Reshape(RotateAll(images), new[] { totalRotatedImages, height, width, channels });

        // Store rotation labels (4-class one-hot): the identity, once per image.
        var augmentedY = Engine.TensorTile(Engine.TensorEye<T>(4), new[] { numImages, 1 });

        // Cast through object is required for generic type conversion at runtime.
        // The compiler cannot verify Tensor<T> -> TInput at compile time, so the
        // intermediate cast to object is necessary to allow the unconstrained generic conversion.
        return ((TInput)(object)augmentedX, (TOutput)(object)augmentedY);
    }

    /// <summary>
    /// Builds all four rotations (0°, 90°, 180°, 270° clockwise) of every image.
    /// </summary>
    /// <param name="images">Square images shaped [N, S, S, C].</param>
    /// <returns>Rotations shaped [N, 4, S, S, C]; index 1 holds rotation class r.</returns>
    /// <remarks>
    /// Rotation class 1 (90° clockwise) maps (i, j) → (j, S-1-i): a transpose followed by a flip of
    /// the width axis. Class 2 (180°) flips both axes. Class 3 (270° clockwise) maps (i, j) →
    /// (S-1-j, i): a transpose followed by a flip of the height axis.
    /// </remarks>
    private Tensor<T> RotateAll(Tensor<T> images)
    {
        var transposed = Engine.TensorPermute(images, new[] { 0, 2, 1, 3 });
        var rotations = new[]
        {
            images,
            Engine.TensorFlip(transposed, new[] { 2 }),
            Engine.TensorFlip(images, new[] { 1, 2 }),
            Engine.TensorFlip(transposed, new[] { 1 }),
        };

        return Engine.TensorStack(rotations, 1);
    }

    /// <summary>
    /// Creates a rotation prediction task from Matrix input (treats each row as a flattened image).
    /// </summary>
    /// <remarks>
    /// For Matrix input, this method treats each row as a flattened square image
    /// and performs 2D rotations by reshaping to square, rotating, and flattening back.
    /// For example, a Matrix with rows of 784 elements is treated as 28x28 images.
    /// </remarks>
    private (TInput augmentedX, TOutput augmentedY) CreateTaskFromMatrix<TInput, TOutput>(Matrix<T> input)
    {
        int numImages = input.Rows;
        int flattenedSize = input.Columns;

        // Determine if this is a square image (e.g., 28*28=784)
        int imageSize = (int)Math.Sqrt(flattenedSize);
        if (imageSize * imageSize != flattenedSize)
        {
            throw new ArgumentException(
                $"Matrix input must have square image dimensions. " +
                $"Got {flattenedSize} elements which is not a perfect square.");
        }

        // Create augmented data with rotations: each row is a square image, rotated exactly as the
        // tensor path rotates [N, H, W, C] images (a single channel here).
        int totalAugmented = numImages * 4;
        var images = Engine.Reshape(Tensor<T>.FromMatrix(input), new[] { numImages, imageSize, imageSize, 1 });
        var augmentedX = Engine.Reshape(RotateAll(images), new[] { totalAugmented, flattenedSize }).ToMatrix();

        // Handle output type conversion
        // If TOutput is Vector<T>, the labels are class indices 0, 1, 2, 3 per image
        // If TOutput is Matrix<T>, they are the 4-class one-hot rows (consistent with tensor path)
        object outputY;
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            outputY = Engine.TensorTile(Engine.TensorArange<T>(0, 4, 1), new[] { numImages }).ToVector();
        }
        else
        {
            outputY = Engine.TensorTile(Engine.TensorEye<T>(4), new[] { numImages, 1 }).ToMatrix();
        }

        // Cast through object is required for generic type conversion at runtime
        // The compiler cannot verify Matrix<T> -> TInput at compile time
        TInput resultX = (TInput)(object)augmentedX;
        TOutput resultY = (TOutput)outputY;
        return (resultX, resultY);
    }

    /// <summary>
    /// Categorical cross-entropy loss for rotation prediction (4-class).
    /// predicted = softmax probabilities [4*N], target = one-hot labels [4*N].
    /// </summary>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> target)
    {
        T eps = NumOps.FromDouble(1e-7);
        var clipped = Engine.TensorAddScalar(predicted.AsTensor(), eps);
        var logP = SafeLogTensor(clipped, NumericalStabilityHelper.SmallEpsilon);
        T loss = NumOps.Negate(Engine.TensorSum(Engine.TensorMultiply(target.AsTensor(), logP)));
        return NumOps.Divide(loss, NumOps.FromDouble(predicted.Length / 4.0));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Categorical cross-entropy via engine ops: -mean(sum(target * log(predicted + eps)))
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        var eps = NumOps.FromDouble(1e-7);
        var clipped = Engine.TensorAddScalar(predicted, eps);
        var logPred = Engine.TensorLog(clipped);
        var weighted = Engine.TensorMultiply(target, logPred);
        var negWeighted = Engine.TensorNegate(weighted);
        var allAxes = Enumerable.Range(0, negWeighted.Shape.Length).ToArray();
        return Engine.ReduceMean(negWeighted, allAxes, keepDims: false);
    }
}
