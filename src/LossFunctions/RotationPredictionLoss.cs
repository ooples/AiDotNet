using AiDotNet.Helpers;
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
public class RotationPredictionLoss<T> : ISelfSupervisedLoss<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public (TInput augmentedX, TOutput augmentedY) CreateTask<TInput, TOutput>(TInput input)
    {
        if (input is not Tensor<T> tensorInput)
        {
            throw new NotSupportedException(
                $"RotationPredictionLoss only supports Tensor<T> input, but received {typeof(TInput)}");
        }

        // Validate input shape (should be [N, H, W] or [N, H, W, C])
        if (tensorInput.Shape.Length < 3)
        {
            throw new ArgumentException(
                $"Input tensor must have at least 3 dimensions [N, H, W] or [N, H, W, C], " +
                $"but got shape [{string.Join(", ", tensorInput.Shape)}]");
        }

        int numImages = tensorInput.Shape[0];
        int height = tensorInput.Shape[1];
        int width = tensorInput.Shape[2];
        int channels = tensorInput.Shape.Length > 3 ? tensorInput.Shape[3] : 1;

        // Create rotated versions (4 rotations per image)
        int totalRotatedImages = numImages * 4;
        var augmentedX = new Tensor<T>(new[] { totalRotatedImages, height, width, channels });
        var augmentedY = new Tensor<T>(new[] { totalRotatedImages, 4 });  // 4-class one-hot

        int outputIdx = 0;
        for (int imgIdx = 0; imgIdx < numImages; imgIdx++)
        {
            // Create 4 rotations (0°, 90°, 180°, 270°)
            for (int rotationClass = 0; rotationClass < 4; rotationClass++)
            {
                // Copy rotated image to output
                RotateAndCopy(tensorInput, augmentedX, imgIdx, outputIdx, rotationClass, height, width, channels);

                // Store rotation label (one-hot encoding)
                for (int classIdx = 0; classIdx < 4; classIdx++)
                {
                    augmentedY[outputIdx, classIdx] = (classIdx == rotationClass) ? NumOps.One : NumOps.Zero;
                }

                outputIdx++;
            }
        }

        return ((TInput)(object)augmentedX, (TOutput)(object)augmentedY);
    }

    /// <summary>
    /// Rotates an image and copies it to the destination tensor.
    /// </summary>
    /// <param name="source">Source tensor containing images.</param>
    /// <param name="dest">Destination tensor for rotated images.</param>
    /// <param name="srcIdx">Index of source image.</param>
    /// <param name="destIdx">Index in destination tensor.</param>
    /// <param name="rotationClass">Rotation class (0=0°, 1=90°, 2=180°, 3=270°).</param>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    /// <param name="channels">Number of color channels.</param>
    /// <remarks>
    /// <b>Note:</b> This implementation assumes square images for correct rotation behavior.
    /// For non-square images, 90° and 270° rotations will result in distorted images,
    /// since the output dimensions remain [height, width] instead of swapping to [width, height].
    /// </remarks>
    private void RotateAndCopy(
        Tensor<T> source,
        Tensor<T> dest,
        int srcIdx,
        int destIdx,
        int rotationClass,
        int height,
        int width,
        int channels)
    {
        switch (rotationClass)
        {
            case 0:
                // No rotation (0°)
                CopyImage(source, dest, srcIdx, destIdx, height, width, channels,
                    (i, j) => (i, j));
                break;

            case 1:
                // Rotate 90° clockwise: (i, j) → (j, height-1-i)
                CopyImage(source, dest, srcIdx, destIdx, height, width, channels,
                    (i, j) => (j, height - 1 - i));
                break;

            case 2:
                // Rotate 180°: (i, j) → (height-1-i, width-1-j)
                CopyImage(source, dest, srcIdx, destIdx, height, width, channels,
                    (i, j) => (height - 1 - i, width - 1 - j));
                break;

            case 3:
                // Rotate 270° clockwise (90° counter-clockwise): (i, j) → (width-1-j, i)
                CopyImage(source, dest, srcIdx, destIdx, height, width, channels,
                    (i, j) => (width - 1 - j, i));
                break;

            default:
                throw new ArgumentException($"Invalid rotation class: {rotationClass}. Must be 0-3.");
        }
    }

    /// <summary>
    /// Copies an image with a coordinate transformation.
    /// </summary>
    /// <param name="source">Source tensor.</param>
    /// <param name="dest">Destination tensor.</param>
    /// <param name="srcIdx">Source image index.</param>
    /// <param name="destIdx">Destination image index.</param>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="transform">Coordinate transformation function (srcCoord → destCoord).</param>
    private void CopyImage(
        Tensor<T> source,
        Tensor<T> dest,
        int srcIdx,
        int destIdx,
        int height,
        int width,
        int channels,
        Func<int, int, (int, int)> transform)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                var (destI, destJ) = transform(i, j);

                // Handle 3D tensors [N, H, W] (grayscale)
                if (source.Shape.Length == 3)
                {
                    dest[destIdx, destI, destJ] = source[srcIdx, i, j];
                }
                // Handle 4D tensors [N, H, W, C] (color images)
                else
                {
                    for (int c = 0; c < channels; c++)
                    {
                        dest[destIdx, destI, destJ, c] = source[srcIdx, i, j, c];
                    }
                }
            }
        }
    }
}
