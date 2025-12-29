
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

        // Cast through object is required for generic type conversion at runtime.
        // The compiler cannot verify Tensor<T> -> TInput at compile time, so the
        // intermediate cast to object is necessary to allow the unconstrained generic conversion.
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
                // Note: dest is always 4D [N, H, W, C] but source may be 3D
                if (source.Shape.Length == 3)
                {
                    dest[destIdx, destI, destJ, 0] = source[srcIdx, i, j];
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

        // Create augmented data with rotations
        int totalAugmented = numImages * 4;
        var augmentedX = new Matrix<T>(totalAugmented, flattenedSize);
        var augmentedY = new Matrix<T>(totalAugmented, 4);  // 4-class one-hot encoding (consistent with tensor path)

        int outputIdx = 0;
        for (int imgIdx = 0; imgIdx < numImages; imgIdx++)
        {
            for (int rotationClass = 0; rotationClass < 4; rotationClass++)
            {
                // Perform rotation on the flattened image
                for (int flatIdx = 0; flatIdx < flattenedSize; flatIdx++)
                {
                    // Convert flat index to 2D coordinates
                    int i = flatIdx / imageSize;
                    int j = flatIdx % imageSize;

                    // Apply rotation transformation
                    int srcI = i;
                    int srcJ = j;

                    switch (rotationClass)
                    {
                        case 0: // 0 degrees - no rotation
                            srcI = i;
                            srcJ = j;
                            break;
                        case 1: // 90 degrees clockwise
                            srcI = imageSize - 1 - j;
                            srcJ = i;
                            break;
                        case 2: // 180 degrees
                            srcI = imageSize - 1 - i;
                            srcJ = imageSize - 1 - j;
                            break;
                        case 3: // 270 degrees clockwise
                            srcI = j;
                            srcJ = imageSize - 1 - i;
                            break;
                    }

                    // Get source value from original image
                    int srcFlatIdx = srcI * imageSize + srcJ;
                    augmentedX[outputIdx, flatIdx] = input[imgIdx, srcFlatIdx];
                }

                // Assign rotation label (one-hot encoding, consistent with tensor path)
                for (int classIdx = 0; classIdx < 4; classIdx++)
                {
                    augmentedY[outputIdx, classIdx] = (classIdx == rotationClass) ? NumOps.One : NumOps.Zero;
                }
                outputIdx++;
            }
        }

        // Handle output type conversion
        // If TOutput is Vector<T>, convert one-hot Matrix to class indices Vector
        // If TOutput is Matrix<T>, use the one-hot Matrix directly
        object outputY;
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            // Convert one-hot Matrix to class index Vector
            var classIndices = new Vector<T>(totalAugmented);
            for (int i = 0; i < totalAugmented; i++)
            {
                // Find the class with value 1 (the rotation class)
                for (int c = 0; c < 4; c++)
                {
                    // Check if this is the 1 in the one-hot encoding
                    if (NumOps.Equals(augmentedY[i, c], NumOps.One))
                    {
                        classIndices[i] = NumOps.FromDouble(c);
                        break;
                    }
                }
            }
            outputY = classIndices;
        }
        else
        {
            outputY = augmentedY;
        }

        // Cast through object is required for generic type conversion at runtime
        // The compiler cannot verify Matrix<T> -> TInput at compile time
        TInput resultX = (TInput)(object)augmentedX;
        TOutput resultY = (TOutput)outputY;
        return (resultX, resultY);
    }
}
