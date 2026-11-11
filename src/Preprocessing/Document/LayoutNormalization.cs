using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// LayoutNormalization - Document layout normalization utilities.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LayoutNormalization provides utilities for normalizing document layouts
/// to standard sizes and aspect ratios for consistent model input.
/// </para>
/// <para>
/// <b>For Beginners:</b> Different documents have different sizes and proportions.
/// This tool standardizes them for AI models:
///
/// - Resize to target dimensions
/// - Preserve aspect ratio options
/// - Handle padding and cropping
/// - Normalize orientation
///
/// Key features:
/// - Multiple normalization strategies
/// - Aspect ratio preservation
/// - Smart padding and cropping
/// - Batch processing support
///
/// Example usage:
/// <code>
/// var normalizer = new LayoutNormalization&lt;float&gt;();
/// var normalized = normalizer.Process(document, targetWidth: 224, targetHeight: 224);
/// </code>
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LayoutNormalization<T> : IDisposable
{
    #region Fields

    private readonly INumericOperations<T> _numOps;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new LayoutNormalization instance.
    /// </summary>
    public LayoutNormalization()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Normalizes a document image to the specified dimensions.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="targetWidth">Target width.</param>
    /// <param name="targetHeight">Target height.</param>
    /// <param name="strategy">The normalization strategy to use.</param>
    /// <returns>The normalized image.</returns>
    public Tensor<T> Process(Tensor<T> image, int targetWidth = 224, int targetHeight = 224,
        NormalizationStrategy strategy = NormalizationStrategy.ResizeWithPadding)
    {
        return strategy switch
        {
            NormalizationStrategy.Stretch => Stretch(image, targetWidth, targetHeight),
            NormalizationStrategy.ResizeWithPadding => ResizeWithPadding(image, targetWidth, targetHeight),
            NormalizationStrategy.CenterCrop => CenterCrop(image, targetWidth, targetHeight),
            NormalizationStrategy.ResizeAndCrop => ResizeAndCrop(image, targetWidth, targetHeight),
            _ => ResizeWithPadding(image, targetWidth, targetHeight)
        };
    }

    /// <summary>
    /// Stretches the image to target dimensions (may distort aspect ratio).
    /// </summary>
    public Tensor<T> Stretch(Tensor<T> image, int targetWidth, int targetHeight)
    {
        return Resize(image, targetWidth, targetHeight);
    }

    /// <summary>
    /// Resizes the image preserving aspect ratio and adds padding.
    /// </summary>
    public Tensor<T> ResizeWithPadding(Tensor<T> image, int targetWidth, int targetHeight,
        T? paddingValue = default)
    {
        int height = image.Shape[1];
        int width = image.Shape[2];

        // Calculate scale to fit within target while preserving aspect ratio
        double scaleX = (double)targetWidth / width;
        double scaleY = (double)targetHeight / height;
        double scale = Math.Min(scaleX, scaleY);

        int newWidth = (int)(width * scale);
        int newHeight = (int)(height * scale);

        // Resize image
        var resized = Resize(image, newWidth, newHeight);

        // Center the resized image on a padded canvas
        T padValue = paddingValue ?? _numOps.FromDouble(1.0); // White padding for documents
        int padLeft = (targetWidth - newWidth) / 2;
        int padTop = (targetHeight - newHeight) / 2;
        return Place(resized, targetWidth, targetHeight, padTop, padLeft, padValue);
    }

    /// <summary>
    /// Crops the center of the image to target dimensions.
    /// </summary>
    public Tensor<T> CenterCrop(Tensor<T> image, int targetWidth, int targetHeight)
    {
        int height = image.Shape[1];
        int width = image.Shape[2];

        // If image is smaller, pad first
        if (width < targetWidth || height < targetHeight)
        {
            int padWidth = Math.Max(width, targetWidth);
            int padHeight = Math.Max(height, targetHeight);
            image = Pad(image, padWidth, padHeight, _numOps.FromDouble(1.0));
            height = padHeight;
            width = padWidth;
        }

        int cropLeft = (width - targetWidth) / 2;
        int cropTop = (height - targetHeight) / 2;

        var engine = AiDotNetEngine.Current;
        return engine.TensorNarrow(engine.TensorNarrow(image, 1, cropTop, targetHeight), 2, cropLeft, targetWidth).Clone();
    }

    /// <summary>
    /// Resizes to cover target dimensions then crops to exact size.
    /// </summary>
    public Tensor<T> ResizeAndCrop(Tensor<T> image, int targetWidth, int targetHeight)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        // Calculate scale to cover target while preserving aspect ratio
        double scaleX = (double)targetWidth / width;
        double scaleY = (double)targetHeight / height;
        double scale = Math.Max(scaleX, scaleY);

        int newWidth = (int)(width * scale);
        int newHeight = (int)(height * scale);

        // Resize image
        var resized = Resize(image, newWidth, newHeight);

        // Center crop to target size
        return CenterCrop(resized, targetWidth, targetHeight);
    }

    /// <summary>
    /// Detects and corrects document orientation (portrait vs landscape).
    /// </summary>
    public Tensor<T> NormalizeOrientation(Tensor<T> image, bool preferPortrait = true)
    {
        int height = image.Shape[1];
        int width = image.Shape[2];

        bool isPortrait = height > width;

        // Rotate if orientation doesn't match preference
        if (preferPortrait && !isPortrait)
            return Rotate90(image, clockwise: true);
        else if (!preferPortrait && isPortrait)
            return Rotate90(image, clockwise: true);

        return image;
    }

    /// <summary>
    /// Rotates the image 90 degrees.
    /// </summary>
    public Tensor<T> Rotate90(Tensor<T> image, bool clockwise = true)
    {
        // Clockwise: dst[y, x] = src[newWidth - 1 - x, y]; counter-clockwise: dst[y, x] = src[x, newHeight - 1 - y]
        var engine = AiDotNetEngine.Current;
        var transposed = engine.TensorSwapAxes(image, 1, 2);
        return engine.TensorFlip(transposed, new[] { clockwise ? 2 : 1 });
    }

    /// <summary>
    /// Flips the image horizontally or vertically.
    /// </summary>
    public Tensor<T> Flip(Tensor<T> image, bool horizontal = true)
    {
        return AiDotNetEngine.Current.TensorFlip(image, new[] { horizontal ? 2 : 1 });
    }

    /// <summary>
    /// Computes aspect ratio of the image.
    /// </summary>
    public double GetAspectRatio(Tensor<T> image)
    {
        return (double)image.Shape[2] / image.Shape[1]; // width / height
    }

    #endregion

    #region Private Methods

    private Tensor<T> Resize(Tensor<T> image, int targetWidth, int targetHeight)
    {
        var engine = AiDotNetEngine.Current;
        int srcHeight = image.Shape[1];
        int srcWidth = image.Shape[2];

        // Source coordinates and clamped bilinear neighbours, separable in x and y
        var srcX = engine.TensorMultiplyScalar(Range(targetWidth), _numOps.FromDouble((double)srcWidth / targetWidth));
        var srcY = engine.TensorMultiplyScalar(Range(targetHeight), _numOps.FromDouble((double)srcHeight / targetHeight));
        var x0 = ClampIndex(engine.TensorFloor(srcX), srcWidth);
        var y0 = ClampIndex(engine.TensorFloor(srcY), srcHeight);
        var x1 = ClampIndex(engine.TensorAddScalar(x0, _numOps.One), srcWidth);
        var y1 = ClampIndex(engine.TensorAddScalar(y0, _numOps.One), srcHeight);
        var fx = engine.Reshape(engine.TensorSubtract(srcX, x0), new[] { 1, 1, targetWidth });
        var fy = engine.Reshape(engine.TensorSubtract(srcY, y0), new[] { 1, targetHeight, 1 });

        var rows0 = engine.TensorIndexSelect(image, y0.Cast<int>(), 1);
        var rows1 = engine.TensorIndexSelect(image, y1.Cast<int>(), 1);
        var x0Index = x0.Cast<int>();
        var x1Index = x1.Cast<int>();
        var oneMinusFx = engine.TensorSubtract(engine.TensorOnesLike(fx), fx);
        var oneMinusFy = engine.TensorSubtract(engine.TensorOnesLike(fy), fy);

        var top = engine.TensorAdd(
            engine.TensorMultiply(engine.TensorIndexSelect(rows0, x0Index, 2), oneMinusFx),
            engine.TensorMultiply(engine.TensorIndexSelect(rows0, x1Index, 2), fx));
        var bottom = engine.TensorAdd(
            engine.TensorMultiply(engine.TensorIndexSelect(rows1, x0Index, 2), oneMinusFx),
            engine.TensorMultiply(engine.TensorIndexSelect(rows1, x1Index, 2), fx));
        return engine.TensorAdd(engine.TensorMultiply(top, oneMinusFy), engine.TensorMultiply(bottom, fy));
    }

    private Tensor<T> Pad(Tensor<T> image, int targetWidth, int targetHeight, T paddingValue)
    {
        int padLeft = (targetWidth - image.Shape[2]) / 2;
        int padTop = (targetHeight - image.Shape[1]) / 2;
        return Place(image, targetWidth, targetHeight, padTop, padLeft, paddingValue);
    }

    /// <summary>0, 1, ..., count - 1 as T.</summary>
    private static Tensor<T> Range(int count) =>
        Tensor<int>.FromVector(new Vector<int>(Enumerable.Range(0, count))).Cast<T>();

    /// <summary>Clamps index values to [0, size - 1].</summary>
    private Tensor<T> ClampIndex(Tensor<T> index, int size) =>
        AiDotNetEngine.Current.TensorClamp(index, _numOps.Zero, _numOps.FromDouble(size - 1));

    /// <summary>
    /// A [channels, targetHeight, targetWidth] canvas filled with <paramref name="fill"/> holding
    /// image[c, y - offsetY, x - offsetX] wherever that source pixel exists.
    /// </summary>
    private Tensor<T> Place(Tensor<T> image, int targetWidth, int targetHeight, int offsetY, int offsetX, T fill)
    {
        var engine = AiDotNetEngine.Current;
        int channels = image.Shape[0];
        var (rowSource, rowValid) = SourceIndices(targetHeight, offsetY, image.Shape[1]);
        var (columnSource, columnValid) = SourceIndices(targetWidth, offsetX, image.Shape[2]);

        var gathered = engine.TensorIndexSelect(engine.TensorIndexSelect(image, rowSource, 1), columnSource, 2);
        var valid = engine.TensorMultiply(
            engine.Reshape(rowValid.Cast<T>(), new[] { 1, targetHeight, 1 }),
            engine.Reshape(columnValid.Cast<T>(), new[] { 1, 1, targetWidth }));
        var canvas = Tensor<T>.CreateDefault(new[] { channels, targetHeight, targetWidth }, fill);
        return engine.TensorWhere(engine.TensorMultiply(engine.TensorOnesLike(canvas), valid), gathered, canvas);
    }

    /// <summary>
    /// For target positions 0..count-1 the clamped source index position - offset and whether it lies in [0, size).
    /// </summary>
    private static (Tensor<int> Source, Tensor<int> Valid) SourceIndices(int count, int offset, int size)
    {
        var source = new int[count];
        var valid = new int[count];
        for (int i = 0; i < count; i++)
        {
            int s = i - offset;
            valid[i] = s >= 0 && s < size ? 1 : 0;
            source[i] = Math.Max(0, Math.Min(size - 1, s));
        }

        return (new Tensor<int>(source, new[] { count }), new Tensor<int>(valid, new[] { count }));
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources used by the layout normalization utility.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            // No managed resources to dispose
        }
        _disposed = true;
    }

    #endregion
}
