using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// DocumentPreprocessor - Comprehensive document image preprocessing pipeline.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DocumentPreprocessor provides a unified interface for applying multiple
/// preprocessing operations to document images before feeding them to AI models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Document preprocessing improves model accuracy by:
///
/// - Normalizing image characteristics
/// - Removing noise and artifacts
/// - Correcting geometric distortions
/// - Enhancing text contrast
///
/// Key features:
/// - Chained preprocessing pipeline
/// - Configurable operation order
/// - Quality-aware preprocessing
/// - Batch processing support
///
/// Example usage:
/// <code>
/// var preprocessor = new DocumentPreprocessor&lt;float&gt;();
/// var processed = preprocessor.Preprocess(documentImage, options);
/// </code>
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DocumentPreprocessor<T> : IDisposable
{
    #region Fields

    private readonly INumericOperations<T> _numOps;
    private readonly Deskew<T> _deskew;
    private readonly Binarization<T> _binarization;
    private readonly NoiseRemoval<T> _noiseRemoval;
    private readonly LayoutNormalization<T> _layoutNormalization;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DocumentPreprocessor with default settings.
    /// </summary>
    public DocumentPreprocessor()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _deskew = new Deskew<T>();
        _binarization = new Binarization<T>();
        _noiseRemoval = new NoiseRemoval<T>();
        _layoutNormalization = new LayoutNormalization<T>();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Applies the full preprocessing pipeline to a document image.
    /// </summary>
    public Tensor<T> Preprocess(Tensor<T> image, DocumentPreprocessingOptions? options = null)
    {
        options ??= new DocumentPreprocessingOptions();
        var result = image;

        // Apply operations in order
        if (options.ApplyDeskew)
            result = _deskew.Process(result, options.DeskewMaxAngle);

        if (options.ApplyBinarization)
            result = _binarization.Process(result, options.BinarizationMethod);

        if (options.ApplyNoiseRemoval)
            result = _noiseRemoval.Process(result, options.NoiseRemovalMethod);

        if (options.ApplyLayoutNormalization)
            result = _layoutNormalization.Process(result, options.TargetWidth, options.TargetHeight);

        if (options.NormalizeIntensity)
            result = NormalizeIntensity(result);

        return result;
    }

    /// <summary>
    /// Applies preprocessing to multiple document images.
    /// </summary>
    public IList<Tensor<T>> PreprocessBatch(IEnumerable<Tensor<T>> images, DocumentPreprocessingOptions? options = null)
    {
        return images.Select(img => Preprocess(img, options)).ToList();
    }

    /// <summary>
    /// Converts an RGB image to grayscale.
    /// </summary>
    public Tensor<T> ToGrayscale(Tensor<T> image)
    {
        if (image.Shape.Length < 3 || image.Shape[0] == 1)
        {
            return image; // Already grayscale
        }

        var engine = AiDotNetEngine.Current;
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        // Luminance formula: 0.299*R + 0.587*G + 0.114*B
        var r = engine.TensorNarrow(image, 0, 0, 1);
        var g = channels > 1 ? engine.TensorNarrow(image, 0, 1, 1) : r;
        var b = channels > 2 ? engine.TensorNarrow(image, 0, 2, 1) : r;
        var gray = engine.TensorAdd(
            engine.TensorAdd(engine.TensorMultiplyScalar(r, _numOps.FromDouble(0.299)), engine.TensorMultiplyScalar(g, _numOps.FromDouble(0.587))),
            engine.TensorMultiplyScalar(b, _numOps.FromDouble(0.114)));
        return engine.Reshape(gray, new[] { 1, height, width });
    }

    /// <summary>
    /// Resizes an image to the specified dimensions.
    /// </summary>
    public Tensor<T> Resize(Tensor<T> image, int targetWidth, int targetHeight, InterpolationMethod method = InterpolationMethod.Bilinear)
    {
        var engine = AiDotNetEngine.Current;
        int srcHeight = image.Shape[1];
        int srcWidth = image.Shape[2];

        // Source coordinates srcX = x * (srcWidth / targetWidth), srcY = y * (srcHeight / targetHeight)
        var srcX = engine.TensorMultiplyScalar(Range(targetWidth), _numOps.FromDouble((double)srcWidth / targetWidth));
        var srcY = engine.TensorMultiplyScalar(Range(targetHeight), _numOps.FromDouble((double)srcHeight / targetHeight));

        if (method == InterpolationMethod.Nearest)
        {
            var nearestY = ClampIndex(engine.TensorRound(srcY), srcHeight);
            var nearestX = ClampIndex(engine.TensorRound(srcX), srcWidth);
            return engine.TensorIndexSelect(engine.TensorIndexSelect(image, nearestY.Cast<int>(), 1), nearestX.Cast<int>(), 2);
        }

        // Bilinear interpolation from the clamped neighbours
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

    /// <summary>
    /// Pads an image to the specified dimensions.
    /// </summary>
    public Tensor<T> Pad(Tensor<T> image, int targetWidth, int targetHeight, T paddingValue)
    {
        int padLeft = (targetWidth - image.Shape[2]) / 2;
        int padTop = (targetHeight - image.Shape[1]) / 2;
        return Place(image, targetWidth, targetHeight, padTop, padLeft, paddingValue);
    }

    /// <summary>
    /// Center crops an image to the specified dimensions.
    /// </summary>
    public Tensor<T> CenterCrop(Tensor<T> image, int targetWidth, int targetHeight)
    {
        int cropLeft = (image.Shape[2] - targetWidth) / 2;
        int cropTop = (image.Shape[1] - targetHeight) / 2;
        return Place(image, targetWidth, targetHeight, -cropTop, -cropLeft, _numOps.Zero);
    }

    #endregion

    #region Private Methods

    private Tensor<T> NormalizeIntensity(Tensor<T> image)
    {
        var engine = AiDotNetEngine.Current;
        var flat = engine.Reshape(image, new[] { image.Length });

        // Find min and max
        T min = _numOps.Negate(engine.ReduceMax(engine.TensorNegate(flat), new[] { 0 }, false)[0]);
        T max = engine.ReduceMax(flat, new[] { 0 }, false)[0];

        // Normalize to 0-1 range
        T range = _numOps.Subtract(max, min);
        if (_numOps.LessThan(range, _numOps.FromDouble(1e-10)))
        {
            range = _numOps.One;
        }

        return engine.TensorDivideScalar(engine.TensorSubtractScalar(image, min), range);
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
    /// Releases resources used by the preprocessor.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            _deskew.Dispose();
            _binarization.Dispose();
            _noiseRemoval.Dispose();
            _layoutNormalization.Dispose();
        }
        _disposed = true;
    }

    #endregion
}
