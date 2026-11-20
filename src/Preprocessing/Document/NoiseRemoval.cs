using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// NoiseRemoval - Document image noise removal with multiple algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// NoiseRemoval provides various filtering techniques to clean up document images
/// by reducing noise while preserving important features like text edges.
/// </para>
/// <para>
/// <b>For Beginners:</b> Scanned documents often have noise (random spots, grain).
/// This tool removes noise while keeping text clear:
///
/// - Median: Best for salt-and-pepper noise
/// - Gaussian: General smoothing
/// - Bilateral: Edge-preserving smoothing
/// - Morphological: Removes small artifacts
///
/// Key features:
/// - Multiple noise removal algorithms
/// - Edge-preserving options
/// - Configurable filter sizes
/// - Works with binary and grayscale images
///
/// Example usage:
/// <code>
/// var noiseRemoval = new NoiseRemoval&lt;float&gt;();
/// var clean = noiseRemoval.Process(noisyImage, NoiseRemovalMethod.Median);
/// </code>
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class NoiseRemoval<T> : IDisposable
{
    #region Fields

    private readonly INumericOperations<T> _numOps;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new NoiseRemoval instance.
    /// </summary>
    public NoiseRemoval()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Applies noise removal to a document image.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="method">The noise removal method to use.</param>
    /// <returns>The denoised image.</returns>
    public Tensor<T> Process(Tensor<T> image, NoiseRemovalMethod method = NoiseRemovalMethod.Median)
    {
        return method switch
        {
            NoiseRemovalMethod.Median => MedianFilter(image),
            NoiseRemovalMethod.Gaussian => GaussianBlur(image),
            NoiseRemovalMethod.Bilateral => BilateralFilter(image),
            NoiseRemovalMethod.Morphological => MorphologicalOpening(image),
            _ => MedianFilter(image)
        };
    }

    /// <summary>
    /// Applies median filtering for salt-and-pepper noise removal.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="kernelSize">Size of the filter kernel (should be odd).</param>
    public Tensor<T> MedianFilter(Tensor<T> image, int kernelSize = 3)
    {
        // Every window position (edge-replicated) stacked on a new leading axis, sorted, middle element taken
        var engine = AiDotNetEngine.Current;
        var shifts = WindowShifts(image, kernelSize / 2);
        var (sorted, _) = engine.TensorSort(engine.TensorStack(shifts, 0), 0, false);
        return engine.Reshape(engine.TensorNarrow(sorted, 0, shifts.Length / 2, 1), image.Shape.ToArray());
    }

    /// <summary>
    /// Applies Gaussian blur for general noise reduction.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="kernelSize">Size of the Gaussian kernel (should be odd).</param>
    /// <param name="sigma">Standard deviation of the Gaussian.</param>
    public Tensor<T> GaussianBlur(Tensor<T> image, int kernelSize = 5, double sigma = 1.0)
    {
        var engine = AiDotNetEngine.Current;
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];
        int halfKernel = kernelSize / 2;
        int span = 2 * halfKernel + 1;

        // Normalized Gaussian kernel exp(-(dx^2 + dy^2) / (2 sigma^2)) over the offsets -half..half
        var offsets = engine.TensorSubtractScalar(Range(span), _numOps.FromDouble(halfKernel));
        var squared = engine.TensorSquare(offsets);
        var distance = engine.TensorAdd(
            engine.Reshape(squared, new[] { span, 1 }), engine.Reshape(squared, new[] { 1, span }));
        var kernel = engine.TensorExp(engine.TensorMultiplyScalar(distance, _numOps.FromDouble(-1.0 / (2 * sigma * sigma))));
        kernel = engine.TensorDivideScalar(kernel, engine.TensorSum(kernel));

        // Correlate every channel of the edge-replicated image with the kernel
        var padded = engine.Reshape(ReplicatePad(image, halfKernel), new[] { channels, 1, height + 2 * halfKernel, width + 2 * halfKernel });
        var blurred = engine.Conv2D(padded, engine.Reshape(kernel, new[] { 1, 1, span, span }));
        return engine.Reshape(blurred, new[] { channels, height, width });
    }

    /// <summary>
    /// Applies bilateral filtering for edge-preserving smoothing.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="kernelSize">Size of the filter kernel.</param>
    /// <param name="sigmaSpace">Spatial sigma.</param>
    /// <param name="sigmaColor">Color/intensity sigma.</param>
    public Tensor<T> BilateralFilter(Tensor<T> image, int kernelSize = 5, double sigmaSpace = 75.0, double sigmaColor = 75.0)
    {
        var engine = AiDotNetEngine.Current;
        int halfKernel = kernelSize / 2;
        var colorScale = _numOps.FromDouble(-1.0 / (2 * sigmaColor * sigmaColor));
        var sum = new Tensor<T>(image.Shape.ToArray());
        var weightSum = new Tensor<T>(image.Shape.ToArray());

        int index = 0;
        var shifts = WindowShifts(image, halfKernel);
        for (int ky = -halfKernel; ky <= halfKernel; ky++)
        {
            for (int kx = -halfKernel; kx <= halfKernel; kx++)
            {
                var neighbor = shifts[index++];

                // Spatial weight times color/intensity weight
                T spatialWeight = Constant(Math.Exp(-(ky * ky + kx * kx) / (2 * sigmaSpace * sigmaSpace)));
                var colorWeight = engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorSquare(engine.TensorSubtract(image, neighbor)), colorScale));
                var weight = engine.TensorMultiplyScalar(colorWeight, spatialWeight);

                sum = engine.TensorAdd(sum, engine.TensorMultiply(neighbor, weight));
                weightSum = engine.TensorAdd(weightSum, weight);
            }
        }

        // sum / weightSum where the weights are positive, the center value otherwise
        var positive = engine.TensorGreaterThan(weightSum, _numOps.Zero);
        var filtered = engine.TensorDivide(sum, engine.TensorWhere(positive, weightSum, engine.TensorOnesLike(weightSum)));
        return engine.TensorWhere(positive, filtered, image);
    }

    /// <summary>
    /// Applies morphological opening (erosion followed by dilation) for artifact removal.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="kernelSize">Size of the structuring element.</param>
    public Tensor<T> MorphologicalOpening(Tensor<T> image, int kernelSize = 3)
    {
        // Opening = erosion followed by dilation
        var eroded = Erode(image, kernelSize);
        return Dilate(eroded, kernelSize);
    }

    /// <summary>
    /// Applies morphological closing (dilation followed by erosion) for hole filling.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="kernelSize">Size of the structuring element.</param>
    public Tensor<T> MorphologicalClosing(Tensor<T> image, int kernelSize = 3)
    {
        // Closing = dilation followed by erosion
        var dilated = Dilate(image, kernelSize);
        return Erode(dilated, kernelSize);
    }

    /// <summary>
    /// Applies morphological erosion.
    /// </summary>
    public Tensor<T> Erode(Tensor<T> image, int kernelSize = 3)
    {
        // Window minimum = -(window maximum of -image)
        var engine = AiDotNetEngine.Current;
        return engine.TensorNegate(WindowMaximum(engine.TensorNegate(image), kernelSize / 2));
    }

    /// <summary>
    /// Applies morphological dilation.
    /// </summary>
    public Tensor<T> Dilate(Tensor<T> image, int kernelSize = 3)
    {
        return WindowMaximum(image, kernelSize / 2);
    }

    /// <summary>Maximum over the edge-replicated (2 half + 1)^2 window of every pixel (max pooling, stride 1).</summary>
    private static Tensor<T> WindowMaximum(Tensor<T> image, int halfKernel)
    {
        var engine = AiDotNetEngine.Current;
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];
        var padded = engine.Reshape(ReplicatePad(image, halfKernel), new[] { channels, 1, height + 2 * halfKernel, width + 2 * halfKernel });
        var pooled = engine.MaxPool2D(padded, 2 * halfKernel + 1, 1, 0);
        return engine.Reshape(pooled, new[] { channels, height, width });
    }

    /// <summary>The image padded by <paramref name="halfKernel"/> on every side, repeating the edge pixels.</summary>
    private static Tensor<T> ReplicatePad(Tensor<T> image, int halfKernel)
    {
        var engine = AiDotNetEngine.Current;
        var rows = ClampedIndices(image.Shape[1], halfKernel);
        var columns = ClampedIndices(image.Shape[2], halfKernel);
        return engine.TensorIndexSelect(engine.TensorIndexSelect(image, rows, 1), columns, 2);
    }

    /// <summary>clamp(i, 0, size - 1) for i = -half .. size - 1 + half.</summary>
    private static Tensor<int> ClampedIndices(int size, int halfKernel)
    {
        var indices = new int[size + 2 * halfKernel];
        for (int i = 0; i < indices.Length; i++)
        {
            indices[i] = Math.Max(0, Math.Min(size - 1, i - halfKernel));
        }

        return new Tensor<int>(indices, new[] { indices.Length });
    }

    /// <summary>
    /// The edge-replicated image shifted by every window offset (ky, kx), ky and kx from -half to half with kx
    /// fastest: element [c, y, x] of shift (ky, kx) is image[c, clamp(y + ky), clamp(x + kx)].
    /// </summary>
    private static Tensor<T>[] WindowShifts(Tensor<T> image, int halfKernel)
    {
        var engine = AiDotNetEngine.Current;
        int height = image.Shape[1];
        int width = image.Shape[2];
        var padded = ReplicatePad(image, halfKernel);
        int span = 2 * halfKernel + 1;
        var shifts = new Tensor<T>[span * span];
        for (int ky = 0; ky < span; ky++)
        {
            for (int kx = 0; kx < span; kx++)
            {
                shifts[ky * span + kx] = engine.TensorNarrow(engine.TensorNarrow(padded, 1, ky, height), 2, kx, width);
            }
        }

        return shifts;
    }

    /// <summary>0, 1, ..., count - 1 as T.</summary>
    private static Tensor<T> Range(int count) =>
        Tensor<int>.FromVector(new Vector<int>(Enumerable.Range(0, count))).Cast<T>();

    /// <summary>A loop-level constant as T (one conversion per window offset, not per pixel).</summary>
    private T Constant(double value) => _numOps.FromDouble(value);

    #endregion

    #region Disposal

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources used by the noise removal utility.
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
