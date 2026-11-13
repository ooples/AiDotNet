using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// Deskew - Document deskewing utility using Hough transform-based angle detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Deskew detects and corrects rotation in scanned documents using Hough transform
/// analysis to find dominant line angles, then applies inverse rotation to straighten the document.
/// </para>
/// <para>
/// <b>For Beginners:</b> When documents are scanned, they often end up slightly rotated.
/// This tool detects and corrects that rotation:
///
/// - Detects dominant line angles in the document
/// - Uses Hough transform for robust angle detection
/// - Applies rotation correction
/// - Preserves document content
///
/// Key features:
/// - Automatic skew angle detection
/// - Configurable angle range
/// - High accuracy for text documents
/// - Works with various document types
///
/// Example usage:
/// <code>
/// var deskew = new Deskew&lt;float&gt;();
/// var straightened = deskew.Process(skewedImage, maxAngle: 45);
/// </code>
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class Deskew<T> : IDisposable
{
    #region Fields

    private readonly INumericOperations<T> _numOps;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new Deskew instance.
    /// </summary>
    public Deskew()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Processes an image to correct skew.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="maxAngle">Maximum skew angle to consider (in degrees).</param>
    /// <returns>The deskewed image.</returns>
    public Tensor<T> Process(Tensor<T> image, double maxAngle = 45.0)
    {
        // Detect skew angle
        double angle = DetectSkewAngle(image, maxAngle);

        // If angle is negligible, return original
        if (Math.Abs(angle) < 0.1)
            return image;

        // Apply rotation correction
        return RotateImage(image, -angle);
    }

    /// <summary>
    /// Detects the skew angle of a document image.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="maxAngle">Maximum skew angle to consider (in degrees).</param>
    /// <returns>The detected skew angle in degrees.</returns>
    public double DetectSkewAngle(Tensor<T> image, double maxAngle = 45.0)
    {
        // Convert to grayscale if needed
        var gray = ToGrayscale(image);

        // Apply edge detection (simple Sobel)
        var edges = DetectEdges(gray);

        // Use Hough transform to find dominant lines
        return HoughSkewDetection(edges, maxAngle);
    }

    /// <summary>
    /// Rotates an image by the specified angle.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="angle">The rotation angle in degrees.</param>
    /// <returns>The rotated image.</returns>
    public Tensor<T> RotateImage(Tensor<T> image, double angle)
    {
        var engine = AiDotNetEngine.Current;
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        double radians = angle * Math.PI / 180.0;
        T cos = _numOps.FromDouble(Math.Cos(radians));
        T sin = _numOps.FromDouble(Math.Sin(radians));
        int centerX = width / 2;
        int centerY = height / 2;

        // Output pixel offsets from the center as [height, 1] and [1, width]
        var dy = engine.Reshape(engine.TensorSubtractScalar(PixelRange(height), _numOps.FromDouble(centerY)), new[] { height, 1 });
        var dx = engine.Reshape(engine.TensorSubtractScalar(PixelRange(width), _numOps.FromDouble(centerX)), new[] { 1, width });

        // Translate to center, rotate, translate back
        var srcX = engine.TensorAddScalar(
            engine.TensorAdd(engine.TensorMultiplyScalar(dx, cos), engine.TensorMultiplyScalar(dy, sin)), _numOps.FromDouble(centerX));
        var srcY = engine.TensorAddScalar(
            engine.TensorSubtract(engine.TensorMultiplyScalar(dy, cos), engine.TensorMultiplyScalar(dx, sin)), _numOps.FromDouble(centerY));

        // Inside test: 0 <= srcX < width - 1 and 0 <= srcY < height - 1
        var zero = _numOps.Zero;
        var negativeX = engine.TensorLessThan(srcX, zero);
        var negativeY = engine.TensorLessThan(srcY, zero);
        var inside = engine.TensorMultiply(
            engine.TensorMultiply(engine.TensorSubtract(engine.TensorOnesLike(negativeX), negativeX), engine.TensorLessThan(srcX, _numOps.FromDouble(width - 1))),
            engine.TensorMultiply(engine.TensorSubtract(engine.TensorOnesLike(negativeY), negativeY), engine.TensorLessThan(srcY, _numOps.FromDouble(height - 1))));

        // Bilinear interpolation from the clamped neighbours
        var floorX = engine.TensorFloor(srcX);
        var floorY = engine.TensorFloor(srcY);
        var x0 = engine.TensorClamp(floorX, zero, _numOps.FromDouble(width - 1));
        var y0 = engine.TensorClamp(floorY, zero, _numOps.FromDouble(height - 1));
        var x1 = engine.TensorClamp(engine.TensorAddScalar(x0, _numOps.One), zero, _numOps.FromDouble(width - 1));
        var y1 = engine.TensorClamp(engine.TensorAddScalar(y0, _numOps.One), zero, _numOps.FromDouble(height - 1));
        var fx = engine.TensorSubtract(srcX, x0);
        var fy = engine.TensorSubtract(srcY, y0);

        var pixels = engine.Reshape(image, new[] { channels, height * width });
        var v00 = Gather(pixels, y0, x0, width);
        var v01 = Gather(pixels, y0, x1, width);
        var v10 = Gather(pixels, y1, x0, width);
        var v11 = Gather(pixels, y1, x1, width);

        var flatFx = engine.Reshape(fx, new[] { 1, height * width });
        var flatFy = engine.Reshape(fy, new[] { 1, height * width });
        var one = _numOps.One;
        var oneMinusFx = engine.TensorSubtract(engine.TensorOnesLike(flatFx), flatFx);
        var oneMinusFy = engine.TensorSubtract(engine.TensorOnesLike(flatFy), flatFy);
        var top = engine.TensorAdd(engine.TensorMultiply(v00, oneMinusFx), engine.TensorMultiply(v01, flatFx));
        var bottom = engine.TensorAdd(engine.TensorMultiply(v10, oneMinusFx), engine.TensorMultiply(v11, flatFx));
        var sampled = engine.TensorAdd(engine.TensorMultiply(top, oneMinusFy), engine.TensorMultiply(bottom, flatFy));

        // Background (white for documents) outside the source image
        var insideAll = engine.TensorMultiply(
            Tensor<T>.CreateDefault(new[] { channels, height * width }, one), engine.Reshape(inside, new[] { 1, height * width }));
        var result = engine.TensorWhere(insideAll, sampled, Tensor<T>.CreateDefault(new[] { channels, height * width }, one));
        return engine.Reshape(result, new[] { channels, height, width });
    }

    #endregion

    #region Private Methods

    private Tensor<T> ToGrayscale(Tensor<T> image)
    {
        if (image.Shape[0] == 1)
        {
            return image;
        }

        var engine = AiDotNetEngine.Current;
        int height = image.Shape[1];
        int width = image.Shape[2];
        var r = engine.TensorNarrow(image, 0, 0, 1);
        var g = image.Shape[0] > 1 ? engine.TensorNarrow(image, 0, 1, 1) : r;
        var b = image.Shape[0] > 2 ? engine.TensorNarrow(image, 0, 2, 1) : r;
        var gray = engine.TensorAdd(
            engine.TensorAdd(engine.TensorMultiplyScalar(r, _numOps.FromDouble(0.299)), engine.TensorMultiplyScalar(g, _numOps.FromDouble(0.587))),
            engine.TensorMultiplyScalar(b, _numOps.FromDouble(0.114)));
        return engine.Reshape(gray, new[] { 1, height, width });
    }

    /// <summary>
    /// Sobel gradient magnitude / 4, capped at 1, on the interior pixels; the one-pixel border is zero.
    /// </summary>
    private Tensor<T> DetectEdges(Tensor<T> gray)
    {
        var engine = AiDotNetEngine.Current;
        int height = gray.Shape[1];
        int width = gray.Shape[2];

        if (height < 3 || width < 3)
        {
            return new Tensor<T>(new[] { 1, height, width });
        }

        var input = engine.Reshape(engine.TensorNarrow(gray, 0, 0, 1), new[] { 1, 1, height, width });
        var sobelX = new Tensor<int>(new[] { -1, 0, 1, -2, 0, 2, -1, 0, 1 }, new[] { 1, 1, 3, 3 }).Cast<T>();
        var sobelY = new Tensor<int>(new[] { -1, -2, -1, 0, 0, 0, 1, 2, 1 }, new[] { 1, 1, 3, 3 }).Cast<T>();
        var gx = engine.Conv2D(input, sobelX);
        var gy = engine.Conv2D(input, sobelY);

        var magnitude = engine.TensorSqrt(engine.TensorAdd(engine.TensorSquare(gx), engine.TensorSquare(gy)));
        var interior = engine.Reshape(
            engine.TensorClampMax(engine.TensorDivideScalar(magnitude, _numOps.FromDouble(4.0)), _numOps.One),
            new[] { height - 2, width - 2 });

        // Zero border
        var columnPad = new Tensor<T>(new[] { height - 2, 1 });
        var rows = engine.TensorConcatenate(new[] { columnPad, interior, columnPad }, 1);
        var rowPad = new Tensor<T>(new[] { 1, width });
        var full = engine.TensorConcatenate(new[] { rowPad, rows, rowPad }, 0);
        return engine.Reshape(full, new[] { 1, height, width });
    }

    /// <summary>
    /// Scores every candidate angle a in [-maxAngle, maxAngle) (180 steps) by the sum over edge pixels
    /// (value &gt; 0.3) of |x cos a + y sin a| and returns the first best angle. Rows are processed in chunks to
    /// bound the [pixels, angles] projection tensor.
    /// </summary>
    private double HoughSkewDetection(Tensor<T> edges, double maxAngle)
    {
        var engine = AiDotNetEngine.Current;
        int height = edges.Shape[1];
        int width = edges.Shape[2];

        // Accumulator for angles
        int numAngles = 180;
        double angleStep = (2 * maxAngle) / numAngles;
        double threshold = 0.3; // Edge threshold

        var radians = engine.TensorMultiplyScalar(
            engine.TensorAddScalar(engine.TensorMultiplyScalar(PixelRange(numAngles), _numOps.FromDouble(angleStep)), _numOps.FromDouble(-maxAngle)),
            _numOps.FromDouble(Math.PI / 180.0));
        var cos = engine.Reshape(engine.TensorCos(radians), new[] { 1, numAngles });
        var sin = engine.Reshape(engine.TensorSin(radians), new[] { 1, numAngles });

        var edgeMask = engine.TensorGreaterThan(
            engine.Reshape(engine.TensorNarrow(edges, 0, 0, 1), new[] { height, width }), _numOps.FromDouble(threshold));
        var xs = engine.Reshape(PixelRange(width), new[] { 1, width });
        var accumulator = new Tensor<T>(new[] { numAngles });

        int rowsPerChunk = Math.Max(1, (1 << 22) / Math.Max(1, width * numAngles));
        for (int start = 0; start < height; start += rowsPerChunk)
        {
            int count = Math.Min(rowsPerChunk, height - start);
            var ys = engine.Reshape(
                engine.TensorAddScalar(PixelRange(count), _numOps.FromDouble(start)), new[] { count, 1 });
            var pixelX = engine.Reshape(engine.TensorMultiply(Tensor<T>.CreateDefault(new[] { count, width }, _numOps.One), xs), new[] { count * width, 1 });
            var pixelY = engine.Reshape(engine.TensorMultiply(Tensor<T>.CreateDefault(new[] { count, width }, _numOps.One), ys), new[] { count * width, 1 });
            var weight = engine.Reshape(engine.TensorNarrow(edgeMask, 0, start, count), new[] { count * width, 1 });

            // Project point onto line at this angle
            var rho = engine.TensorAbs(engine.TensorAdd(engine.TensorMatMul(pixelX, cos), engine.TensorMatMul(pixelY, sin)));
            accumulator = engine.TensorAdd(accumulator, engine.ReduceSum(engine.TensorMultiply(rho, weight), new[] { 0 }, false));
        }

        // Find peak in accumulator (first maximum; 0 when nothing voted)
        int maxIdx = AiDotNet.AnomalyDetection.AnomalyTensorOps.FirstArgMax(accumulator);
        return -maxAngle + maxIdx * angleStep;
    }

    /// <summary>0, 1, ..., count - 1 as T.</summary>
    private static Tensor<T> PixelRange(int count) =>
        Tensor<int>.FromVector(new Vector<int>(Enumerable.Range(0, count))).Cast<T>();

    /// <summary>Gathers pixels[c, y * width + x] for every output pixel, [channels, height * width].</summary>
    private static Tensor<T> Gather(Tensor<T> pixels, Tensor<T> y, Tensor<T> x, int width)
    {
        var engine = AiDotNetEngine.Current;
        var flat = engine.TensorAdd(engine.TensorMultiplyScalar(y.Cast<int>(), width), x.Cast<int>());
        return engine.TensorIndexSelect(pixels, engine.Reshape(flat, new[] { flat.Length }), 1);
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
    /// Releases resources used by the deskew utility.
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
