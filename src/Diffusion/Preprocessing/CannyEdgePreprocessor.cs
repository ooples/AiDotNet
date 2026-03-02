using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Canny edge detection preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Applies the Canny edge detection algorithm to extract edges from images.
/// The output is a single-channel binary edge map used for structural control.
/// </para>
/// <para>
/// <b>For Beginners:</b> This finds the outlines/edges in your image.
/// The result looks like a drawing showing only the borders of objects.
/// ControlNet uses this to generate new images with the same structure.
/// </para>
/// <para>
/// Reference: Canny, "A Computational Approach to Edge Detection", IEEE TPAMI 1986
/// </para>
/// </remarks>
public class CannyEdgePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly double _lowThreshold;
    private readonly double _highThreshold;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Canny;
    /// <inheritdoc />
    public override int OutputChannels => 1;

    /// <summary>
    /// Initializes a new Canny edge preprocessor.
    /// </summary>
    /// <param name="lowThreshold">Lower hysteresis threshold. Default: 100.</param>
    /// <param name="highThreshold">Upper hysteresis threshold. Default: 200.</param>
    public CannyEdgePreprocessor(double lowThreshold = 100.0, double highThreshold = 200.0)
    {
        _lowThreshold = lowThreshold;
        _highThreshold = highThreshold;
    }

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var shape = data.Shape;
        int batch = shape[0];
        int height = shape[2];
        int width = shape[3];
        var result = new Tensor<T>(new[] { batch, 1, height, width });

        for (int b = 0; b < batch; b++)
        {
            // Convert to grayscale
            var gray = new double[height, width];
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double r = NumOps.ToDouble(data[b, 0, h, w]);
                    double g = shape[1] > 1 ? NumOps.ToDouble(data[b, 1, h, w]) : r;
                    double bv = shape[1] > 2 ? NumOps.ToDouble(data[b, 2, h, w]) : r;
                    gray[h, w] = 0.299 * r + 0.587 * g + 0.114 * bv;
                }
            }

            // Compute gradient magnitude using Sobel
            for (int h = 1; h < height - 1; h++)
            {
                for (int w = 1; w < width - 1; w++)
                {
                    double gx = -gray[h - 1, w - 1] + gray[h - 1, w + 1]
                              - 2 * gray[h, w - 1] + 2 * gray[h, w + 1]
                              - gray[h + 1, w - 1] + gray[h + 1, w + 1];
                    double gy = -gray[h - 1, w - 1] - 2 * gray[h - 1, w] - gray[h - 1, w + 1]
                              + gray[h + 1, w - 1] + 2 * gray[h + 1, w] + gray[h + 1, w + 1];
                    double mag = Math.Sqrt(gx * gx + gy * gy);

                    T edgeVal = mag > _highThreshold ? NumOps.One
                              : mag > _lowThreshold ? NumOps.FromDouble(0.5)
                              : NumOps.Zero;
                    result[b, 0, h, w] = edgeVal;
                }
            }
        }

        return result;
    }
}
