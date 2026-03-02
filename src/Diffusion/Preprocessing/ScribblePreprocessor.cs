using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Scribble/sketch preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Converts images into simplified scribble-like sketches by applying thresholded
/// edge detection with line thinning. Produces binary (black/white) output similar
/// to hand-drawn scribbles.
/// </para>
/// <para>
/// <b>For Beginners:</b> This turns your image into a rough sketch, like someone
/// quickly drew it with a pen. Unlike line art, scribbles are simpler and less
/// detailed, giving ControlNet more creative freedom.
/// </para>
/// </remarks>
public class ScribblePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly double _threshold;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Scribble;
    /// <inheritdoc />
    public override int OutputChannels => 1;

    /// <summary>
    /// Initializes a new scribble preprocessor.
    /// </summary>
    /// <param name="threshold">Binary threshold for scribble detection. Default: 0.3.</param>
    public ScribblePreprocessor(double threshold = 0.3)
    {
        _threshold = threshold;
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
            for (int h = 1; h < height - 1; h++)
            {
                for (int w = 1; w < width - 1; w++)
                {
                    // Compute Laplacian for edge detection
                    double center = NumOps.ToDouble(data[b, 0, h, w]) * 4;
                    double neighbors = NumOps.ToDouble(data[b, 0, h - 1, w])
                                     + NumOps.ToDouble(data[b, 0, h + 1, w])
                                     + NumOps.ToDouble(data[b, 0, h, w - 1])
                                     + NumOps.ToDouble(data[b, 0, h, w + 1]);
                    double edge = Math.Abs(center - neighbors);

                    // Binary threshold to produce scribble-like output
                    result[b, 0, h, w] = edge > _threshold
                        ? NumOps.One
                        : NumOps.Zero;
                }
            }
        }

        return result;
    }
}
