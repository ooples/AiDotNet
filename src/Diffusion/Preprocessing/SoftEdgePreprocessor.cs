using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Soft edge detection preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Produces soft (non-binary) edge maps with smooth transitions, providing more
/// flexible structural guidance than hard Canny edges.
/// </para>
/// <para>
/// <b>For Beginners:</b> Unlike Canny edges which are sharp black-and-white lines,
/// soft edges have smooth gradients. This gives ControlNet more flexibility in how
/// strictly it follows the structure.
/// </para>
/// </remarks>
public class SoftEdgePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.SoftEdge;
    /// <inheritdoc />
    public override int OutputChannels => 1;

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
                    double gx = -NumOps.ToDouble(data[b, 0, h - 1, w - 1]) + NumOps.ToDouble(data[b, 0, h - 1, w + 1])
                              - 2 * NumOps.ToDouble(data[b, 0, h, w - 1]) + 2 * NumOps.ToDouble(data[b, 0, h, w + 1])
                              - NumOps.ToDouble(data[b, 0, h + 1, w - 1]) + NumOps.ToDouble(data[b, 0, h + 1, w + 1]);
                    double gy = -NumOps.ToDouble(data[b, 0, h - 1, w - 1]) - 2 * NumOps.ToDouble(data[b, 0, h - 1, w])
                              - NumOps.ToDouble(data[b, 0, h - 1, w + 1]) + NumOps.ToDouble(data[b, 0, h + 1, w - 1])
                              + 2 * NumOps.ToDouble(data[b, 0, h + 1, w]) + NumOps.ToDouble(data[b, 0, h + 1, w + 1]);
                    double softEdge = Math.Min(1.0, Math.Sqrt(gx * gx + gy * gy));
                    result[b, 0, h, w] = NumOps.FromDouble(softEdge);
                }
            }
        }

        return result;
    }
}
