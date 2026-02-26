using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Normal map estimation preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Estimates surface normals from image gradients, producing a 3-channel normal map
/// where RGB channels represent the X, Y, Z components of surface normals.
/// </para>
/// <para>
/// <b>For Beginners:</b> A normal map shows which direction each surface in the image
/// is facing. The R/G/B channels encode the X/Y/Z direction. This helps ControlNet
/// generate images with correct lighting and surface detail.
/// </para>
/// </remarks>
public class NormalMapPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Normal;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var shape = data.Shape;
        int batch = shape[0];
        int height = shape[2];
        int width = shape[3];
        var result = new Tensor<T>(new[] { batch, 3, height, width });

        for (int b = 0; b < batch; b++)
        {
            for (int h = 1; h < height - 1; h++)
            {
                for (int w = 1; w < width - 1; w++)
                {
                    double center = NumOps.ToDouble(data[b, 0, h, w]);
                    double dx = NumOps.ToDouble(data[b, 0, h, w + 1]) - NumOps.ToDouble(data[b, 0, h, w - 1]);
                    double dy = NumOps.ToDouble(data[b, 0, h + 1, w]) - NumOps.ToDouble(data[b, 0, h - 1, w]);

                    // Normal = normalize(-dx, -dy, 1)
                    double mag = Math.Sqrt(dx * dx + dy * dy + 1.0);
                    result[b, 0, h, w] = NumOps.FromDouble((-dx / mag + 1.0) * 0.5); // X -> R
                    result[b, 1, h, w] = NumOps.FromDouble((-dy / mag + 1.0) * 0.5); // Y -> G
                    result[b, 2, h, w] = NumOps.FromDouble((1.0 / mag + 1.0) * 0.5); // Z -> B
                }
            }
        }

        return result;
    }
}
