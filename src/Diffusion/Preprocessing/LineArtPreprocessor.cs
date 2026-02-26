using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Line art extraction preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Extracts clean line art from images, producing a single-channel sketch-like output.
/// Unlike edge detection, line art preserves artistic line quality and thickness variation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This turns your photo into a clean line drawing, like a coloring
/// book page. It's different from edge detection because it focuses on artistic lines
/// rather than just boundaries.
/// </para>
/// </remarks>
public class LineArtPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.LineArt;
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
                    // Laplacian for line extraction
                    double center = NumOps.ToDouble(data[b, 0, h, w]) * 4;
                    double neighbors = NumOps.ToDouble(data[b, 0, h - 1, w])
                                     + NumOps.ToDouble(data[b, 0, h + 1, w])
                                     + NumOps.ToDouble(data[b, 0, h, w - 1])
                                     + NumOps.ToDouble(data[b, 0, h, w + 1]);
                    double line = Math.Min(1.0, Math.Abs(center - neighbors) * 3.0);

                    // Invert so lines are white on black
                    result[b, 0, h, w] = NumOps.FromDouble(line);
                }
            }
        }

        return result;
    }
}
