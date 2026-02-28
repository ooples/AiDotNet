using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// MLSD (Mobile Line Segment Detection) preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Detects straight line segments in images, useful for architectural and geometric
/// structure preservation. The output shows detected line segments on a black background.
/// </para>
/// <para>
/// <b>For Beginners:</b> This finds straight lines in your image (walls, edges of buildings,
/// table edges). It's especially useful for architectural images where you want to
/// preserve geometric structure.
/// </para>
/// <para>
/// Reference: Gu et al., "Towards Light-weight and Real-time Line Segment Detection", AAAI 2022
/// </para>
/// </remarks>
public class MLSDPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Mlsd;
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
            // Detect strong gradients in horizontal and vertical directions
            for (int h = 1; h < height - 1; h++)
            {
                for (int w = 1; w < width - 1; w++)
                {
                    double dx = Math.Abs(NumOps.ToDouble(data[b, 0, h, w + 1]) - NumOps.ToDouble(data[b, 0, h, w - 1]));
                    double dy = Math.Abs(NumOps.ToDouble(data[b, 0, h + 1, w]) - NumOps.ToDouble(data[b, 0, h - 1, w]));

                    // Suppress non-line features (keep mostly horizontal or vertical edges)
                    double lineScore = Math.Max(dx, dy) - Math.Min(dx, dy) * 0.5;
                    lineScore = Math.Max(0, lineScore);
                    result[b, 0, h, w] = NumOps.FromDouble(Math.Min(1.0, lineScore * 4.0));
                }
            }
        }

        return result;
    }
}
