using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Semantic segmentation preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Produces semantic segmentation maps where each pixel is assigned a class label
/// encoded as a color. The output guides ControlNet to respect object boundaries and regions.
/// </para>
/// <para>
/// <b>For Beginners:</b> This labels every pixel in the image (sky, person, car, etc.)
/// and paints them different colors. ControlNet uses this to generate images where
/// objects are in the same regions.
/// </para>
/// </remarks>
public class SemanticSegPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Segmentation;
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
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double r = NumOps.ToDouble(data[b, 0, h, w]);
                    double g = shape[1] > 1 ? NumOps.ToDouble(data[b, 1, h, w]) : r;
                    double bv = shape[1] > 2 ? NumOps.ToDouble(data[b, 2, h, w]) : r;

                    // Simple quantization-based segmentation (placeholder)
                    int classId = (int)(r * 4) * 25 + (int)(g * 4) * 5 + (int)(bv * 4);
                    classId = classId % 150; // ADE20K has 150 classes

                    // Map class to deterministic color
                    result[b, 0, h, w] = NumOps.FromDouble((classId * 37 % 256) / 255.0);
                    result[b, 1, h, w] = NumOps.FromDouble((classId * 67 % 256) / 255.0);
                    result[b, 2, h, w] = NumOps.FromDouble((classId * 97 % 256) / 255.0);
                }
            }
        }

        return result;
    }
}
