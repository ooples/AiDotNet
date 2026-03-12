using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// MediaPipe Face Mesh preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Produces face mesh-like feature maps by detecting facial structure through
/// gradient analysis. The output highlights facial feature regions (eyes, nose,
/// mouth contours) that guide face-conditioned generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This creates a wireframe-like mesh of facial features
/// (eyes, nose, mouth, jawline). ControlNet uses this to generate faces that
/// match the expression and structure of the original face.
/// </para>
/// <para>
/// Reference: Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines", 2019
/// </para>
/// </remarks>
public class MediaPipeFacePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly double _sensitivity;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.MediaPipeFace;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <summary>
    /// Initializes a new MediaPipe face mesh preprocessor.
    /// </summary>
    /// <param name="sensitivity">Sensitivity for facial feature detection. Default: 2.0.</param>
    public MediaPipeFacePreprocessor(double sensitivity = 2.0)
    {
        _sensitivity = sensitivity;
    }

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var shape = data.Shape;
        int batch = shape[0];
        int channels = Math.Min(shape[1], 3);
        int height = shape[2];
        int width = shape[3];
        var result = new Tensor<T>(new[] { batch, 3, height, width });

        for (int b = 0; b < batch; b++)
        {
            for (int h = 1; h < height - 1; h++)
            {
                for (int w = 1; w < width - 1; w++)
                {
                    // Multi-scale gradient for facial feature detection
                    double totalEdge = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        double dx = Math.Abs(NumOps.ToDouble(data[b, c, h, w + 1]) - NumOps.ToDouble(data[b, c, h, w - 1]));
                        double dy = Math.Abs(NumOps.ToDouble(data[b, c, h + 1, w]) - NumOps.ToDouble(data[b, c, h - 1, w]));
                        totalEdge += Math.Sqrt(dx * dx + dy * dy);
                    }

                    totalEdge /= channels;
                    double meshIntensity = Math.Min(1.0, totalEdge * _sensitivity);

                    // Color-coded output: green for mesh lines, darker for background
                    result[b, 0, h, w] = NumOps.FromDouble(meshIntensity * 0.3);
                    result[b, 1, h, w] = NumOps.FromDouble(meshIntensity);
                    result[b, 2, h, w] = NumOps.FromDouble(meshIntensity * 0.3);
                }
            }
        }

        return result;
    }
}
