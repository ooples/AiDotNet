using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// SAM (Segment Anything Model) preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Produces segmentation masks using gradient-based region detection as an
/// approximation of SAM-style segmentation. Each detected region receives
/// a unique color label in the output.
/// </para>
/// <para>
/// <b>For Beginners:</b> SAM can segment any object in an image. This preprocessor
/// creates a colored map where each object gets its own color, helping ControlNet
/// understand object boundaries for generation.
/// </para>
/// <para>
/// Reference: Kirillov et al., "Segment Anything", ICCV 2023
/// </para>
/// </remarks>
public class SAMPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly double _edgeThreshold;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.SAMSegment;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <summary>
    /// Initializes a new SAM preprocessor.
    /// </summary>
    /// <param name="edgeThreshold">Threshold for region boundary detection. Default: 0.15.</param>
    public SAMPreprocessor(double edgeThreshold = 0.15)
    {
        _edgeThreshold = edgeThreshold;
    }

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
            // Simple flood-fill-like region labeling using gradient thresholds
            var labels = new int[height, width];
            int nextLabel = 1;

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    if (labels[h, w] != 0) continue;

                    // Check if this pixel is similar to its left and top neighbors
                    bool mergeLeft = w > 0 && Math.Abs(
                        NumOps.ToDouble(data[b, 0, h, w]) - NumOps.ToDouble(data[b, 0, h, w - 1])) < _edgeThreshold;
                    bool mergeTop = h > 0 && Math.Abs(
                        NumOps.ToDouble(data[b, 0, h, w]) - NumOps.ToDouble(data[b, 0, h - 1, w])) < _edgeThreshold;

                    if (mergeLeft && labels[h, w - 1] != 0)
                    {
                        labels[h, w] = labels[h, w - 1];
                    }
                    else if (mergeTop && labels[h - 1, w] != 0)
                    {
                        labels[h, w] = labels[h - 1, w];
                    }
                    else
                    {
                        labels[h, w] = nextLabel++;
                    }
                }
            }

            // Map labels to deterministic colors
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int label = labels[h, w];
                    result[b, 0, h, w] = NumOps.FromDouble((label * 47 % 256) / 255.0);
                    result[b, 1, h, w] = NumOps.FromDouble((label * 83 % 256) / 255.0);
                    result[b, 2, h, w] = NumOps.FromDouble((label * 127 % 256) / 255.0);
                }
            }
        }

        return result;
    }
}
