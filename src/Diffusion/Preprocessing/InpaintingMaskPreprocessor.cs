using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Inpainting mask preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Processes inpainting masks by binarizing and optionally feathering edges.
/// The output is a single-channel mask where 1.0 indicates regions to inpaint
/// and 0.0 indicates regions to preserve.
/// </para>
/// <para>
/// <b>For Beginners:</b> This prepares a mask that tells the AI which parts
/// of the image to regenerate (white) and which to keep (black). The feathering
/// option creates smooth transitions at mask edges for more natural blending.
/// </para>
/// </remarks>
public class InpaintingMaskPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly double _binarizeThreshold;
    private readonly int _featherRadius;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.InpaintMask;
    /// <inheritdoc />
    public override int OutputChannels => 1;

    /// <summary>
    /// Initializes a new inpainting mask preprocessor.
    /// </summary>
    /// <param name="binarizeThreshold">Threshold for binarizing the mask. Default: 0.5.</param>
    /// <param name="featherRadius">Radius for edge feathering. 0 disables feathering. Default: 0.</param>
    public InpaintingMaskPreprocessor(double binarizeThreshold = 0.5, int featherRadius = 0)
    {
        _binarizeThreshold = binarizeThreshold;
        _featherRadius = featherRadius;
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
            // Binarize the mask
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double val = NumOps.ToDouble(data[b, 0, h, w]);
                    result[b, 0, h, w] = val > _binarizeThreshold ? NumOps.One : NumOps.Zero;
                }
            }

            // Apply feathering if requested
            if (_featherRadius > 0)
            {
                var feathered = new double[height, width];
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        double sum = 0;
                        int count = 0;
                        int hMin = Math.Max(0, h - _featherRadius);
                        int hMax = Math.Min(height - 1, h + _featherRadius);
                        int wMin = Math.Max(0, w - _featherRadius);
                        int wMax = Math.Min(width - 1, w + _featherRadius);

                        for (int kh = hMin; kh <= hMax; kh++)
                        {
                            for (int kw = wMin; kw <= wMax; kw++)
                            {
                                sum += NumOps.ToDouble(result[b, 0, kh, kw]);
                                count++;
                            }
                        }

                        feathered[h, w] = sum / count;
                    }
                }

                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        result[b, 0, h, w] = NumOps.FromDouble(feathered[h, w]);
                    }
                }
            }
        }

        return result;
    }
}
