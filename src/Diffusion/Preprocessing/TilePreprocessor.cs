using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Tile preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Produces a blurred/downsampled version of the input image for tile-based
/// ControlNet conditioning. This guides the model to preserve overall color
/// and composition while allowing fine detail regeneration.
/// </para>
/// <para>
/// <b>For Beginners:</b> This creates a blurry version of your image that
/// ControlNet uses to keep the same colors and general layout, while letting
/// the AI add sharp details. It's commonly used for upscaling and detail enhancement.
/// </para>
/// </remarks>
public class TilePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly int _blurRadius;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Tile;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <summary>
    /// Initializes a new tile preprocessor.
    /// </summary>
    /// <param name="blurRadius">Radius of the Gaussian-like blur. Default: 4.</param>
    public TilePreprocessor(int blurRadius = 4)
    {
        _blurRadius = blurRadius;
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
            for (int c = 0; c < 3; c++)
            {
                int srcC = Math.Min(c, channels - 1);
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        // Box blur approximation
                        double sum = 0;
                        int count = 0;
                        int hMin = Math.Max(0, h - _blurRadius);
                        int hMax = Math.Min(height - 1, h + _blurRadius);
                        int wMin = Math.Max(0, w - _blurRadius);
                        int wMax = Math.Min(width - 1, w + _blurRadius);

                        for (int kh = hMin; kh <= hMax; kh++)
                        {
                            for (int kw = wMin; kw <= wMax; kw++)
                            {
                                sum += NumOps.ToDouble(data[b, srcC, kh, kw]);
                                count++;
                            }
                        }

                        result[b, c, h, w] = NumOps.FromDouble(sum / count);
                    }
                }
            }
        }

        return result;
    }
}
