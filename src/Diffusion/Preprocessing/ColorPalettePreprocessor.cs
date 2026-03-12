using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Color palette extraction preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Extracts dominant colors from an image and produces a quantized color map.
/// Each pixel is mapped to the nearest dominant color, creating a simplified
/// color palette representation for color-guided generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This reduces your image to a small number of main colors,
/// like a paint-by-numbers version. ControlNet uses this to generate new images
/// that match the color scheme of your original.
/// </para>
/// </remarks>
public class ColorPalettePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly int _numColors;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.ColorPalette;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <summary>
    /// Initializes a new color palette preprocessor.
    /// </summary>
    /// <param name="numColors">Number of colors in the palette. Default: 8.</param>
    public ColorPalettePreprocessor(int numColors = 8)
    {
        _numColors = numColors;
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
            // Uniform quantization: divide color space into _numColors levels per channel
            double step = 1.0 / _numColors;

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        int srcC = Math.Min(c, channels - 1);
                        double val = NumOps.ToDouble(data[b, srcC, h, w]);

                        // Quantize to nearest palette level
                        double quantized = Math.Floor(val / step) * step + step * 0.5;
                        quantized = Math.Min(1.0, Math.Max(0.0, quantized));

                        result[b, c, h, w] = NumOps.FromDouble(quantized);
                    }
                }
            }
        }

        return result;
    }
}
