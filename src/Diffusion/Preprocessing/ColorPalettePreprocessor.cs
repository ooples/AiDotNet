using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
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
        var engine = AiDotNetEngine.Current;

        // Uniform quantization: divide color space into _numColors levels per channel and
        // quantize to the level centre, floor(v / step) * step + step / 2, clamped to [0, 1]
        double step = 1.0 / _numColors;
        var rgb = PreprocessorFilters.ToRgb(data);
        var levels = engine.TensorFloor(engine.TensorDivideScalar(rgb, NumOps.FromDouble(step)));
        var quantized = engine.TensorAddScalar(engine.TensorMultiplyScalar(levels, NumOps.FromDouble(step)), NumOps.FromDouble(step * 0.5));
        return engine.TensorClamp(quantized, NumOps.Zero, NumOps.One);
    }
}
