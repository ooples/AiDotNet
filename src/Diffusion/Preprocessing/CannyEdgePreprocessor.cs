using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Canny edge detection preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Applies the Canny edge detection algorithm to extract edges from images.
/// The output is a single-channel binary edge map used for structural control.
/// </para>
/// <para>
/// <b>For Beginners:</b> This finds the outlines/edges in your image.
/// The result looks like a drawing showing only the borders of objects.
/// ControlNet uses this to generate new images with the same structure.
/// </para>
/// <para>
/// Reference: Canny, "A Computational Approach to Edge Detection", IEEE TPAMI 1986
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CannyEdgePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly double _lowThreshold;
    private readonly double _highThreshold;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Canny;
    /// <inheritdoc />
    public override int OutputChannels => 1;

    /// <summary>
    /// Initializes a new Canny edge preprocessor.
    /// </summary>
    /// <param name="lowThreshold">Lower hysteresis threshold. Default: 100.</param>
    /// <param name="highThreshold">Upper hysteresis threshold. Default: 200.</param>
    public CannyEdgePreprocessor(double lowThreshold = 100.0, double highThreshold = 200.0)
    {
        _lowThreshold = lowThreshold;
        _highThreshold = highThreshold;
    }

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        if (!PreprocessorFilters.HasInterior(shape[2], shape[3]))
            return new Tensor<T>(new[] { shape[0], 1, shape[2], shape[3] });

        // Sobel gradient magnitude of the BT.601 grayscale image over the interior; the border stays 0
        var gray = PreprocessorFilters.Grayscale(data);
        var magnitude = PreprocessorFilters.Magnitude(PreprocessorFilters.Interior3x3(gray, PreprocessorFilters.Sobel));

        // 1 above the high threshold, 0.5 above the low threshold, 0 otherwise
        var zeros = new Tensor<T>(magnitude._shape);
        var weak = engine.TensorWhere(PreprocessorFilters.Above(magnitude, _lowThreshold),
            Tensor<T>.CreateDefault(magnitude._shape, NumOps.FromDouble(0.5)), zeros);
        var edges = engine.TensorWhere(PreprocessorFilters.Above(magnitude, _highThreshold),
            Tensor<T>.CreateDefault(magnitude._shape, NumOps.One), weak);
        return PreprocessorFilters.PadBorder(edges);
    }
}
