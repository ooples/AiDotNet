using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Scribble/sketch preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Converts images into simplified scribble-like sketches by applying thresholded
/// edge detection with line thinning. Produces binary (black/white) output similar
/// to hand-drawn scribbles.
/// </para>
/// <para>
/// <b>For Beginners:</b> This turns your image into a rough sketch, like someone
/// quickly drew it with a pen. Unlike line art, scribbles are simpler and less
/// detailed, giving ControlNet more creative freedom.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ScribblePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly double _threshold;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Scribble;
    /// <inheritdoc />
    public override int OutputChannels => 1;

    /// <summary>
    /// Initializes a new scribble preprocessor.
    /// </summary>
    /// <param name="threshold">Binary threshold for scribble detection. Default: 0.3.</param>
    public ScribblePreprocessor(double threshold = 0.3)
    {
        _threshold = threshold;
    }

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        if (!PreprocessorFilters.HasInterior(shape[2], shape[3]))
            return new Tensor<T>(new[] { shape[0], 1, shape[2], shape[3] });

        // Binary threshold of the Laplacian edge strength |4 c - neighbours| of channel 0; the border stays 0
        var laplacian = PreprocessorFilters.Interior3x3(PreprocessorFilters.Channel(data, 0), PreprocessorFilters.Laplacian);
        return PreprocessorFilters.PadBorder(PreprocessorFilters.Above(engine.TensorAbs(laplacian), _threshold));
    }
}
