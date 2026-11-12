using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Soft edge detection preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Produces soft (non-binary) edge maps with smooth transitions, providing more
/// flexible structural guidance than hard Canny edges.
/// </para>
/// <para>
/// <b>For Beginners:</b> Unlike Canny edges which are sharp black-and-white lines,
/// soft edges have smooth gradients. This gives ControlNet more flexibility in how
/// strictly it follows the structure.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SoftEdgePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.SoftEdge;
    /// <inheritdoc />
    public override int OutputChannels => 1;

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        if (!PreprocessorFilters.HasInterior(shape[2], shape[3]))
            return new Tensor<T>(new[] { shape[0], 1, shape[2], shape[3] });

        // Sobel gradient magnitude of channel 0 over the interior, capped at 1; the border stays 0
        var gradients = PreprocessorFilters.Interior3x3(PreprocessorFilters.Channel(data, 0), PreprocessorFilters.Sobel);
        var softEdge = engine.TensorMin(PreprocessorFilters.Magnitude(gradients), NumOps.One);
        return PreprocessorFilters.PadBorder(softEdge);
    }
}
