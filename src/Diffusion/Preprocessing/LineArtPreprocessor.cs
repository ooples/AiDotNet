using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Line art extraction preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Extracts clean line art from images, producing a single-channel sketch-like output.
/// Unlike edge detection, line art preserves artistic line quality and thickness variation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This turns your photo into a clean line drawing, like a coloring
/// book page. It's different from edge detection because it focuses on artistic lines
/// rather than just boundaries.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LineArtPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.LineArt;
    /// <inheritdoc />
    public override int OutputChannels => 1;

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        if (!PreprocessorFilters.HasInterior(shape[2], shape[3]))
            return new Tensor<T>(new[] { shape[0], 1, shape[2], shape[3] });

        // Laplacian line response of channel 0 over the interior, min(1, 3 |4 c - neighbours|); the border stays 0
        var laplacian = PreprocessorFilters.Interior3x3(PreprocessorFilters.Channel(data, 0), PreprocessorFilters.Laplacian);
        var line = engine.TensorMin(engine.TensorMultiplyScalar(engine.TensorAbs(laplacian), NumOps.FromDouble(3.0)), NumOps.One);
        return PreprocessorFilters.PadBorder(line);
    }
}
