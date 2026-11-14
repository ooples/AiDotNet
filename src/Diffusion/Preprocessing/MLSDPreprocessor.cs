using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// MLSD (Mobile Line Segment Detection) preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Detects straight line segments in images, useful for architectural and geometric
/// structure preservation. The output shows detected line segments on a black background.
/// </para>
/// <para>
/// <b>For Beginners:</b> This finds straight lines in your image (walls, edges of buildings,
/// table edges). It's especially useful for architectural images where you want to
/// preserve geometric structure.
/// </para>
/// <para>
/// Reference: Gu et al., "Towards Light-weight and Real-time Line Segment Detection", AAAI 2022
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MLSDPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Mlsd;
    /// <inheritdoc />
    public override int OutputChannels => 1;

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        if (!PreprocessorFilters.HasInterior(shape[2], shape[3]))
            return new Tensor<T>(new[] { shape[0], 1, shape[2], shape[3] });

        // Detect strong gradients in horizontal and vertical directions (central differences of channel 0)
        var differences = engine.TensorAbs(PreprocessorFilters.Interior3x3(PreprocessorFilters.Channel(data, 0), PreprocessorFilters.CentralDifferences));
        var dx = PreprocessorFilters.Channel(differences, 0);
        var dy = PreprocessorFilters.Channel(differences, 1);

        // Suppress non-line features (keep mostly horizontal or vertical edges):
        // min(1, 4 max(0, max(dx, dy) - 0.5 min(dx, dy)))
        var lineScore = engine.TensorSubtract(engine.TensorMax(dx, dy), engine.TensorMultiplyScalar(engine.TensorMin(dx, dy), NumOps.FromDouble(0.5)));
        lineScore = engine.TensorMax(lineScore, NumOps.Zero);
        return PreprocessorFilters.PadBorder(engine.TensorMin(engine.TensorMultiplyScalar(lineScore, NumOps.FromDouble(4.0)), NumOps.One));
    }
}
