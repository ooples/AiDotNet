using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Semantic segmentation preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Produces semantic segmentation maps where each pixel is assigned a class label
/// encoded as a color. The output guides ControlNet to respect object boundaries and regions.
/// </para>
/// <para>
/// <b>For Beginners:</b> This labels every pixel in the image (sky, person, car, etc.)
/// and paints them different colors. ControlNet uses this to generate images where
/// objects are in the same regions.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SemanticSegPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Segmentation;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        var r = PreprocessorFilters.Channel(data, 0);
        var g = shape[1] > 1 ? PreprocessorFilters.Channel(data, 1) : r;
        var bv = shape[1] > 2 ? PreprocessorFilters.Channel(data, 2) : r;
        var four = NumOps.FromDouble(4.0);

        // Simple quantization-based segmentation (placeholder):
        // classId = (trunc(4 r) * 25 + trunc(4 g) * 5 + trunc(4 b)) mod 150 (ADE20K has 150 classes)
        var classId = engine.TensorAdd(engine.TensorAdd(
                engine.TensorMultiplyScalar(engine.TensorTrunc(engine.TensorMultiplyScalar(r, four)), NumOps.FromDouble(25.0)),
                engine.TensorMultiplyScalar(engine.TensorTrunc(engine.TensorMultiplyScalar(g, four)), NumOps.FromDouble(5.0))),
            engine.TensorTrunc(engine.TensorMultiplyScalar(bv, four)));
        classId = PreprocessorFilters.Mod(classId, 150);

        // Map class to deterministic color
        return PreprocessorFilters.LabelColors(classId, 37, 67, 97);
    }
}
