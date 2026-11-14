using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Normal map estimation preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Estimates surface normals from image gradients, producing a 3-channel normal map
/// where RGB channels represent the X, Y, Z components of surface normals.
/// </para>
/// <para>
/// <b>For Beginners:</b> A normal map shows which direction each surface in the image
/// is facing. The R/G/B channels encode the X/Y/Z direction. This helps ControlNet
/// generate images with correct lighting and surface detail.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class NormalMapPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Normal;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        if (!PreprocessorFilters.HasInterior(shape[2], shape[3]))
            return new Tensor<T>(new[] { shape[0], 3, shape[2], shape[3] });

        // Central differences of channel 0 over the interior; normal = normalize(-dx, -dy, 1) mapped to [0, 1]
        var differences = PreprocessorFilters.Interior3x3(PreprocessorFilters.Channel(data, 0), PreprocessorFilters.CentralDifferences);
        var dx = PreprocessorFilters.Channel(differences, 0);
        var dy = PreprocessorFilters.Channel(differences, 1);
        var squared = engine.TensorSquare(differences);
        var magnitude = engine.TensorSqrt(engine.TensorAddScalar(
            engine.TensorAdd(PreprocessorFilters.Channel(squared, 0), PreprocessorFilters.Channel(squared, 1)), NumOps.One));

        var half = NumOps.FromDouble(0.5);
        var red = engine.TensorMultiplyScalar(engine.ScalarMinusTensor(NumOps.One, engine.TensorDivide(dx, magnitude)), half);
        var green = engine.TensorMultiplyScalar(engine.ScalarMinusTensor(NumOps.One, engine.TensorDivide(dy, magnitude)), half);
        var blue = engine.TensorMultiplyScalar(engine.TensorAddScalar(engine.TensorReciprocal(magnitude), NumOps.One), half);
        return PreprocessorFilters.PadBorder(engine.TensorConcatenate(new[] { red, green, blue }, 1));
    }
}
