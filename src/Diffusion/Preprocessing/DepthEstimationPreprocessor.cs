using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Depth estimation preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Estimates monocular depth from a single image using gradient-based approximation.
/// The output is a single-channel depth map where brighter values indicate closer objects.
/// </para>
/// <para>
/// <b>For Beginners:</b> This estimates how far away each part of the image is.
/// Bright areas are close, dark areas are far away. ControlNet uses this to
/// generate images with correct 3D perspective and depth.
///
/// In production, models like MiDaS or Depth Anything would be used for accurate
/// depth estimation. This implementation provides a gradient-based approximation.
/// </para>
/// <para>
/// Reference: Ranftl et al., "Towards Robust Monocular Depth Estimation", IEEE TPAMI 2022
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DepthEstimationPreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Depth;
    /// <inheritdoc />
    public override int OutputChannels => 1;

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        int height = shape[2];
        int width = shape[3];

        // Approximate depth from local contrast (low frequency = far): the mean over the replicate-padded 3x3
        // neighbourhood of |gray - neighbour|, where neighbours use channel 0 at unit luma weight
        var gray = PreprocessorFilters.Grayscale(data);
        var red = PreprocessorFilters.Channel(data, 0);
        var neighbourGray = engine.TensorAdd(engine.TensorAdd(
                engine.TensorMultiplyScalar(red, NumOps.FromDouble(0.299)), engine.TensorMultiplyScalar(red, NumOps.FromDouble(0.587))),
            engine.TensorMultiplyScalar(red, NumOps.FromDouble(0.114)));
        var padded = engine.PadNd(neighbourGray, new[] { 1, 1, 1, 1 }, PadMode.Replicate, NumOps.Zero);

        var localContrast = new Tensor<T>(gray._shape);
        for (int dh = 0; dh <= 2; dh++)
        {
            var rows = engine.TensorNarrow(padded, 2, dh, height);
            for (int dw = 0; dw <= 2; dw++)
            {
                var neighbour = engine.TensorNarrow(rows, 3, dw, width);
                localContrast = engine.TensorAdd(localContrast, engine.TensorAbs(engine.TensorSubtract(gray, neighbour)));
            }
        }

        // Higher local contrast suggests closer objects
        localContrast = engine.TensorDivideScalar(localContrast, NumOps.FromDouble(9.0));
        return engine.TensorMin(engine.TensorMultiplyScalar(localContrast, NumOps.FromDouble(5.0)), NumOps.One);
    }
}
