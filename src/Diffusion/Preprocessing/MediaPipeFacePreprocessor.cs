using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// MediaPipe Face Mesh preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Produces face mesh-like feature maps by detecting facial structure through
/// gradient analysis. The output highlights facial feature regions (eyes, nose,
/// mouth contours) that guide face-conditioned generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This creates a wireframe-like mesh of facial features
/// (eyes, nose, mouth, jawline). ControlNet uses this to generate faces that
/// match the expression and structure of the original face.
/// </para>
/// <para>
/// Reference: Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines", 2019
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MediaPipeFacePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly double _sensitivity;

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.MediaPipeFace;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <summary>
    /// Initializes a new MediaPipe face mesh preprocessor.
    /// </summary>
    /// <param name="sensitivity">Sensitivity for facial feature detection. Default: 2.0.</param>
    public MediaPipeFacePreprocessor(double sensitivity = 2.0)
    {
        _sensitivity = sensitivity;
    }

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        int channels = Math.Min(shape[1], 3);
        if (!PreprocessorFilters.HasInterior(shape[2], shape[3]))
            return new Tensor<T>(new[] { shape[0], 3, shape[2], shape[3] });

        // Multi-channel gradient for facial feature detection: mean over channels of the central-difference magnitude
        var magnitudes = PreprocessorFilters.ChannelGradientMagnitudes(data, channels);
        var totalEdge = engine.TensorDivideScalar(engine.ReduceSum(magnitudes, new[] { 1 }, true), NumOps.FromDouble(channels));
        var meshIntensity = engine.TensorMin(engine.TensorMultiplyScalar(totalEdge, NumOps.FromDouble(_sensitivity)), NumOps.One);

        // Color-coded output: green for mesh lines, darker for background
        return PreprocessorFilters.PadBorder(PreprocessorFilters.ColorChannels(meshIntensity, 0.3, 1.0, 0.3));
    }
}
