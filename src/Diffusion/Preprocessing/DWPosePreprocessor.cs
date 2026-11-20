using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// DWPose whole-body keypoint detection preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DWPose (Dual Whole-body Pose) detects body, hand, and face keypoints with improved
/// accuracy over OpenPose. It produces a more detailed skeleton including finger and
/// facial landmark detection.
/// </para>
/// <para>
/// <b>For Beginners:</b> DWPose is an improved version of OpenPose that also detects
/// hand gestures and facial expressions, not just body pose. This gives ControlNet
/// much finer control over generated human figures.
/// </para>
/// <para>
/// Reference: Yang et al., "Effective Whole-body Pose Estimation with Two-stages Distillation", ICCV 2023
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DWPosePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.DWPose;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var engine = AiDotNetEngine.Current;
        var shape = data._shape;
        if (!PreprocessorFilters.HasInterior(shape[2], shape[3]))
            return new Tensor<T>(new[] { shape[0], 3, shape[2], shape[3] });

        // Edge strength min(1, 3 (|dx| + |dy|)) of channel 0 from central differences, colour-coded (1, 0.7, 0.3)
        var differences = engine.TensorAbs(PreprocessorFilters.Interior3x3(PreprocessorFilters.Channel(data, 0), PreprocessorFilters.CentralDifferences));
        var sum = engine.TensorAdd(PreprocessorFilters.Channel(differences, 0), PreprocessorFilters.Channel(differences, 1));
        var edge = engine.TensorMin(engine.TensorMultiplyScalar(sum, NumOps.FromDouble(3.0)), NumOps.One);
        return PreprocessorFilters.PadBorder(PreprocessorFilters.ColorChannels(edge, 1.0, 0.7, 0.3));
    }
}
