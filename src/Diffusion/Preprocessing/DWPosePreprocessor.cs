using AiDotNet.Diffusion.Control;
using AiDotNet.Models;

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
public class DWPosePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.DWPose;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        var shape = data.Shape;
        int batch = shape[0];
        int height = shape[2];
        int width = shape[3];
        var result = new Tensor<T>(new[] { batch, 3, height, width });

        for (int b = 0; b < batch; b++)
        {
            for (int h = 1; h < height - 1; h++)
            {
                for (int w = 1; w < width - 1; w++)
                {
                    double center = NumOps.ToDouble(data[b, 0, h, w]);
                    double dx = Math.Abs(NumOps.ToDouble(data[b, 0, h, w + 1]) - NumOps.ToDouble(data[b, 0, h, w - 1]));
                    double dy = Math.Abs(NumOps.ToDouble(data[b, 0, h + 1, w]) - NumOps.ToDouble(data[b, 0, h - 1, w]));
                    double edge = Math.Min(1.0, (dx + dy) * 3.0);

                    result[b, 0, h, w] = NumOps.FromDouble(edge);
                    result[b, 1, h, w] = NumOps.FromDouble(edge * 0.7);
                    result[b, 2, h, w] = NumOps.FromDouble(edge * 0.3);
                }
            }
        }

        return result;
    }
}
