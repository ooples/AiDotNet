using AiDotNet.Safety;
using AiDotNet.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for safety modules that operate on video content.
/// </summary>
/// <remarks>
/// <para>
/// Video safety modules analyze video frames and their temporal relationships for safety
/// risks such as harmful visual content, deepfake videos, and policy-violating material.
/// </para>
/// <para>
/// <b>For Beginners:</b> Video safety modules check video content for harmful material.
/// They can analyze individual frames (like image safety) and also detect temporal
/// inconsistencies that reveal deepfake manipulation — things like unnatural blinking,
/// facial warping between frames, or audio-visual mismatches.
/// </para>
/// <para>
/// <b>References:</b>
/// - Spatio-temporal consistency exploitation for deepfake video detection (2025)
/// - NACO: Self-supervised natural consistency for face forgery detection (ECCV 2024)
/// - Generalizable deepfake detection across benchmarks (CVPR 2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IVideoSafetyModule<T> : ISafetyModule<T>
{
    /// <summary>
    /// Evaluates the given video frames for safety and returns any findings.
    /// </summary>
    /// <param name="frames">
    /// A list of video frames as image tensors. Each tensor shape: [C, H, W].
    /// Frames should be in temporal order.
    /// </param>
    /// <param name="frameRate">The video frame rate in FPS.</param>
    /// <returns>
    /// A list of safety findings. An empty list means no safety issues were detected.
    /// </returns>
    IReadOnlyList<SafetyFinding> EvaluateVideo(IReadOnlyList<Tensor<T>> frames, double frameRate);
}
