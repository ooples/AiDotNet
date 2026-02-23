using AiDotNet.Safety;
using AiDotNet.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for safety modules that operate on image content.
/// </summary>
/// <remarks>
/// <para>
/// Image safety modules analyze image tensors for safety risks such as NSFW content,
/// graphic violence, deepfakes, and AI-generated content.
/// </para>
/// <para>
/// <b>For Beginners:</b> Image safety modules check pictures and generated images for
/// harmful content. They can detect things like nudity, violence, manipulated photos
/// (deepfakes), and whether an image was created by AI.
/// </para>
/// <para>
/// <b>References:</b>
/// - UnsafeBench: 11 categories of unsafe images (Qu et al., 2024)
/// - USD: Scene-graph NSFW detection (USENIX Security 2025)
/// - Safe-CLIP: Removing NSFW concepts from CLIP (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IImageSafetyModule<T> : ISafetyModule<T>
{
    /// <summary>
    /// Evaluates the given image tensor for safety and returns any findings.
    /// </summary>
    /// <param name="image">
    /// The image tensor to evaluate. Expected shape: [C, H, W] or [B, C, H, W].
    /// </param>
    /// <returns>
    /// A list of safety findings. An empty list means no safety issues were detected.
    /// </returns>
    IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image);
}
