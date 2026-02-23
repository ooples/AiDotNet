using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Multimodal;

/// <summary>
/// Interface for multimodal safety modules that analyze cross-modal content interactions.
/// </summary>
/// <remarks>
/// <para>
/// Multimodal safety modules detect risks that arise from the interaction between different
/// content modalities — for example, safe text paired with an unsafe image, or cross-modal
/// attacks that exploit mismatches between text and image safety classifiers.
/// </para>
/// <para>
/// <b>For Beginners:</b> A multimodal safety module checks the combination of different
/// content types together. For example, someone might pair innocent-sounding text with a
/// harmful image to bypass safety checks — this module catches those cross-modal attacks.
/// </para>
/// <para>
/// <b>References:</b>
/// - Cross-modal safety mechanism transfer failure in VLMs (2024, arxiv:2410.12662)
/// - OmniSafeBench-MM: 13 attacks, 15 defenses, 9 risk domains (2025, arxiv:2512.06589)
/// - MM-SafetyBench: 5,040 text-image pairs across 13 scenarios (ECCV 2024)
/// - Llama Guard 3 Vision: Multimodal safety classification (Meta, 2024, arxiv:2411.10414)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IMultimodalSafetyModule<T> : ISafetyModule<T>
{
    /// <summary>
    /// Evaluates a text-image pair for cross-modal safety risks.
    /// </summary>
    /// <param name="text">The text content.</param>
    /// <param name="image">The image tensor.</param>
    /// <returns>A list of cross-modal safety findings.</returns>
    IReadOnlyList<SafetyFinding> EvaluateTextImage(string text, Tensor<T> image);

    /// <summary>
    /// Evaluates a text-audio pair for cross-modal safety risks.
    /// </summary>
    /// <param name="text">The text content.</param>
    /// <param name="audio">The audio samples.</param>
    /// <param name="sampleRate">The audio sample rate.</param>
    /// <returns>A list of cross-modal safety findings.</returns>
    IReadOnlyList<SafetyFinding> EvaluateTextAudio(string text, Vector<T> audio, int sampleRate);
}
