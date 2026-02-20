using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Base interface for all safety modules in the composable safety pipeline.
/// </summary>
/// <remarks>
/// <para>
/// Safety modules are composable units that each check for a specific type of safety risk.
/// Multiple modules are combined into a <see cref="AiDotNet.Safety.SafetyPipeline{T}"/> that
/// runs them in sequence and aggregates their findings into a single <see cref="SafetyReport"/>.
/// </para>
/// <para>
/// <b>For Beginners:</b> Each safety module is like a specialist inspector. One checks for
/// toxic language, another for NSFW images, another for PII, etc. You assemble the inspectors
/// you need into a pipeline, and the pipeline runs them all and gives you a combined report.
/// </para>
/// <para>
/// <b>Architecture:</b> This replaces the monolithic ISafetyFilter with composable modules:
/// <code>
/// ISafetyModule{T}
///   |-- ITextSafetyModule{T}   (toxicity, PII, jailbreak, hallucination, copyright)
///   |-- IImageSafetyModule{T}  (NSFW, violence, deepfake)
///   |-- IAudioSafetyModule{T}  (deepfake, toxic speech)
///   |-- IVideoSafetyModule{T}  (content moderation, temporal deepfake)
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ISafetyModule<T>
{
    /// <summary>
    /// Gets the unique name of this safety module.
    /// </summary>
    /// <remarks>
    /// Used in <see cref="SafetyFinding.SourceModule"/> and <see cref="SafetyReport.ModulesExecuted"/>
    /// to identify which module produced each finding.
    /// </remarks>
    string ModuleName { get; }

    /// <summary>
    /// Evaluates the given content for safety and returns any findings.
    /// </summary>
    /// <param name="content">The content to evaluate, represented as a numeric vector.</param>
    /// <returns>
    /// A list of safety findings. An empty list means no safety issues were detected.
    /// </returns>
    IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content);

    /// <summary>
    /// Gets whether this module is ready to evaluate content.
    /// </summary>
    /// <remarks>
    /// A module may not be ready if it requires a model that hasn't been loaded,
    /// or if required configuration is missing.
    /// </remarks>
    bool IsReady { get; }
}
