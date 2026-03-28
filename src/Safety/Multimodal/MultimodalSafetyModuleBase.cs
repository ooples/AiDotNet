using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety.Multimodal;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Multimodal;

/// <summary>
/// Abstract base class for multimodal safety modules that analyze cross-modal content.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for multimodal safety modules including cross-modal
/// feature extraction and alignment utilities. Concrete implementations provide
/// the actual cross-modal analysis (text-image alignment, cross-modal consistency, guardrail).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for modules that check
/// the combination of different content types (text + image, text + audio) for safety risks.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class MultimodalSafetyModuleBase<T> : IMultimodalSafetyModule<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public abstract string ModuleName { get; }

    /// <inheritdoc />
    public virtual bool IsReady => true;

    /// <inheritdoc />
    public abstract IReadOnlyList<SafetyFinding> EvaluateTextImage(string text, Tensor<T> image);

    /// <inheritdoc />
    public virtual IReadOnlyList<SafetyFinding> EvaluateTextAudio(string text, Vector<T> audio, int sampleRate)
    {
        // Default: no cross-modal audio checking. Override in subclasses that support it.
        return Array.Empty<SafetyFinding>();
    }

    /// <inheritdoc />
    public virtual IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        return Array.Empty<SafetyFinding>();
    }
}
