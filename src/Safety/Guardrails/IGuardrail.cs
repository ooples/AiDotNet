using AiDotNet.Interfaces;

namespace AiDotNet.Safety.Guardrails;

/// <summary>
/// Interface for guardrail modules that validate and filter input/output content.
/// </summary>
/// <remarks>
/// <para>
/// Guardrails are pre/post-processing safety checks that validate content before and after
/// model processing. They enforce length limits, topic restrictions, content policies,
/// and custom rules. Guardrails can block, warn, log, or modify content.
/// </para>
/// <para>
/// <b>For Beginners:</b> A guardrail is like a safety fence around your AI. Input guardrails
/// check what goes into the AI (blocking dangerous prompts), and output guardrails check
/// what comes out (filtering harmful responses). You can also restrict specific topics.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IGuardrail<T> : ITextSafetyModule<T>
{
    /// <summary>
    /// Gets whether this guardrail checks input content, output content, or both.
    /// </summary>
    GuardrailDirection Direction { get; }
}

/// <summary>
/// Specifies whether a guardrail checks input, output, or both directions.
/// </summary>
public enum GuardrailDirection
{
    /// <summary>Checks input content before processing.</summary>
    Input,

    /// <summary>Checks output content after processing.</summary>
    Output,

    /// <summary>Checks both input and output content.</summary>
    Both
}
