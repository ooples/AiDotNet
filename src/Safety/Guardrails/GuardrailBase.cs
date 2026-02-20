using AiDotNet.Safety.Text;
using AiDotNet.Safety.Guardrails;

namespace AiDotNet.Safety.Guardrails;

/// <summary>
/// Abstract base class for guardrail modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for guardrails including direction configuration
/// and common content validation utilities. Concrete implementations provide
/// the actual guardrail logic (input validation, output filtering, topic restriction, custom rules).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all guardrails.
/// Each guardrail type extends this and adds its own rules for validating
/// content entering or leaving the AI system.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class GuardrailBase<T> : TextSafetyModuleBase<T>, IGuardrail<T>
{
    /// <inheritdoc />
    public abstract GuardrailDirection Direction { get; }
}
