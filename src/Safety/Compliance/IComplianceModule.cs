using AiDotNet.Interfaces;

namespace AiDotNet.Safety.Compliance;

/// <summary>
/// Interface for regulatory compliance checking modules.
/// </summary>
/// <remarks>
/// <para>
/// Compliance modules evaluate AI system configurations and outputs against regulatory
/// requirements such as the EU AI Act, GDPR, and SOC2. They check for required
/// transparency, data protection, watermarking, and audit trail compliance.
/// </para>
/// <para>
/// <b>For Beginners:</b> A compliance module checks if your AI system meets legal
/// requirements. Different laws around the world require AI systems to be transparent,
/// protect personal data, and maintain audit trails. This module checks all of that.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IComplianceModule<T> : ITextSafetyModule<T>
{
    /// <summary>
    /// Gets the name of the regulation this module checks compliance for.
    /// </summary>
    string RegulationName { get; }

    /// <summary>
    /// Evaluates the current safety configuration for compliance.
    /// </summary>
    /// <param name="config">The safety configuration to evaluate.</param>
    /// <returns>A list of compliance findings (violations, warnings, and recommendations).</returns>
    IReadOnlyList<SafetyFinding> EvaluateCompliance(SafetyConfig config);
}
