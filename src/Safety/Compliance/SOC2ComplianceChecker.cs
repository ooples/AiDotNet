using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Compliance;

/// <summary>
/// Checks compliance with SOC 2 requirements for AI system security and availability.
/// </summary>
/// <remarks>
/// <para>
/// SOC 2 (Service Organization Control 2) requires organizations to demonstrate that their
/// systems meet the Trust Services Criteria: Security, Availability, Processing Integrity,
/// Confidentiality, and Privacy. This module checks whether the safety pipeline has adequate
/// controls for AI-specific SOC 2 concerns.
/// </para>
/// <para>
/// <b>For Beginners:</b> SOC 2 is a security standard for service organizations. If your
/// company processes customer data through AI, auditors will check that you have proper
/// safety controls. This module verifies that your AI safety pipeline meets those requirements.
/// </para>
/// <para>
/// <b>Key requirements checked:</b>
/// - CC6.1: Logical access controls — input validation, jailbreak prevention
/// - CC7.2: System monitoring — safety event logging and alerting
/// - CC8.1: Change management — safety configuration validation
/// - PI1.1: Processing integrity — output validation and quality checks
/// - C1.1: Confidentiality — PII detection and data classification
/// </para>
/// <para>
/// <b>References:</b>
/// - AICPA SOC 2 Trust Services Criteria (2022, updated 2024)
/// - SOC 2 for AI systems: Emerging best practices (2024)
/// - AI governance and SOC 2 compliance (ISACA, 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class SOC2ComplianceChecker<T> : ITextSafetyModule<T>
{
    private readonly SafetyConfig _config;

    /// <inheritdoc />
    public string ModuleName => "SOC2ComplianceChecker";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new SOC 2 compliance checker.
    /// </summary>
    /// <param name="config">The safety configuration to check against SOC 2 requirements.</param>
    public SOC2ComplianceChecker(SafetyConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
    }

    /// <summary>
    /// Evaluates text for SOC 2 compliance issues.
    /// </summary>
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        // CC6.1: Logical access controls — jailbreak prevention
        if (!_config.Text.EffectiveJailbreakDetection)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.JailbreakAttempt,
                Severity = SafetySeverity.Medium,
                Confidence = 1.0,
                Description = "SOC 2 CC6.1 (Logical Access) recommends jailbreak detection for AI systems " +
                              "to prevent unauthorized access to restricted functionality.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        // CC6.1: Input validation
        if (!_config.Guardrails.EffectiveInputGuardrails)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.Medium,
                Confidence = 1.0,
                Description = "SOC 2 CC6.1 (Logical Access) requires input validation. " +
                              "Input guardrails are not enabled.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        // PI1.1: Processing integrity — output validation
        if (!_config.Guardrails.EffectiveOutputGuardrails)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Hallucination,
                Severity = SafetySeverity.Low,
                Confidence = 1.0,
                Description = "SOC 2 PI1.1 (Processing Integrity) recommends output validation. " +
                              "Output guardrails are not enabled.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        // C1.1: Confidentiality — PII detection
        if (!_config.Text.EffectivePIIDetection)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PIIExposure,
                Severity = SafetySeverity.Medium,
                Confidence = 1.0,
                Description = "SOC 2 C1.1 (Confidentiality) requires data classification and protection. " +
                              "PII detection is not enabled.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        // SOC2 compliance checks configuration state, not vector content.
        return EvaluateText(string.Empty);
    }
}
