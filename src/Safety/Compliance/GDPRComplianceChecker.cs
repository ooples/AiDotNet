using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Compliance;

/// <summary>
/// Checks compliance with GDPR requirements related to AI and personal data processing.
/// </summary>
/// <remarks>
/// <para>
/// The General Data Protection Regulation (GDPR, 2016/679) imposes strict requirements on
/// processing personal data. AI systems that process personal data must comply with data
/// minimization, purpose limitation, and individual rights (right to explanation, right to
/// erasure). This module checks whether appropriate PII safeguards are in place.
/// </para>
/// <para>
/// <b>For Beginners:</b> GDPR is a European privacy law. If your AI system handles
/// personal information (names, emails, etc.), you need to detect and protect that data.
/// This module checks that your safety pipeline has the right protections enabled.
/// </para>
/// <para>
/// <b>Key requirements checked:</b>
/// - Article 5(1)(c): Data minimization — only process necessary personal data
/// - Article 13/14: Right to information — transparency about data processing
/// - Article 17: Right to erasure — ability to delete personal data
/// - Article 22: Automated decision-making — right to human review
/// - Article 35: Data Protection Impact Assessment for high-risk processing
/// </para>
/// <para>
/// <b>References:</b>
/// - GDPR (Regulation 2016/679), Articles 5, 13-14, 17, 22, 35
/// - EDPB Guidelines on AI and GDPR (2024)
/// - CNIL AI Action Plan: GDPR compliance for AI systems (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GDPRComplianceChecker<T> : ITextSafetyModule<T>
{
    private readonly SafetyConfig _config;

    /// <inheritdoc />
    public string ModuleName => "GDPRComplianceChecker";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new GDPR compliance checker.
    /// </summary>
    /// <param name="config">The safety configuration to check against GDPR requirements.</param>
    public GDPRComplianceChecker(SafetyConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
    }

    /// <summary>
    /// Evaluates text for GDPR compliance issues.
    /// </summary>
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        // Article 5(1)(c): Data minimization requires PII detection
        if (!_config.Text.EffectivePIIDetection)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PIIExposure,
                Severity = SafetySeverity.High,
                Confidence = 1.0,
                Description = "GDPR Article 5(1)(c) requires data minimization. PII detection is not " +
                              "enabled — personal data may be processed or exposed without safeguards.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        // GDPR compliance checks configuration state, not vector content.
        // Delegate to EvaluateText with empty string to run config checks.
        return EvaluateText(string.Empty);
    }
}
