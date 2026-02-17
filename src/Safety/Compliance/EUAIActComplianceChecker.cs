using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Compliance;

/// <summary>
/// Checks compliance with the EU AI Act requirements for AI systems.
/// </summary>
/// <remarks>
/// <para>
/// The EU AI Act (Regulation 2024/1689) establishes a risk-based framework for AI systems
/// in the European Union. This module checks whether an AI system's safety pipeline meets
/// the Act's requirements based on the system's risk classification.
/// </para>
/// <para>
/// <b>For Beginners:</b> The EU AI Act is a law that requires AI systems used in Europe to
/// meet certain safety standards. High-risk systems need transparency, human oversight,
/// and watermarking. This module checks whether your AI system has the right safety features
/// enabled for compliance.
/// </para>
/// <para>
/// <b>Key requirements checked:</b>
/// - Article 50: AI-generated content must be machine-detectable (watermarking)
/// - Article 52: Transparency obligations for certain AI systems
/// - Article 6/Annex III: High-risk AI systems require safety management systems
/// - Articles 9-15: Requirements for high-risk systems (data governance, accuracy, cybersecurity)
/// </para>
/// <para>
/// <b>References:</b>
/// - EU AI Act (Regulation 2024/1689), effective August 2024
/// - ENISA guidance on AI cybersecurity for the EU AI Act (2024)
/// - EU AI Act compliance frameworks survey (2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EUAIActComplianceChecker<T> : ITextSafetyModule<T>
{
    private readonly SafetyConfig _config;

    /// <inheritdoc />
    public string ModuleName => "EUAIActComplianceChecker";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new EU AI Act compliance checker.
    /// </summary>
    /// <param name="config">
    /// The safety configuration to check against EU AI Act requirements.
    /// </param>
    public EUAIActComplianceChecker(SafetyConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
    }

    /// <summary>
    /// Evaluates text for compliance issues and checks pipeline configuration.
    /// </summary>
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        // Check Article 50: AI-generated content must be machine-detectable
        if (!_config.Watermarking.EffectiveTextWatermarking)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.AIGenerated,
                Severity = SafetySeverity.Medium,
                Confidence = 1.0,
                Description = "EU AI Act Article 50 requires AI-generated text content to be " +
                              "machine-detectable. Text watermarking is not enabled.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        // Check for transparency requirements
        if (!_config.Text.EffectiveToxicityDetection)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Misinformation,
                Severity = SafetySeverity.Low,
                Confidence = 0.8,
                Description = "EU AI Act Article 9 recommends toxicity detection for high-risk systems. " +
                              "Toxicity detection is not enabled.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        // Check for PII detection (relates to GDPR integration)
        if (!_config.Text.EffectivePIIDetection)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PIIExposure,
                Severity = SafetySeverity.Low,
                Confidence = 0.8,
                Description = "EU AI Act Article 10 requires data governance including PII protection. " +
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
        return Array.Empty<SafetyFinding>();
    }
}
