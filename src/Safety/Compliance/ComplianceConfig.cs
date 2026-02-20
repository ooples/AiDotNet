namespace AiDotNet.Safety.Compliance;

/// <summary>
/// Configuration for regulatory compliance checking modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure which regulatory frameworks to check
/// compliance for. Enable the ones that apply to your deployment region and use case.
/// </para>
/// </remarks>
public class ComplianceConfig
{
    /// <summary>Whether to check EU AI Act compliance. Default: false.</summary>
    public bool? EUAIAct { get; set; }

    /// <summary>Whether to check GDPR compliance. Default: false.</summary>
    public bool? GDPR { get; set; }

    /// <summary>Whether to check SOC2 compliance. Default: false.</summary>
    public bool? SOC2 { get; set; }

    /// <summary>The EU AI Act risk level classification. Default: Limited.</summary>
    public RiskLevel? AIActRiskLevel { get; set; }

    internal bool EffectiveEUAIAct => EUAIAct ?? false;
    internal bool EffectiveGDPR => GDPR ?? false;
    internal bool EffectiveSOC2 => SOC2 ?? false;
    internal RiskLevel EffectiveAIActRiskLevel => AIActRiskLevel ?? RiskLevel.Limited;
}
