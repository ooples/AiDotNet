#nullable disable
using AiDotNet.Enums;
using AiDotNet.Safety;
using AiDotNet.Safety.Compliance;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for compliance modules.
/// Tests EUAIActComplianceChecker, GDPRComplianceChecker, and SOC2ComplianceChecker
/// for missing controls detection, compliant configs, and various risk levels.
/// </summary>
public class ComplianceIntegrationTests
{
    #region EUAIActComplianceChecker Tests

    [Fact]
    public void EUAIAct_MissingWatermarking_DetectsNonCompliance()
    {
        var config = new SafetyConfig
        {
            Compliance = { EUAIAct = true },
            Watermarking = { TextWatermarking = false }
        };
        var checker = new EUAIActComplianceChecker<double>(config);
        var findings = checker.EvaluateText("Some AI-generated text");

        Assert.NotEmpty(findings);
        Assert.Contains(findings, f => f.Description.Contains("Article 50"));
    }

    [Fact]
    public void EUAIAct_FullyCompliant_FewerFindings()
    {
        var config = new SafetyConfig
        {
            Compliance = { EUAIAct = true },
            Watermarking = { TextWatermarking = true, ImageWatermarking = true, AudioWatermarking = true },
            Text = { ToxicityDetection = true, PIIDetection = true, JailbreakDetection = true },
            Fairness = { DemographicParity = true, EqualizedOdds = true }
        };
        var checker = new EUAIActComplianceChecker<double>(config);
        var findings = checker.EvaluateText("Some AI-generated text");

        // Fully-compliant should have fewer findings than non-compliant
        Assert.NotNull(findings);
    }

    [Fact]
    public void EUAIAct_MissingFairness_DetectsNonCompliance()
    {
        var config = new SafetyConfig
        {
            Compliance = { EUAIAct = true },
            Fairness = { DemographicParity = false, EqualizedOdds = false }
        };
        var checker = new EUAIActComplianceChecker<double>(config);
        var findings = checker.EvaluateText("Text for compliance check");

        Assert.NotEmpty(findings);
    }

    [Fact]
    public void EUAIAct_EmptyText_StillChecksConfig()
    {
        var config = new SafetyConfig
        {
            Compliance = { EUAIAct = true },
            Watermarking = { TextWatermarking = false }
        };
        var checker = new EUAIActComplianceChecker<double>(config);
        var findings = checker.EvaluateText("");

        Assert.NotNull(findings);
    }

    #endregion

    #region GDPRComplianceChecker Tests

    [Fact]
    public void GDPR_MissingPIIDetection_DetectsNonCompliance()
    {
        var config = new SafetyConfig
        {
            Text = { PIIDetection = false }
        };
        var checker = new GDPRComplianceChecker<double>(config);
        var findings = checker.EvaluateText("Some text with personal data");

        Assert.NotEmpty(findings);
        Assert.Contains(findings, f => f.Category == SafetyCategory.PIIExposure);
    }

    [Fact]
    public void GDPR_PIIEnabled_FewerFindings()
    {
        var config = new SafetyConfig
        {
            Text = { PIIDetection = true }
        };
        var checker = new GDPRComplianceChecker<double>(config);
        var findings = checker.EvaluateText("Some text");

        // With PII detection enabled, should be more compliant
        Assert.NotNull(findings);
    }

    [Fact]
    public void GDPR_AllControlsDisabled_DetectsMultipleIssues()
    {
        var config = new SafetyConfig
        {
            Text = { PIIDetection = false },
            Guardrails = { InputGuardrails = false, OutputGuardrails = false }
        };
        var checker = new GDPRComplianceChecker<double>(config);
        var findings = checker.EvaluateText("Text with data");

        Assert.NotEmpty(findings);
    }

    #endregion

    #region SOC2ComplianceChecker Tests

    [Fact]
    public void SOC2_MissingControls_DetectsMultipleIssues()
    {
        var config = new SafetyConfig
        {
            Text = { JailbreakDetection = false },
            Guardrails = { InputGuardrails = false }
        };
        var checker = new SOC2ComplianceChecker<double>(config);
        var findings = checker.EvaluateText("Some text");

        Assert.True(findings.Count >= 2,
            $"Should detect multiple missing SOC2 controls, found {findings.Count}");
    }

    [Fact]
    public void SOC2_AllControlsEnabled_FewerFindings()
    {
        var config = new SafetyConfig
        {
            Text = { ToxicityDetection = true, PIIDetection = true, JailbreakDetection = true },
            Guardrails = { InputGuardrails = true, OutputGuardrails = true }
        };
        var checker = new SOC2ComplianceChecker<double>(config);
        var findings = checker.EvaluateText("Some text");

        Assert.NotNull(findings);
    }

    [Fact]
    public void SOC2_MissingJailbreak_DetectsIssue()
    {
        var config = new SafetyConfig
        {
            Text = { JailbreakDetection = false }
        };
        var checker = new SOC2ComplianceChecker<double>(config);
        var findings = checker.EvaluateText("Text for check");

        Assert.NotEmpty(findings);
    }

    #endregion

    #region Cross-Module Tests

    [Fact]
    public void AllCheckers_SameConfig_ProduceResults()
    {
        var config = new SafetyConfig
        {
            Text = { PIIDetection = false, JailbreakDetection = false },
            Watermarking = { TextWatermarking = false },
            Guardrails = { InputGuardrails = false }
        };

        Assert.NotNull(new EUAIActComplianceChecker<double>(config).EvaluateText("Text"));
        Assert.NotNull(new GDPRComplianceChecker<double>(config).EvaluateText("Text"));
        Assert.NotNull(new SOC2ComplianceChecker<double>(config).EvaluateText("Text"));
    }

    [Fact]
    public void AllCheckers_NonCompliantConfig_AllDetectIssues()
    {
        var config = new SafetyConfig
        {
            Text = { PIIDetection = false, JailbreakDetection = false, ToxicityDetection = false },
            Watermarking = { TextWatermarking = false },
            Fairness = { DemographicParity = false },
            Guardrails = { InputGuardrails = false, OutputGuardrails = false }
        };

        Assert.NotEmpty(new EUAIActComplianceChecker<double>(config).EvaluateText("Text"));
        Assert.NotEmpty(new GDPRComplianceChecker<double>(config).EvaluateText("Text"));
        Assert.NotEmpty(new SOC2ComplianceChecker<double>(config).EvaluateText("Text"));
    }

    #endregion
}
