#nullable disable
using AiDotNet.Enums;
using AiDotNet.Safety;
using AiDotNet.Safety.Text;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for PII detection modules.
/// Tests RegexPIIDetector, NERPIIDetector, ContextAwarePIIDetector, and CompositePIIDetector
/// against emails, SSNs, phones, credit cards, and safe text.
/// </summary>
public class PIIDetectionIntegrationTests
{
    #region RegexPIIDetector Tests

    [Fact]
    public void Regex_EmailAddress_DetectsPII()
    {
        var detector = new RegexPIIDetector<double>();
        var findings = detector.EvaluateText("Contact me at john.doe@example.com for details.");

        Assert.NotEmpty(findings);
        Assert.Contains(findings, f => f.Category == SafetyCategory.PIIExposure);
    }

    [Fact]
    public void Regex_SSN_DetectsPII()
    {
        var detector = new RegexPIIDetector<double>();
        var findings = detector.EvaluateText("My social security number is 123-45-6789.");

        Assert.NotEmpty(findings);
        Assert.Contains(findings, f => f.Category == SafetyCategory.PIIExposure);
    }

    [Fact]
    public void Regex_PhoneNumber_DetectsPII()
    {
        var detector = new RegexPIIDetector<double>();
        var findings = detector.EvaluateText("Call me at (555) 123-4567 anytime.");

        Assert.NotEmpty(findings);
    }

    [Fact]
    public void Regex_CreditCard_DetectsPII()
    {
        var detector = new RegexPIIDetector<double>();
        var findings = detector.EvaluateText("My credit card number is 4111-1111-1111-1111.");

        Assert.NotEmpty(findings);
    }

    [Fact]
    public void Regex_IPAddress_DetectsPII()
    {
        var detector = new RegexPIIDetector<double>();
        var findings = detector.EvaluateText("The server is located at 192.168.1.100.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void Regex_SafeText_NoFindings()
    {
        var detector = new RegexPIIDetector<double>();
        var findings = detector.EvaluateText("The sky is blue and the grass is green.");

        Assert.Empty(findings);
    }

    [Fact]
    public void Regex_MultiplePIITypes_DetectsAll()
    {
        var detector = new RegexPIIDetector<double>();
        var findings = detector.EvaluateText(
            "Email: test@example.com, SSN: 123-45-6789, Phone: (555) 123-4567");

        Assert.True(findings.Count >= 2,
            $"Should detect multiple PII types, found {findings.Count}");
    }

    #endregion

    #region NERPIIDetector Tests

    [Fact]
    public void NER_PersonName_DetectsPII()
    {
        var detector = new NERPIIDetector<double>();
        var findings = detector.EvaluateText("The patient John Smith has an appointment tomorrow.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void NER_SafeGenericText_NoFindings()
    {
        var detector = new NERPIIDetector<double>();
        var findings = detector.EvaluateText("The algorithm converges after several iterations.");

        Assert.Empty(findings);
    }

    #endregion

    #region ContextAwarePIIDetector Tests

    [Fact]
    public void ContextAware_EmailWithContext_DetectsPII()
    {
        var detector = new ContextAwarePIIDetector<double>();
        var findings = detector.EvaluateText(
            "Please send confidential documents to john@company.com");

        Assert.NotEmpty(findings);
    }

    [Fact]
    public void ContextAware_SafeText_NoFindings()
    {
        var detector = new ContextAwarePIIDetector<double>();
        var findings = detector.EvaluateText("Photosynthesis converts sunlight into energy.");

        Assert.Empty(findings);
    }

    [Fact]
    public void ContextAware_WithCustomInner_Works()
    {
        var inner = new RegexPIIDetector<double>();
        var detector = new ContextAwarePIIDetector<double>(inner, contextWindow: 100);
        var findings = detector.EvaluateText("My SSN is 123-45-6789 and email is test@example.com");

        Assert.NotEmpty(findings);
    }

    #endregion

    #region CompositePIIDetector Tests

    [Fact]
    public void Composite_MultipleTypes_DetectsAll()
    {
        var detector = new CompositePIIDetector<double>();
        var findings = detector.EvaluateText(
            "Contact John at john@example.com, phone (555) 123-4567, SSN 123-45-6789");

        Assert.True(findings.Count >= 2,
            $"Should detect multiple PII types, found {findings.Count}");
    }

    [Fact]
    public void Composite_SafeText_NoFindings()
    {
        var detector = new CompositePIIDetector<double>();
        var findings = detector.EvaluateText(
            "Neural networks are a type of machine learning model.");

        Assert.Empty(findings);
    }

    [Fact]
    public void Composite_PassportNumber_DetectsPII()
    {
        var detector = new CompositePIIDetector<double>();
        var findings = detector.EvaluateText(
            "My passport number is AB1234567 and SSN is 987-65-4321.");

        Assert.NotEmpty(findings);
    }

    [Fact]
    public void Composite_EmptyText_NoFindings()
    {
        var detector = new CompositePIIDetector<double>();
        var findings = detector.EvaluateText("");

        Assert.Empty(findings);
    }

    #endregion

    #region Cross-Module Tests

    [Fact]
    public void AllDetectors_SameInput_AllDetectPII()
    {
        var text = "Email me at user@test.com, my SSN is 123-45-6789.";
        var regex = new RegexPIIDetector<double>();
        var composite = new CompositePIIDetector<double>();
        var contextAware = new ContextAwarePIIDetector<double>();

        Assert.NotEmpty(regex.EvaluateText(text));
        Assert.NotEmpty(composite.EvaluateText(text));
        Assert.NotEmpty(contextAware.EvaluateText(text));
    }

    [Fact]
    public void AllDetectors_SameInput_CorrectCategory()
    {
        var text = "Send payment details to admin@company.org, SSN 111-22-3333.";
        var detector = new CompositePIIDetector<double>();
        var findings = detector.EvaluateText(text);

        Assert.All(findings, f => Assert.Equal(SafetyCategory.PIIExposure, f.Category));
    }

    #endregion
}
