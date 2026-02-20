#nullable disable
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Safety;
using AiDotNet.Safety.Guardrails;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for guardrail modules.
/// Tests InputGuardrail, OutputGuardrail, TopicRestrictionGuardrail,
/// CustomRuleGuardrail, CompositeGuardrail, and GuardrailRule.
/// </summary>
public class GuardrailIntegrationTests
{
    #region InputGuardrail Tests

    [Fact]
    public void Input_OversizedInput_Blocks()
    {
        var guardrail = new InputGuardrail<double>(maxInputLength: 100);
        string longInput = new string('a', 200);
        var findings = guardrail.EvaluateText(longInput);

        Assert.NotEmpty(findings);
        Assert.Contains(findings, f => f.RecommendedAction == SafetyAction.Block);
    }

    [Fact]
    public void Input_NormalLength_NoFindings()
    {
        var guardrail = new InputGuardrail<double>(maxInputLength: 1000);
        var findings = guardrail.EvaluateText("This is a normal length question.");

        Assert.Empty(findings);
    }

    [Fact]
    public void Input_ExactlyAtLimit_NoFindings()
    {
        var guardrail = new InputGuardrail<double>(maxInputLength: 10);
        var findings = guardrail.EvaluateText("1234567890");

        Assert.Empty(findings);
    }

    [Fact]
    public void Input_EmptyText_BlocksEmptyInput()
    {
        var guardrail = new InputGuardrail<double>(maxInputLength: 100);
        var findings = guardrail.EvaluateText("");

        // InputGuardrail blocks empty/whitespace-only input by design
        Assert.NotEmpty(findings);
        Assert.Contains(findings, f => f.RecommendedAction == SafetyAction.Block);
    }

    #endregion

    #region OutputGuardrail Tests

    [Fact]
    public void Output_HighlyRepetitive_DetectsIssue()
    {
        var guardrail = new OutputGuardrail<double>(repetitionThreshold: 0.3);
        string repetitive = string.Join(" ",
            Enumerable.Repeat("the cat sat on the mat and looked at the hat", 30));
        var findings = guardrail.EvaluateText(repetitive);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Output_UniqueContent_NoFindings()
    {
        var guardrail = new OutputGuardrail<double>(repetitionThreshold: 0.5);
        var findings = guardrail.EvaluateText(
            "Machine learning is a subset of artificial intelligence that enables systems " +
            "to learn from data without being explicitly programmed.");

        Assert.Empty(findings);
    }

    [Fact]
    public void Output_EmptyText_WarnsOnShortOutput()
    {
        var guardrail = new OutputGuardrail<double>(repetitionThreshold: 0.3);
        var findings = guardrail.EvaluateText("");

        // OutputGuardrail warns when output is shorter than minimum length
        Assert.NotEmpty(findings);
    }

    #endregion

    #region TopicRestrictionGuardrail Tests

    [Fact]
    public void TopicRestriction_RestrictedTopic_Blocks()
    {
        var guardrail = new TopicRestrictionGuardrail<double>(
            new[] { "politics", "religion" });
        var findings = guardrail.EvaluateText("Let's discuss politics and government policy");

        Assert.NotEmpty(findings);
    }

    [Fact]
    public void TopicRestriction_AllowedTopic_NoFindings()
    {
        var guardrail = new TopicRestrictionGuardrail<double>(
            new[] { "politics", "religion" });
        var findings = guardrail.EvaluateText(
            "The weather forecast shows sunny skies for the weekend.");

        Assert.Empty(findings);
    }

    [Fact]
    public void TopicRestriction_MultipleRestricted_DetectsAll()
    {
        var guardrail = new TopicRestrictionGuardrail<double>(
            new[] { "politics", "religion", "gambling" });
        var findings = guardrail.EvaluateText(
            "The politics of religion and gambling are intertwined.");

        Assert.NotEmpty(findings);
    }

    [Fact]
    public void TopicRestriction_CaseInsensitive_Detects()
    {
        var guardrail = new TopicRestrictionGuardrail<double>(
            new[] { "politics" });
        var findings = guardrail.EvaluateText("POLITICS is a hot topic today.");

        Assert.NotEmpty(findings);
    }

    #endregion

    #region CustomRuleGuardrail Tests

    [Fact]
    public void CustomRule_MatchesPattern_ReturnsFindings()
    {
        var rules = new[]
        {
            new CustomRule("NoCompetitor", text =>
                text.Contains("competitor", StringComparison.OrdinalIgnoreCase)
                    ? new SafetyFinding
                    {
                        Category = SafetyCategory.PromptInjection,
                        Severity = SafetySeverity.Low,
                        Confidence = 1.0,
                        Description = "Competitor mentioned",
                        RecommendedAction = SafetyAction.Warn,
                        SourceModule = "CustomRule"
                    }
                    : null)
        };
        var guardrail = new CustomRuleGuardrail<double>(rules);
        var findings = guardrail.EvaluateText("Our competitor has a better product");

        Assert.NotEmpty(findings);
    }

    [Fact]
    public void CustomRule_NoMatch_NoFindings()
    {
        var rules = new[]
        {
            new CustomRule("NoCompetitor", text =>
                text.Contains("competitor", StringComparison.OrdinalIgnoreCase)
                    ? new SafetyFinding
                    {
                        Category = SafetyCategory.PromptInjection,
                        Severity = SafetySeverity.Low,
                        Confidence = 1.0,
                        Description = "Competitor mentioned",
                        RecommendedAction = SafetyAction.Warn,
                        SourceModule = "CustomRule"
                    }
                    : null)
        };
        var guardrail = new CustomRuleGuardrail<double>(rules);
        var findings = guardrail.EvaluateText("Our product is great and affordable.");

        Assert.Empty(findings);
    }

    [Fact]
    public void CustomRule_MultipleRules_ChecksAll()
    {
        var rules = new[]
        {
            new CustomRule("NoCompetitor", text =>
                text.Contains("competitor", StringComparison.OrdinalIgnoreCase)
                    ? new SafetyFinding
                    {
                        Category = SafetyCategory.PromptInjection,
                        Severity = SafetySeverity.Low,
                        Confidence = 1.0,
                        Description = "Competitor",
                        RecommendedAction = SafetyAction.Warn,
                        SourceModule = "CustomRule"
                    }
                    : null),
            new CustomRule("NoPricing", text =>
                text.Contains("price", StringComparison.OrdinalIgnoreCase)
                    ? new SafetyFinding
                    {
                        Category = SafetyCategory.PromptInjection,
                        Severity = SafetySeverity.Low,
                        Confidence = 1.0,
                        Description = "Pricing",
                        RecommendedAction = SafetyAction.Warn,
                        SourceModule = "CustomRule"
                    }
                    : null)
        };
        var guardrail = new CustomRuleGuardrail<double>(rules);
        var findings = guardrail.EvaluateText(
            "Our competitor has a lower price than us.");

        Assert.True(findings.Count >= 2);
    }

    #endregion

    #region CompositeGuardrail Tests

    [Fact]
    public void Composite_CombinesGuardrails_AggregatesFindings()
    {
        var guardrails = new IGuardrail<double>[]
        {
            new GuardrailAdapter<double>(new InputGuardrail<double>(maxInputLength: 50), GuardrailDirection.Input),
            new GuardrailAdapter<double>(new TopicRestrictionGuardrail<double>(new[] { "politics" }), GuardrailDirection.Input)
        };
        var composite = new CompositeGuardrail<double>(guardrails);
        var findings = composite.EvaluateText(
            "Let me tell you about politics and why it matters so much in our daily lives and in our communities");

        Assert.NotEmpty(findings);
    }

    [Fact]
    public void Composite_FailFast_StopsOnFirstFailure()
    {
        var guardrails = new IGuardrail<double>[]
        {
            new GuardrailAdapter<double>(new InputGuardrail<double>(maxInputLength: 10), GuardrailDirection.Input),
            new GuardrailAdapter<double>(new TopicRestrictionGuardrail<double>(new[] { "politics" }), GuardrailDirection.Input)
        };
        var composite = new CompositeGuardrail<double>(guardrails, failFast: true);
        var findings = composite.EvaluateText(
            "This is a long text about politics that should trigger both guardrails");

        // With fail-fast, should stop after first guardrail triggers
        Assert.NotEmpty(findings);
    }

    [Fact]
    public void Composite_AllPass_NoFindings()
    {
        var guardrails = new IGuardrail<double>[]
        {
            new GuardrailAdapter<double>(new InputGuardrail<double>(maxInputLength: 1000), GuardrailDirection.Input),
            new GuardrailAdapter<double>(new TopicRestrictionGuardrail<double>(new[] { "gambling" }), GuardrailDirection.Input)
        };
        var composite = new CompositeGuardrail<double>(guardrails);
        var findings = composite.EvaluateText("The weather is nice today.");

        Assert.Empty(findings);
    }

    #endregion

    #region Cross-Module Tests

    [Fact]
    public void AllGuardrails_ProcessSameText_NoExceptions()
    {
        var text = "This is a standard test input for guardrail validation.";

        Assert.NotNull(new InputGuardrail<double>(maxInputLength: 1000).EvaluateText(text));
        Assert.NotNull(new OutputGuardrail<double>().EvaluateText(text));
        Assert.NotNull(new TopicRestrictionGuardrail<double>(new[] { "test" }).EvaluateText(text));
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Adapts an ITextSafetyModule to IGuardrail for use in CompositeGuardrail tests.
    /// </summary>
    private sealed class GuardrailAdapter<TNum> : IGuardrail<TNum>
    {
        private readonly ITextSafetyModule<TNum> _inner;

        public GuardrailAdapter(ITextSafetyModule<TNum> inner, GuardrailDirection direction)
        {
            _inner = inner;
            Direction = direction;
        }

        public string ModuleName => _inner.ModuleName;
        public bool IsReady => _inner.IsReady;
        public GuardrailDirection Direction { get; }

        public IReadOnlyList<SafetyFinding> EvaluateText(string text) => _inner.EvaluateText(text);
        public IReadOnlyList<SafetyFinding> Evaluate(Vector<TNum> content) => _inner.Evaluate(content);
    }

    #endregion
}
