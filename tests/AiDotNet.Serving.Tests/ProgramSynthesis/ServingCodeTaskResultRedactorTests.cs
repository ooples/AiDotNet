using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Security;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests.ProgramSynthesis;

public sealed class ServingCodeTaskResultRedactorTests
{
    [Fact]
    public void Redact_FreeTier_RedactsErrorsAndExecutionTelemetry()
    {
        var redactor = CreateRedactor(maxChars: 5, maxItems: 5);

        var ctx = new ServingRequestContext { Tier = ServingTier.Free, IsAuthenticated = false };
        var result = new CodeSummarizationResult
        {
            Language = ProgramLanguage.CSharp,
            Success = false,
            Error = "details",
            Telemetry = new CodeTaskTelemetry { Execution = new CodeExecutionTelemetry { ExitCode = 1 } },
            Summary = "1234567890"
        };

        var redacted = (CodeSummarizationResult)redactor.Redact(result, ctx);
        Assert.False(redacted.Success);
        Assert.Equal("Request failed.", redacted.Error);
        Assert.NotNull(redacted.Telemetry);
        Assert.Null(redacted.Telemetry.Execution);
        Assert.Equal("12345", redacted.Summary);
    }

    [Fact]
    public void Redact_TruncatesCompletionCandidatesAndText()
    {
        var redactor = CreateRedactor(maxChars: 3, maxItems: 1);
        var ctx = new ServingRequestContext { Tier = ServingTier.Premium, IsAuthenticated = true };

        var result = new CodeCompletionResult
        {
            Language = ProgramLanguage.CSharp,
            Success = true,
            Candidates = new List<CodeCompletionCandidate>
            {
                new() { CompletionText = "abcdef" },
                new() { CompletionText = "xyz" }
            }
        };

        var redacted = (CodeCompletionResult)redactor.Redact(result, ctx);
        Assert.Single(redacted.Candidates);
        Assert.Equal("abc", redacted.Candidates[0].CompletionText);
    }

    [Fact]
    public void Redact_BugFixing_TruncatesDiffAndIssues()
    {
        var redactor = CreateRedactor(maxChars: 4, maxItems: 1);
        var ctx = new ServingRequestContext { Tier = ServingTier.Premium, IsAuthenticated = true };

        var result = new CodeBugFixingResult
        {
            Language = ProgramLanguage.CSharp,
            Success = true,
            FixedCode = "123456",
            Diff = new CodeTransformDiff
            {
                UnifiedDiff = "abcdef",
                Edits = new List<CodeEditOperation>
                {
                    new() { Text = "edit-one" },
                    new() { Text = "edit-two" }
                }
            },
            FixedIssues = new List<CodeIssue>
            {
                new() { Summary = "summary-too-long", Details = "details-too-long" },
                new() { Summary = "other" }
            }
        };

        var redacted = (CodeBugFixingResult)redactor.Redact(result, ctx);
        Assert.Equal("1234", redacted.FixedCode);
        Assert.NotNull(redacted.Diff);
        Assert.Equal("abcd", redacted.Diff!.UnifiedDiff);
        Assert.Single(redacted.Diff.Edits);
        Assert.Equal("edit", redacted.Diff.Edits[0].Text);
        Assert.Single(redacted.FixedIssues);
        Assert.Equal("summ", redacted.FixedIssues[0].Summary);
        Assert.Equal("deta", redacted.FixedIssues[0].Details);
    }

    [Fact]
    public void Redact_Review_TruncatesFixSuggestionsAndPlan()
    {
        var redactor = CreateRedactor(maxChars: 4, maxItems: 1);
        var ctx = new ServingRequestContext { Tier = ServingTier.Premium, IsAuthenticated = true };

        var result = new CodeReviewResult
        {
            Language = ProgramLanguage.CSharp,
            Success = true,
            Issues = new List<CodeIssue>
            {
                new() { Summary = "summary-too-long", Details = "details-too-long" },
                new() { Summary = "other" }
            },
            FixSuggestions = new List<CodeFixSuggestion>
            {
                new()
                {
                    Summary = "suggestion-too-long",
                    Diff = new CodeTransformDiff
                    {
                        UnifiedDiff = "unified-too-long",
                        Edits = new List<CodeEditOperation>
                        {
                            new() { Text = "edit-one" },
                            new() { Text = "edit-two" }
                        }
                    }
                },
                new() { Summary = "other" }
            },
            PrioritizedPlan = new List<string> { "plan-step-one", "plan-step-two" }
        };

        var redacted = (CodeReviewResult)redactor.Redact(result, ctx);
        Assert.Single(redacted.Issues);
        Assert.Equal("summ", redacted.Issues[0].Summary);
        Assert.Equal("deta", redacted.Issues[0].Details);

        Assert.Single(redacted.FixSuggestions);
        Assert.Equal("sugg", redacted.FixSuggestions[0].Summary);
        Assert.NotNull(redacted.FixSuggestions[0].Diff);
        Assert.Equal("unif", redacted.FixSuggestions[0].Diff!.UnifiedDiff);
        Assert.Single(redacted.FixSuggestions[0].Diff.Edits);
        Assert.Equal("edit", redacted.FixSuggestions[0].Diff.Edits[0].Text);

        Assert.Single(redacted.PrioritizedPlan);
        Assert.Equal("plan", redacted.PrioritizedPlan[0]);
    }

    [Fact]
    public void Redact_MaxItemsZero_ClearsLists()
    {
        var redactor = CreateRedactor(maxChars: 10, maxItems: 0);
        var ctx = new ServingRequestContext { Tier = ServingTier.Premium, IsAuthenticated = true };

        var result = new CodeCompletionResult
        {
            Language = ProgramLanguage.CSharp,
            Success = true,
            Candidates = new List<CodeCompletionCandidate> { new() { CompletionText = "a" } }
        };

        var redacted = (CodeCompletionResult)redactor.Redact(result, ctx);
        Assert.Empty(redacted.Candidates);
    }

    [Fact]
    public void Redact_ThrowsForUnknownResultType()
    {
        var redactor = CreateRedactor(maxChars: 10, maxItems: 10);
        var ctx = new ServingRequestContext { Tier = ServingTier.Premium, IsAuthenticated = true };

        Assert.Throws<NotSupportedException>(() => redactor.Redact(new UnknownResult(), ctx));
    }

    private static ServingCodeTaskResultRedactor CreateRedactor(int maxChars, int maxItems)
    {
        var options = new ServingProgramSynthesisOptions
        {
            Free = new ServingProgramSynthesisLimitOptions { MaxResultChars = maxChars, MaxListItems = maxItems, MaxTaskTimeSeconds = 1 },
            Premium = new ServingProgramSynthesisLimitOptions { MaxResultChars = maxChars, MaxListItems = maxItems, MaxTaskTimeSeconds = 1 },
            Enterprise = new ServingProgramSynthesisLimitOptions { MaxResultChars = maxChars, MaxListItems = maxItems, MaxTaskTimeSeconds = 1 }
        };

        return new ServingCodeTaskResultRedactor(Options.Create(options));
    }

    private sealed class UnknownResult : CodeTaskResultBase
    {
        public override CodeTask Task => (CodeTask)999;
    }
}
