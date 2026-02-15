using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Security;
using Microsoft.Extensions.Options;
using AiDotNet.Validation;

namespace AiDotNet.Serving.ProgramSynthesis;

public sealed class ServingCodeTaskResultRedactor : IServingCodeTaskResultRedactor
{
    private readonly ServingProgramSynthesisOptions _options;

    public ServingCodeTaskResultRedactor(IOptions<ServingProgramSynthesisOptions> options)
    {
        Guard.NotNull(options);
        Guard.NotNull(options.Value);
        _options = options.Value;
    }

    public CodeTaskResultBase Redact(CodeTaskResultBase result, ServingRequestContext requestContext)
    {
        if (result is null) throw new ArgumentNullException(nameof(result));
        if (requestContext is null) throw new ArgumentNullException(nameof(requestContext));

        var limits = GetLimits(requestContext.Tier);

        if (requestContext.Tier == ServingTier.Free)
        {
            if (!result.Success)
            {
                result.Error = "Request failed.";
            }

            if (result.Telemetry is not null)
            {
                result.Telemetry.Execution = null;
            }
        }

        switch (result)
        {
            case CodeCompletionResult completion:
                TruncateListInPlace(completion.Candidates, limits.MaxListItems);
                foreach (var candidate in completion.Candidates)
                {
                    candidate.CompletionText = Truncate(candidate.CompletionText, limits.MaxResultChars);
                }

                break;

            case CodeGenerationResult generation:
                generation.GeneratedCode = Truncate(generation.GeneratedCode, limits.MaxResultChars);
                break;

            case CodeTranslationResult translation:
                translation.TranslatedCode = Truncate(translation.TranslatedCode, limits.MaxResultChars);
                break;

            case CodeSummarizationResult summarization:
                summarization.Summary = Truncate(summarization.Summary, limits.MaxResultChars);
                break;

            case CodeBugDetectionResult bugDetection:
                TruncateListInPlace(bugDetection.Issues, limits.MaxListItems);
                foreach (var issue in bugDetection.Issues)
                {
                    RedactIssue(issue, limits);
                }

                break;

            case CodeBugFixingResult bugFixing:
                bugFixing.FixedCode = Truncate(bugFixing.FixedCode, limits.MaxResultChars);
                RedactDiff(bugFixing.Diff, limits);
                TruncateListInPlace(bugFixing.FixedIssues, limits.MaxListItems);
                foreach (var issue in bugFixing.FixedIssues)
                {
                    RedactIssue(issue, limits);
                }

                break;

            case CodeRefactoringResult refactoring:
                refactoring.RefactoredCode = Truncate(refactoring.RefactoredCode, limits.MaxResultChars);
                RedactDiff(refactoring.Diff, limits);
                break;

            case CodeUnderstandingResult understanding:
                TruncateListInPlace(understanding.Symbols, limits.MaxListItems);
                TruncateListInPlace(understanding.Dependencies, limits.MaxListItems);
                TruncateListInPlace(understanding.CallGraph, limits.MaxListItems);
                TruncateListInPlace(understanding.Hotspots, limits.MaxListItems);
                TruncateListInPlace(understanding.ControlFlowSummaries, limits.MaxListItems);
                TruncateListInPlace(understanding.DataFlowSummaries, limits.MaxListItems);
                TruncateListInPlace(understanding.SecurityHotspots, limits.MaxListItems);
                break;

            case CodeTestGenerationResult testGeneration:
                TruncateListInPlace(testGeneration.Tests, limits.MaxListItems);
                for (var i = 0; i < testGeneration.Tests.Count; i++)
                {
                    testGeneration.Tests[i] = Truncate(testGeneration.Tests[i], limits.MaxResultChars);
                }

                break;

            case CodeDocumentationResult documentation:
                documentation.Documentation = Truncate(documentation.Documentation, limits.MaxResultChars);
                documentation.UpdatedCode = Truncate(documentation.UpdatedCode, limits.MaxResultChars);
                break;

            case CodeSearchResult search:
                TruncateListInPlace(search.FiltersApplied, limits.MaxListItems);
                TruncateListInPlace(search.Results, limits.MaxListItems);
                foreach (var hit in search.Results)
                {
                    hit.SnippetText = Truncate(hit.SnippetText, limits.MaxResultChars);
                    hit.MatchExplanation = hit.MatchExplanation is null ? null : Truncate(hit.MatchExplanation, limits.MaxResultChars);
                }

                break;

            case CodeCloneDetectionResult cloneDetection:
                TruncateListInPlace(cloneDetection.CloneGroups, limits.MaxListItems);
                foreach (var group in cloneDetection.CloneGroups)
                {
                    group.NormalizationSummary = group.NormalizationSummary is null ? null : Truncate(group.NormalizationSummary, limits.MaxResultChars);
                    TruncateListInPlace(group.Instances, limits.MaxListItems);
                    TruncateListInPlace(group.RefactorSuggestions, limits.MaxListItems);

                    foreach (var instance in group.Instances)
                    {
                        instance.SnippetText = Truncate(instance.SnippetText, limits.MaxResultChars);
                    }
                }

                break;

            case CodeReviewResult review:
                TruncateListInPlace(review.Issues, limits.MaxListItems);
                foreach (var issue in review.Issues)
                {
                    RedactIssue(issue, limits);
                }

                TruncateListInPlace(review.FixSuggestions, limits.MaxListItems);
                foreach (var suggestion in review.FixSuggestions)
                {
                    suggestion.Summary = Truncate(suggestion.Summary, limits.MaxResultChars);
                    suggestion.Rationale = suggestion.Rationale is null ? null : Truncate(suggestion.Rationale, limits.MaxResultChars);
                    suggestion.FixGuidance = suggestion.FixGuidance is null ? null : Truncate(suggestion.FixGuidance, limits.MaxResultChars);
                    suggestion.TestGuidance = suggestion.TestGuidance is null ? null : Truncate(suggestion.TestGuidance, limits.MaxResultChars);
                    if (suggestion.Diff is not null)
                    {
                        RedactDiff(suggestion.Diff, limits);
                    }
                }

                TruncateListInPlace(review.PrioritizedPlan, limits.MaxListItems);
                for (var i = 0; i < review.PrioritizedPlan.Count; i++)
                {
                    review.PrioritizedPlan[i] = Truncate(review.PrioritizedPlan[i], limits.MaxResultChars);
                }

                break;

            default:
                throw new NotSupportedException($"Result type {result.GetType().Name} is not supported for redaction.");
        }

        return result;
    }

    private ServingProgramSynthesisLimitOptions GetLimits(ServingTier tier) =>
        tier switch
        {
            ServingTier.Premium => _options.Premium,
            ServingTier.Enterprise => _options.Enterprise,
            _ => _options.Free
        };

    private static string Truncate(string value, int maxChars)
    {
        if (string.IsNullOrEmpty(value))
        {
            return value ?? string.Empty;
        }

        if (maxChars <= 0)
        {
            return string.Empty;
        }

        return value.Length <= maxChars ? value : value.Substring(0, maxChars);
    }

    private static void TruncateListInPlace<T>(List<T> list, int maxItems)
    {
        if (list is null || maxItems <= 0)
        {
            list?.Clear();
            return;
        }

        if (list.Count <= maxItems)
        {
            return;
        }

        list.RemoveRange(maxItems, list.Count - maxItems);
    }

    private static void RedactIssue(CodeIssue issue, ServingProgramSynthesisLimitOptions limits)
    {
        if (issue is null)
        {
            return;
        }

        issue.Summary = Truncate(issue.Summary, limits.MaxResultChars);
        issue.Details = issue.Details is null ? null : Truncate(issue.Details, limits.MaxResultChars);
        issue.Rationale = issue.Rationale is null ? null : Truncate(issue.Rationale, limits.MaxResultChars);
        issue.FixGuidance = issue.FixGuidance is null ? null : Truncate(issue.FixGuidance, limits.MaxResultChars);
        issue.TestGuidance = issue.TestGuidance is null ? null : Truncate(issue.TestGuidance, limits.MaxResultChars);
    }

    private static void RedactDiff(CodeTransformDiff diff, ServingProgramSynthesisLimitOptions limits)
    {
        if (diff is null)
        {
            return;
        }

        diff.UnifiedDiff = diff.UnifiedDiff is null ? null : Truncate(diff.UnifiedDiff, limits.MaxResultChars);
        TruncateListInPlace(diff.Edits, limits.MaxListItems);
        if (diff.Edits is null)
        {
            return;
        }

        foreach (var edit in diff.Edits)
        {
            edit.Text = edit.Text is null ? null : Truncate(edit.Text, limits.MaxResultChars);
        }
    }
}
