using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Security;
using Microsoft.Extensions.Options;

namespace AiDotNet.Serving.ProgramSynthesis;

public sealed class ServingCodeTaskRequestValidator : IServingCodeTaskRequestValidator
{
    private readonly ServingProgramSynthesisOptions _options;

    public ServingCodeTaskRequestValidator(IOptions<ServingProgramSynthesisOptions> options)
    {
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
    }

    public bool TryValidate(
        CodeTaskRequestBase request,
        ServingRequestContext requestContext,
        out string error)
    {
        if (request is null) throw new ArgumentNullException(nameof(request));
        if (requestContext is null) throw new ArgumentNullException(nameof(requestContext));

        var limits = GetLimits(requestContext.Tier);

        switch (request)
        {
            case CodeCompletionRequest completion:
                if (!RequireNonEmpty(completion.Code, "Code", out error) ||
                    !RequireMaxChars(completion.Code, limits.MaxRequestChars, "Code", out error))
                {
                    return false;
                }

                if (completion.CursorOffset is not null &&
                    (completion.CursorOffset.Value < 0 || completion.CursorOffset.Value > completion.Code.Length))
                {
                    error = "CursorOffset must be within the Code length.";
                    return false;
                }

                if (completion.MaxCandidates <= 0)
                {
                    error = "MaxCandidates must be >= 1.";
                    return false;
                }

                if (completion.MaxCandidates > limits.MaxListItems)
                {
                    error = $"MaxCandidates exceeds tier limit ({limits.MaxListItems}).";
                    return false;
                }

                break;

            case CodeGenerationRequest generation:
                var hasDescription = !string.IsNullOrWhiteSpace(generation.Description);
                var hasExamples = generation.Examples is not null && generation.Examples.Count > 0;

                if (!hasDescription && !hasExamples)
                {
                    error = "Description or Examples is required.";
                    return false;
                }

                if (hasDescription &&
                    !RequireMaxChars(generation.Description, limits.MaxRequestChars, "Description", out error))
                {
                    return false;
                }

                if (generation.Examples is not null)
                {
                    if (generation.Examples.Count > limits.MaxListItems)
                    {
                        error = $"Examples exceeds tier limit ({limits.MaxListItems}).";
                        return false;
                    }

                    for (var i = 0; i < generation.Examples.Count; i++)
                    {
                        var example = generation.Examples[i] ?? new ProgramInputOutputExample();

                        if (!RequireMaxChars(example.Input, limits.MaxRequestChars, $"Examples[{i}].Input", out error) ||
                            !RequireMaxChars(example.ExpectedOutput, limits.MaxRequestChars, $"Examples[{i}].ExpectedOutput", out error))
                        {
                            return false;
                        }
                    }
                }

                break;

            case CodeTranslationRequest translation:
                if (!RequireNonEmpty(translation.Code, "Code", out error) ||
                    !RequireMaxChars(translation.Code, limits.MaxRequestChars, "Code", out error))
                {
                    return false;
                }

                break;

            case CodeSummarizationRequest summarization:
                if (!RequireNonEmpty(summarization.Code, "Code", out error) ||
                    !RequireMaxChars(summarization.Code, limits.MaxRequestChars, "Code", out error))
                {
                    return false;
                }

                break;

            case CodeBugDetectionRequest bugDetection:
                if (!RequireNonEmpty(bugDetection.Code, "Code", out error) ||
                    !RequireMaxChars(bugDetection.Code, limits.MaxRequestChars, "Code", out error))
                {
                    return false;
                }

                break;

            case CodeBugFixingRequest bugFixing:
                if (!RequireNonEmpty(bugFixing.Code, "Code", out error) ||
                    !RequireMaxChars(bugFixing.Code, limits.MaxRequestChars, "Code", out error))
                {
                    return false;
                }

                if (bugFixing.BugDescription is not null &&
                    !RequireMaxChars(bugFixing.BugDescription, limits.MaxRequestChars, "BugDescription", out error))
                {
                    return false;
                }

                break;

            case CodeRefactoringRequest refactoring:
                if (!RequireNonEmpty(refactoring.Code, "Code", out error) ||
                    !RequireMaxChars(refactoring.Code, limits.MaxRequestChars, "Code", out error))
                {
                    return false;
                }

                if (refactoring.Goal is not null &&
                    !RequireMaxChars(refactoring.Goal, limits.MaxRequestChars, "Goal", out error))
                {
                    return false;
                }

                break;

            case CodeUnderstandingRequest understanding:
                if (!RequireNonEmpty(understanding.Code, "Code", out error) ||
                    !RequireMaxChars(understanding.Code, limits.MaxRequestChars, "Code", out error))
                {
                    return false;
                }

                if (understanding.FilePath is not null &&
                    !RequireMaxChars(understanding.FilePath, limits.MaxRequestChars, "FilePath", out error))
                {
                    return false;
                }

                break;

            case CodeTestGenerationRequest testGeneration:
                if (!RequireNonEmpty(testGeneration.Code, "Code", out error) ||
                    !RequireMaxChars(testGeneration.Code, limits.MaxRequestChars, "Code", out error))
                {
                    return false;
                }

                break;

            case CodeDocumentationRequest documentation:
                if (!RequireNonEmpty(documentation.Code, "Code", out error) ||
                    !RequireMaxChars(documentation.Code, limits.MaxRequestChars, "Code", out error))
                {
                    return false;
                }

                break;

            case CodeSearchRequest search:
                if (!RequireNonEmpty(search.Query, "Query", out error) ||
                    !RequireMaxChars(search.Query, limits.MaxRequestChars, "Query", out error))
                {
                    return false;
                }

                if (!TryValidateCorpus(search.Corpus, minDocumentCount: 1, limits, out error))
                {
                    return false;
                }

                if (search.Filters is not null && search.Filters.Count > limits.MaxListItems)
                {
                    error = $"Filters exceeds tier limit ({limits.MaxListItems}).";
                    return false;
                }

                break;

            case CodeCloneDetectionRequest cloneDetection:
                if (!TryValidateCorpus(cloneDetection.Corpus, minDocumentCount: 2, limits, out error))
                {
                    return false;
                }

                if (cloneDetection.MinSimilarity < 0.0 || cloneDetection.MinSimilarity > 1.0)
                {
                    error = "MinSimilarity must be between 0 and 1.";
                    return false;
                }

                break;

            case CodeReviewRequest review:
                if (!RequireNonEmpty(review.Code, "Code", out error) ||
                    !RequireMaxChars(review.Code, limits.MaxRequestChars, "Code", out error))
                {
                    return false;
                }

                if (review.FilePath is not null &&
                    !RequireMaxChars(review.FilePath, limits.MaxRequestChars, "FilePath", out error))
                {
                    return false;
                }

                break;

            default:
                error = "Unsupported request type.";
                return false;
        }

        if (request.MaxWallClockMilliseconds is not null)
        {
            if (request.MaxWallClockMilliseconds.Value <= 0)
            {
                error = "MaxWallClockMilliseconds must be > 0.";
                return false;
            }

            var maxTierMs = checked(limits.MaxTaskTimeSeconds * 1000);
            if (request.MaxWallClockMilliseconds.Value > maxTierMs)
            {
                error = $"MaxWallClockMilliseconds exceeds tier limit ({maxTierMs} ms).";
                return false;
            }
        }

        error = string.Empty;
        return true;
    }

    private ServingProgramSynthesisLimitOptions GetLimits(ServingTier tier) =>
        tier switch
        {
            ServingTier.Premium => _options.Premium,
            ServingTier.Enterprise => _options.Enterprise,
            _ => _options.Free
        };

    private static bool RequireNonEmpty(string? value, string fieldName, out string error)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            error = $"{fieldName} is required.";
            return false;
        }

        error = string.Empty;
        return true;
    }

    private static bool RequireMaxChars(string? value, int maxChars, string fieldName, out string error)
    {
        if (value is null)
        {
            error = $"{fieldName} is required.";
            return false;
        }

        if (value.Length > maxChars)
        {
            error = $"{fieldName} exceeds tier limit ({maxChars} chars).";
            return false;
        }

        error = string.Empty;
        return true;
    }

    private static bool TryValidateCorpus(
        CodeCorpusReference corpus,
        int minDocumentCount,
        ServingProgramSynthesisLimitOptions limits,
        out string error)
    {
        if (corpus is null)
        {
            error = "Corpus is required.";
            return false;
        }

        var documents = corpus.Documents ?? new List<CodeCorpusDocument>();

        if (!string.IsNullOrWhiteSpace(corpus.CorpusId) || !string.IsNullOrWhiteSpace(corpus.IndexId))
        {
            error = "Serving-indexed corpora are not implemented in this build; provide request-scoped Documents.";
            return false;
        }

        if (documents.Count < minDocumentCount)
        {
            error = minDocumentCount == 1
                ? "Corpus must include at least 1 document."
                : $"Corpus must include at least {minDocumentCount} documents.";
            return false;
        }

        if (documents.Count > limits.MaxCorpusDocuments)
        {
            error = $"Corpus exceeds tier limit ({limits.MaxCorpusDocuments} documents).";
            return false;
        }

        for (var i = 0; i < documents.Count; i++)
        {
            var document = documents[i] ?? new CodeCorpusDocument();

            if (!RequireNonEmpty(document.Content, $"Corpus.Documents[{i}].Content", out error))
            {
                return false;
            }

            if (!RequireMaxChars(document.Content, limits.MaxCorpusDocumentChars, $"Corpus.Documents[{i}].Content", out error))
            {
                return false;
            }

            if (document.FilePath is not null &&
                !RequireMaxChars(document.FilePath, limits.MaxRequestChars, $"Corpus.Documents[{i}].FilePath", out error))
            {
                return false;
            }

            if (document.DocumentId is not null &&
                !RequireMaxChars(document.DocumentId, limits.MaxRequestChars, $"Corpus.Documents[{i}].DocumentId", out error))
            {
                return false;
            }
        }

        error = string.Empty;
        return true;
    }
}
