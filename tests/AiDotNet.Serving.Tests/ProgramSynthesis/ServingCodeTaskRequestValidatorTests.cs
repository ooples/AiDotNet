using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Security;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests.ProgramSynthesis;

public sealed class ServingCodeTaskRequestValidatorTests
{
    private static ServingCodeTaskRequestValidator CreateValidator(ServingProgramSynthesisLimitOptions limits)
    {
        var options = new ServingProgramSynthesisOptions
        {
            Free = limits,
            Premium = limits,
            Enterprise = limits
        };

        return new ServingCodeTaskRequestValidator(Options.Create(options));
    }

    private static ServingRequestContext CreateContext(ServingTier tier) => new()
    {
        Tier = tier,
        IsAuthenticated = true,
        ApiKeyId = "key",
        Subject = "sub"
    };

    [Fact]
    public void TryValidate_Completion_EnforcesCursorOffsetAndCandidates()
    {
        var validator = CreateValidator(new ServingProgramSynthesisLimitOptions
        {
            MaxRequestChars = 10,
            MaxCorpusDocuments = 2,
            MaxCorpusDocumentChars = 10,
            MaxResultChars = 10,
            MaxListItems = 2,
            MaxConcurrentRequests = 1,
            MaxTaskTimeSeconds = 1
        });

        var context = CreateContext(ServingTier.Free);

        var request = new CodeCompletionRequest
        {
            Code = "abc",
            CursorOffset = 99,
            MaxCandidates = 1
        };

        Assert.False(validator.TryValidate(request, context, out var error));
        Assert.Equal("CursorOffset must be within the Code length.", error);

        request.CursorOffset = 1;
        request.MaxCandidates = 0;
        Assert.False(validator.TryValidate(request, context, out error));
        Assert.Equal("MaxCandidates must be >= 1.", error);

        request.MaxCandidates = 3;
        Assert.False(validator.TryValidate(request, context, out error));
        Assert.Equal("MaxCandidates exceeds tier limit (2).", error);
    }

    [Fact]
    public void TryValidate_Generation_RequiresDescriptionOrExamples_AndAppliesExampleLimits()
    {
        var validator = CreateValidator(new ServingProgramSynthesisLimitOptions
        {
            MaxRequestChars = 10,
            MaxCorpusDocuments = 2,
            MaxCorpusDocumentChars = 10,
            MaxResultChars = 10,
            MaxListItems = 2,
            MaxConcurrentRequests = 1,
            MaxTaskTimeSeconds = 1
        });

        var context = CreateContext(ServingTier.Free);

        var request = new CodeGenerationRequest
        {
            Description = "   ",
            Examples = new List<ProgramInputOutputExample>()
        };

        Assert.False(validator.TryValidate(request, context, out var error));
        Assert.Equal("Description or Examples is required.", error);

        request.Description = "01234567890";
        Assert.False(validator.TryValidate(request, context, out error));
        Assert.Equal("Description exceeds tier limit (10 chars).", error);

        request.Description = "ok";
        request.Examples = new List<ProgramInputOutputExample>
        {
            new ProgramInputOutputExample { Input = "a", ExpectedOutput = "b" },
            new ProgramInputOutputExample { Input = "a", ExpectedOutput = "b" },
            new ProgramInputOutputExample { Input = "a", ExpectedOutput = "b" }
        };

        Assert.False(validator.TryValidate(request, context, out error));
        Assert.Equal("Examples exceeds tier limit (2).", error);
    }

    [Fact]
    public void TryValidate_Search_AndCorpus_Validation_EnforcesLimits()
    {
        var validator = CreateValidator(new ServingProgramSynthesisLimitOptions
        {
            MaxRequestChars = 10,
            MaxCorpusDocuments = 2,
            MaxCorpusDocumentChars = 5,
            MaxResultChars = 10,
            MaxListItems = 2,
            MaxConcurrentRequests = 1,
            MaxTaskTimeSeconds = 1
        });

        var context = CreateContext(ServingTier.Free);

        var request = new CodeSearchRequest
        {
            Query = "ok",
            Corpus = new CodeCorpusReference
            {
                CorpusId = "server-side"
            }
        };

        Assert.False(validator.TryValidate(request, context, out var error));
        Assert.Equal("Serving-indexed corpora are not implemented in this build; provide request-scoped Documents.", error);

        request.Corpus = new CodeCorpusReference
        {
            Documents = new List<CodeCorpusDocument>
            {
                new CodeCorpusDocument { DocumentId = "d1", FilePath = "a.cs", Content = "" }
            }
        };

        Assert.False(validator.TryValidate(request, context, out error));
        Assert.Equal("Corpus.Documents[0].Content is required.", error);

        request.Corpus.Documents[0]!.Content = "012345";
        Assert.False(validator.TryValidate(request, context, out error));
        Assert.Equal("Corpus.Documents[0].Content exceeds tier limit (5 chars).", error);

        request.Corpus.Documents[0]!.Content = "12345";
        request.Filters = new List<string> { "a", "b", "c" };
        Assert.False(validator.TryValidate(request, context, out error));
        Assert.Equal("Filters exceeds tier limit (2).", error);
    }

    [Fact]
    public void TryValidate_CloneDetection_ValidatesMinSimilarity()
    {
        var validator = CreateValidator(new ServingProgramSynthesisLimitOptions
        {
            MaxRequestChars = 10,
            MaxCorpusDocuments = 2,
            MaxCorpusDocumentChars = 10,
            MaxResultChars = 10,
            MaxListItems = 2,
            MaxConcurrentRequests = 1,
            MaxTaskTimeSeconds = 1
        });

        var context = CreateContext(ServingTier.Free);

        var request = new CodeCloneDetectionRequest
        {
            Corpus = new CodeCorpusReference
            {
                Documents = new List<CodeCorpusDocument>
                {
                    new CodeCorpusDocument { Content = "a" }
                }
            },
            MinSimilarity = -0.1
        };

        Assert.False(validator.TryValidate(request, context, out var error));
        Assert.Equal("Corpus must include at least 2 documents.", error);

        request.Corpus.Documents.Add(new CodeCorpusDocument { Content = "b" });
        Assert.False(validator.TryValidate(request, context, out error));
        Assert.Equal("MinSimilarity must be between 0 and 1.", error);
    }

    [Fact]
    public void TryValidate_MaxWallClockMilliseconds_IsClampedByTier()
    {
        var validator = CreateValidator(new ServingProgramSynthesisLimitOptions
        {
            MaxRequestChars = 100,
            MaxCorpusDocuments = 2,
            MaxCorpusDocumentChars = 10,
            MaxResultChars = 10,
            MaxListItems = 2,
            MaxConcurrentRequests = 1,
            MaxTaskTimeSeconds = 1
        });

        var context = CreateContext(ServingTier.Free);

        var request = new CodeSummarizationRequest
        {
            Code = "ok",
            MaxWallClockMilliseconds = 0
        };

        Assert.False(validator.TryValidate(request, context, out var error));
        Assert.Equal("MaxWallClockMilliseconds must be > 0.", error);

        request.MaxWallClockMilliseconds = 2000;
        Assert.False(validator.TryValidate(request, context, out error));
        Assert.Equal("MaxWallClockMilliseconds exceeds tier limit (1000 ms).", error);

        request.MaxWallClockMilliseconds = 1000;
        Assert.True(validator.TryValidate(request, context, out error));
        Assert.Equal(string.Empty, error);
    }
}
