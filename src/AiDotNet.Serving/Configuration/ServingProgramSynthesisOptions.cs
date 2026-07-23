namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Serving configuration for Program Synthesis features (code tasks, corpora, evaluation).
/// </summary>
public sealed class ServingProgramSynthesisOptions
{
    public ServingProgramSynthesisLimitOptions Free { get; set; } = new()
    {
        MaxRequestChars = 50_000,
        MaxCorpusDocuments = 25,
        MaxCorpusDocumentChars = 20_000,
        MaxResultChars = 16_000,
        MaxListItems = 50,
        MaxConcurrentRequests = 4,
        MaxTaskTimeSeconds = 2
    };

    public ServingProgramSynthesisLimitOptions Premium { get; set; } = new()
    {
        MaxRequestChars = 200_000,
        MaxCorpusDocuments = 250,
        MaxCorpusDocumentChars = 100_000,
        MaxResultChars = 64_000,
        MaxListItems = 200,
        MaxConcurrentRequests = 16,
        MaxTaskTimeSeconds = 5
    };

    public ServingProgramSynthesisLimitOptions Enterprise { get; set; } = new()
    {
        MaxRequestChars = 1_000_000,
        MaxCorpusDocuments = 2_000,
        MaxCorpusDocumentChars = 250_000,
        MaxResultChars = 256_000,
        MaxListItems = 1_000,
        MaxConcurrentRequests = 64,
        MaxTaskTimeSeconds = 15
    };
}

