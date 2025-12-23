namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Tier-scoped limits for Program Synthesis requests and responses in AiDotNet.Serving.
/// </summary>
public sealed class ServingProgramSynthesisLimitOptions
{
    /// <summary>
    /// Maximum allowed characters for request text fields (code, description, query, etc.).
    /// </summary>
    public int MaxRequestChars { get; set; } = 50_000;

    /// <summary>
    /// Maximum number of request-scoped corpus documents allowed in a single request.
    /// </summary>
    public int MaxCorpusDocuments { get; set; } = 25;

    /// <summary>
    /// Maximum allowed characters per request-scoped corpus document.
    /// </summary>
    public int MaxCorpusDocumentChars { get; set; } = 20_000;

    /// <summary>
    /// Maximum allowed characters for response text fields (generated code, summaries, etc.).
    /// </summary>
    public int MaxResultChars { get; set; } = 16_000;

    /// <summary>
    /// Maximum number of list items returned in results (candidates, issues, hits, etc.).
    /// </summary>
    public int MaxListItems { get; set; } = 50;

    /// <summary>
    /// Maximum number of concurrent Program Synthesis task requests allowed per tier.
    /// </summary>
    public int MaxConcurrentRequests { get; set; } = 8;

    /// <summary>
    /// Hard wall-clock time limit for Program Synthesis task endpoints (in seconds).
    /// </summary>
    public int MaxTaskTimeSeconds { get; set; } = 5;

    public void Validate(string tierName)
    {
        if (string.IsNullOrWhiteSpace(tierName))
        {
            tierName = "Unknown";
        }

        if (MaxRequestChars <= 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxRequestChars must be > 0.");
        }

        if (MaxCorpusDocuments < 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxCorpusDocuments must be >= 0.");
        }

        if (MaxCorpusDocumentChars < 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxCorpusDocumentChars must be >= 0.");
        }

        if (MaxResultChars <= 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxResultChars must be > 0.");
        }

        if (MaxListItems <= 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxListItems must be > 0.");
        }

        if (MaxConcurrentRequests <= 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxConcurrentRequests must be > 0.");
        }

        if (MaxTaskTimeSeconds <= 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxTaskTimeSeconds must be > 0.");
        }
    }
}

