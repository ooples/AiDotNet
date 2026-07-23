namespace AiDotNet.Serving.Security;

/// <summary>
/// Represents resolved request context used for tier enforcement.
/// </summary>
public sealed class ServingRequestContext
{
    public required ServingTier Tier { get; init; }

    public required bool IsAuthenticated { get; init; }

    public string? ApiKeyId { get; init; }

    public string? Subject { get; init; }
}

