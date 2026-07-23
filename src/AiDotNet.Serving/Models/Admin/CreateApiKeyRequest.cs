using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.Models.Admin;

public sealed class CreateApiKeyRequest
{
    public ServingTier Tier { get; set; } = ServingTier.Premium;

    public DateTimeOffset? ExpiresAt { get; set; }

    public string? DisplayName { get; set; }
}

