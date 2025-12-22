namespace AiDotNet.Serving.Security;

/// <summary>
/// Scopes granted to an API key.
/// </summary>
[Flags]
public enum ApiKeyScopes
{
    None = 0,

    Inference = 1 << 0,

    Federated = 1 << 1,

    ArtifactDownload = 1 << 2,

    ArtifactKeyRelease = 1 << 3,

    Admin = 1 << 4,

    All = Inference | Federated | ArtifactDownload | ArtifactKeyRelease | Admin
}

