namespace AiDotNet.Serving.Security;

/// <summary>
/// Policy for a given subscription tier.
/// </summary>
public sealed class TierPolicy
{
    public TierPolicy(
        SubscriptionTier tier,
        ModelArtifactAccessMode artifactAccessMode,
        bool requireAttestationForJoin,
        bool requireAttestationForKeyRelease)
    {
        Tier = tier;
        ArtifactAccessMode = artifactAccessMode;
        RequireAttestationForJoin = requireAttestationForJoin;
        RequireAttestationForKeyRelease = requireAttestationForKeyRelease;
    }

    /// <summary>
    /// Gets the tier.
    /// </summary>
    public SubscriptionTier Tier { get; }

    /// <summary>
    /// Gets the model artifact access mode.
    /// </summary>
    public ModelArtifactAccessMode ArtifactAccessMode { get; }

    /// <summary>
    /// Gets whether joining a federated run requires attestation.
    /// </summary>
    public bool RequireAttestationForJoin { get; }

    /// <summary>
    /// Gets whether key release requires attestation.
    /// </summary>
    public bool RequireAttestationForKeyRelease { get; }

    /// <summary>
    /// Gets whether downloading an artifact is allowed for this tier.
    /// </summary>
    public bool AllowArtifactDownload => ArtifactAccessMode != ModelArtifactAccessMode.ServerOnly;

    /// <summary>
    /// Gets whether the artifact is expected to be encrypted.
    /// </summary>
    public bool ArtifactIsEncrypted =>
        ArtifactAccessMode == ModelArtifactAccessMode.EncryptedWithKeyRelease ||
        ArtifactAccessMode == ModelArtifactAccessMode.EncryptedWithAttestedKeyRelease;

    /// <summary>
    /// Gets whether key release is allowed.
    /// </summary>
    public bool AllowKeyRelease => ArtifactIsEncrypted;
}
