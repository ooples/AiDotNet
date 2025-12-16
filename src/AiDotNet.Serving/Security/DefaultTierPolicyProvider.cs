namespace AiDotNet.Serving.Security;

/// <summary>
/// Default tier policy provider implementing Options A/B/C.
/// </summary>
public sealed class DefaultTierPolicyProvider : ITierPolicyProvider
{
    public TierPolicy GetPolicy(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Free => new TierPolicy(SubscriptionTier.Free, ModelArtifactAccessMode.ServerOnly, requireAttestationForKeyRelease: false),
            SubscriptionTier.Pro => new TierPolicy(SubscriptionTier.Pro, ModelArtifactAccessMode.DirectDownload, requireAttestationForKeyRelease: false),
            _ => new TierPolicy(SubscriptionTier.Enterprise, ModelArtifactAccessMode.EncryptedWithAttestedKeyRelease, requireAttestationForKeyRelease: true)
        };
    }
}

