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
            SubscriptionTier.Free => new TierPolicy(
                SubscriptionTier.Free,
                ModelArtifactAccessMode.ServerOnly,
                requireAttestationForJoin: true,
                requireAttestationForKeyRelease: false),
            SubscriptionTier.Pro => new TierPolicy(
                SubscriptionTier.Pro,
                ModelArtifactAccessMode.EncryptedWithKeyRelease,
                requireAttestationForJoin: true,
                requireAttestationForKeyRelease: false),
            _ => new TierPolicy(
                SubscriptionTier.Enterprise,
                ModelArtifactAccessMode.EncryptedWithAttestedKeyRelease,
                requireAttestationForJoin: true,
                requireAttestationForKeyRelease: true)
        };
    }
}
