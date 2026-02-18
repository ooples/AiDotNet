namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the attestation policy for TEE remote attestation verification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Remote attestation lets a client verify that the server is actually
/// running inside a genuine TEE (not just pretending). The policy controls how strict this
/// verification is:</para>
/// <list type="bullet">
/// <item><description><b>Strict:</b> Exact measurement match required. Any deviation rejects the enclave.
/// Most secure but may break on minor firmware updates.</description></item>
/// <item><description><b>Relaxed:</b> Allows minor version differences in measurements.
/// More practical but slightly less secure.</description></item>
/// <item><description><b>Custom:</b> User-defined verification logic via callback.</description></item>
/// </list>
/// </remarks>
public enum AttestationPolicy
{
    /// <summary>Exact measurement match required.</summary>
    Strict,

    /// <summary>Allows minor version differences in enclave measurements.</summary>
    Relaxed,

    /// <summary>User-defined verification logic.</summary>
    Custom
}
