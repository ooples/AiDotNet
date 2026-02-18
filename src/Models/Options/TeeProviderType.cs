namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the Trusted Execution Environment hardware provider.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A TEE is a secure area in a processor that guarantees code and data
/// loaded inside are protected from the outside world — even from the operating system.
/// Different chip vendors provide different TEE implementations:</para>
/// <list type="bullet">
/// <item><description><b>Sgx:</b> Intel SGX — process-level enclaves, 256MB memory limit.</description></item>
/// <item><description><b>Tdx:</b> Intel TDX — confidential VMs with GB-scale memory (recommended).</description></item>
/// <item><description><b>SevSnp:</b> AMD SEV-SNP — full memory encryption for VMs.</description></item>
/// <item><description><b>ArmCca:</b> ARM CCA/Realms — for ARM-based edge and cloud servers.</description></item>
/// <item><description><b>Simulated:</b> Software simulation for testing without hardware.</description></item>
/// </list>
/// </remarks>
public enum TeeProviderType
{
    /// <summary>Intel SGX — process-level enclave with 256MB EPC limit.</summary>
    Sgx,

    /// <summary>Intel TDX — confidential VM with GB-scale protected memory.</summary>
    Tdx,

    /// <summary>AMD SEV-SNP — VM-level memory encryption.</summary>
    SevSnp,

    /// <summary>ARM CCA/Realms — ARM confidential computing architecture.</summary>
    ArmCca,

    /// <summary>Software simulation for testing without hardware.</summary>
    Simulated
}
