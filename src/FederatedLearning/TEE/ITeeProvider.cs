namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// Abstracts a Trusted Execution Environment backend for enclave lifecycle, data sealing,
/// and attestation quote generation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A TEE provider manages a hardware-isolated "vault" (enclave) on a processor.
/// Think of it like a bank's safe-deposit box: you can put data in (seal), take data out (unseal),
/// and ask the bank for a signed letter proving the box exists (generate attestation quote).</para>
///
/// <para><b>Implementations:</b></para>
/// <list type="bullet">
/// <item><description><see cref="IntelSgxTeeProvider{T}"/>: Intel SGX process-level enclaves (256 MB EPC).</description></item>
/// <item><description><see cref="IntelTdxTeeProvider{T}"/>: Intel TDX confidential VMs (GB-scale).</description></item>
/// <item><description><see cref="AmdSevSnpTeeProvider{T}"/>: AMD SEV-SNP VM-level encryption.</description></item>
/// <item><description><see cref="ArmCcaTeeProvider{T}"/>: ARM CCA/Realms for ARM servers.</description></item>
/// <item><description><see cref="SimulatedTeeProvider{T}"/>: Software simulation for testing.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public interface ITeeProvider<T>
{
    /// <summary>
    /// Gets the TEE provider type.
    /// </summary>
    AiDotNet.Models.Options.TeeProviderType ProviderType { get; }

    /// <summary>
    /// Gets a value indicating whether the enclave is currently initialized and ready.
    /// </summary>
    bool IsInitialized { get; }

    /// <summary>
    /// Initializes the TEE enclave. Must be called before other operations.
    /// </summary>
    /// <param name="options">TEE configuration.</param>
    void Initialize(AiDotNet.Models.Options.TeeOptions options);

    /// <summary>
    /// Destroys the enclave and releases all protected resources.
    /// </summary>
    void Destroy();

    /// <summary>
    /// Seals (encrypts) data to the enclave so only this enclave can unseal it.
    /// </summary>
    /// <param name="plaintext">Data to seal.</param>
    /// <returns>Sealed (encrypted) data bound to this enclave identity.</returns>
    byte[] SealData(byte[] plaintext);

    /// <summary>
    /// Unseals (decrypts) data that was previously sealed by this enclave.
    /// </summary>
    /// <param name="sealedData">Sealed data.</param>
    /// <returns>Original plaintext.</returns>
    byte[] UnsealData(byte[] sealedData);

    /// <summary>
    /// Generates a remote attestation quote proving this enclave's identity and integrity.
    /// </summary>
    /// <param name="reportData">Application data (up to 64 bytes) to bind into the quote.</param>
    /// <returns>Platform-signed attestation quote.</returns>
    byte[] GenerateAttestationQuote(byte[] reportData);

    /// <summary>
    /// Gets the measurement hash (code identity) of the running enclave.
    /// </summary>
    /// <returns>Hex-encoded measurement hash.</returns>
    string GetMeasurementHash();

    /// <summary>
    /// Gets the maximum memory (in bytes) available inside this enclave.
    /// </summary>
    long GetMaxEnclaveMemory();
}
