using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// Base class for TEE providers with common enclave lifecycle, sealing, and attestation logic.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> All TEE providers share common patterns — initializing an enclave,
/// sealing/unsealing data, generating attestation quotes. This base class implements those patterns
/// so that each hardware-specific provider (SGX, TDX, SEV-SNP, etc.) only needs to implement
/// the platform-specific parts.</para>
///
/// <para><b>Data sealing</b> uses AES-256-GCM with a key derived from the enclave identity.
/// In simulation mode the key is derived from HKDF; in production the hardware provides a
/// sealing key bound to the enclave measurement.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public abstract class TeeProviderBase<T> : FederatedLearningComponentBase<T>, ITeeProvider<T>
{
    private byte[] _sealingKey = Array.Empty<byte>();
    private string _measurementHash = string.Empty;
    private bool _initialized;
    private TeeOptions _options = new TeeOptions();

    /// <inheritdoc/>
    public abstract TeeProviderType ProviderType { get; }

    /// <inheritdoc/>
    public bool IsInitialized => _initialized;

    /// <summary>
    /// Gets the current TEE options.
    /// </summary>
    protected TeeOptions Options => _options;

    /// <inheritdoc/>
    public virtual void Initialize(TeeOptions options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;

        // Derive sealing key from enclave identity
        _sealingKey = DeriveSealingKey();
        _measurementHash = ComputeMeasurementHash();
        _initialized = true;
    }

    /// <inheritdoc/>
    public virtual void Destroy()
    {
        if (_sealingKey.Length > 0)
        {
            Array.Clear(_sealingKey, 0, _sealingKey.Length);
        }

        _sealingKey = Array.Empty<byte>();
        _measurementHash = string.Empty;
        _initialized = false;
    }

    /// <inheritdoc/>
    public byte[] SealData(byte[] plaintext)
    {
        EnsureInitialized();

        if (plaintext is null || plaintext.Length == 0)
        {
            throw new ArgumentException("Plaintext must not be null or empty.", nameof(plaintext));
        }

        return TeeAesHelper.Encrypt(_sealingKey, plaintext);
    }

    /// <inheritdoc/>
    public byte[] UnsealData(byte[] sealedData)
    {
        EnsureInitialized();

        if (sealedData is null || sealedData.Length < 17)
        {
            throw new ArgumentException("Sealed data is too short.", nameof(sealedData));
        }

        return TeeAesHelper.Decrypt(_sealingKey, sealedData);
    }

    /// <inheritdoc/>
    public virtual byte[] GenerateAttestationQuote(byte[] reportData)
    {
        EnsureInitialized();

        if (reportData is null)
        {
            throw new ArgumentNullException(nameof(reportData));
        }

        if (reportData.Length > 64)
        {
            throw new ArgumentException("Report data must be at most 64 bytes.", nameof(reportData));
        }

        return BuildQuote(reportData);
    }

    /// <inheritdoc/>
    public string GetMeasurementHash()
    {
        EnsureInitialized();
        return _measurementHash;
    }

    /// <inheritdoc/>
    public abstract long GetMaxEnclaveMemory();

    /// <summary>
    /// Derives the sealing key for this enclave. Hardware providers derive from the CPU;
    /// simulated providers use HKDF.
    /// </summary>
    protected abstract byte[] DeriveSealingKey();

    /// <summary>
    /// Computes the measurement hash (code identity) of this enclave.
    /// </summary>
    protected abstract string ComputeMeasurementHash();

    /// <summary>
    /// Builds a platform-specific attestation quote.
    /// </summary>
    /// <param name="reportData">Application data to bind into the quote.</param>
    /// <returns>Attestation quote bytes.</returns>
    protected abstract byte[] BuildQuote(byte[] reportData);

    /// <summary>
    /// Throws if the enclave has not been initialized.
    /// </summary>
    protected void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("TEE provider has not been initialized. Call Initialize() first.");
        }
    }
}
