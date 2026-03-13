using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// Software-simulated TEE provider for testing and development without hardware.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Real TEEs require specific hardware (Intel SGX, AMD SEV-SNP, etc.).
/// This simulated provider lets you develop and test TEE-based federated learning on any machine.
/// It provides the same API — enclave creation, sealing, attestation — using standard cryptography
/// instead of hardware isolation.</para>
///
/// <para><b>Important:</b> The simulated provider does NOT provide actual hardware security.
/// It is functionally equivalent (same encrypt/decrypt/attest flow) but offers no protection
/// against a compromised host. Use <see cref="TeeOptions.SimulationMode"/> = false in production.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class SimulatedTeeProvider<T> : TeeProviderBase<T>
{
    private string _simulatedMeasurement = string.Empty;
    private byte[] _enclaveIdentity = Array.Empty<byte>();

    /// <inheritdoc/>
    public override TeeProviderType ProviderType => TeeProviderType.Simulated;

    /// <inheritdoc/>
    public override void Initialize(TeeOptions options)
    {
        // Generate a fresh simulated enclave identity for this session.
        // Note: This identity is ephemeral — it changes on each Initialize() call.
        // Previously sealed blobs will not be unsealable after re-initialization.
        _enclaveIdentity = new byte[32];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(_enclaveIdentity);
        }

        base.Initialize(options);
    }

    /// <inheritdoc/>
    public override void Destroy()
    {
        if (_enclaveIdentity.Length > 0)
        {
            Array.Clear(_enclaveIdentity, 0, _enclaveIdentity.Length);
        }

        _enclaveIdentity = Array.Empty<byte>();
        _simulatedMeasurement = string.Empty;

        base.Destroy();
    }

    /// <inheritdoc/>
    public override long GetMaxEnclaveMemory()
    {
        // Simulated mode: use configured max or default 256 MB
        return (long)Options.MaxEnclaveMemoryMb * 1024 * 1024;
    }

    /// <inheritdoc/>
    protected override byte[] DeriveSealingKey()
    {
        // Derive a 256-bit key from the simulated enclave identity using HKDF
        var salt = System.Text.Encoding.UTF8.GetBytes("SimulatedTEE-SealingKey-v1");
        var info = System.Text.Encoding.UTF8.GetBytes("AES-256-GCM-Sealing");
        return HkdfSha256.DeriveKey(_enclaveIdentity, salt, info, 32);
    }

    /// <inheritdoc/>
    protected override string ComputeMeasurementHash()
    {
        // Simulate MRENCLAVE: SHA-256 of the enclave identity
        using var sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(_enclaveIdentity);
        _simulatedMeasurement = BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
        return _simulatedMeasurement;
    }

    /// <inheritdoc/>
    protected override byte[] BuildQuote(byte[] reportData)
    {
        // Build a simulated quote: version(2) + providerType(1) + timestamp(8)
        //   + measurementHash(32) + reportDataLen(4) + reportData + signature(32)
        using var sha256 = SHA256.Create();

        byte[] measurementBytes = sha256.ComputeHash(_enclaveIdentity);
        long timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        int quoteLength = 2 + 1 + 8 + 32 + 4 + reportData.Length + 32;
        var quote = new byte[quoteLength];
        int offset = 0;

        // Version
        quote[offset++] = 0x01;
        quote[offset++] = 0x00;

        // Provider type
        quote[offset++] = (byte)TeeProviderType.Simulated;

        // Timestamp (8 bytes, little-endian)
        var timestampBytes = BitConverter.GetBytes(timestamp);
        Buffer.BlockCopy(timestampBytes, 0, quote, offset, 8);
        offset += 8;

        // Measurement hash (32 bytes)
        Buffer.BlockCopy(measurementBytes, 0, quote, offset, 32);
        offset += 32;

        // Report data length + data
        var reportLenBytes = BitConverter.GetBytes(reportData.Length);
        Buffer.BlockCopy(reportLenBytes, 0, quote, offset, 4);
        offset += 4;
        Buffer.BlockCopy(reportData, 0, quote, offset, reportData.Length);
        offset += reportData.Length;

        // Simulated signature: HMAC-SHA256(enclaveIdentity, quote[0..offset])
        using var hmac = new HMACSHA256(_enclaveIdentity);
        byte[] sig = hmac.ComputeHash(quote, 0, offset);
        Buffer.BlockCopy(sig, 0, quote, offset, 32);

        return quote;
    }
}
