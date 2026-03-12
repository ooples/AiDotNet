using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// TEE provider for ARM CCA (Confidential Compute Architecture) / Realm Management Extension.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> ARM CCA introduces "Realms" — isolated execution environments
/// on ARM processors (Armv9+). A Realm is like an enclave that runs a full OS or application,
/// protected from the hypervisor and other Realms. ARM CCA is important for edge/IoT federated
/// learning because many edge devices use ARM processors.</para>
///
/// <para><b>Key concepts:</b></para>
/// <list type="bullet">
/// <item><description><b>Realm:</b> An isolated execution environment managed by the Realm Management Monitor (RMM).</description></item>
/// <item><description><b>RIM:</b> Realm Initial Measurement — hash of the Realm's initial state (analogous to MRENCLAVE).</description></item>
/// <item><description><b>REM:</b> Realm Extensible Measurement — hash chain of runtime measurements.</description></item>
/// <item><description><b>CCA attestation:</b> Platform token signed by the CCA platform, Realm token signed by the Realm.</description></item>
/// </list>
///
/// <para><b>Target hardware:</b> Armv9-A with RME (Realm Management Extension), available in
/// ARM Neoverse V2+ and Cortex-X4+ cores.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class ArmCcaTeeProvider<T> : TeeProviderBase<T>
{
    /// <summary>
    /// ARM CCA default max Realm memory: 8 GB for edge/cloud workloads.
    /// </summary>
    private const long CcaDefaultMaxBytes = 8L * 1024 * 1024 * 1024;

    private byte[] _realmInitialMeasurement = Array.Empty<byte>(); // RIM
    private byte[] _realmExtensibleMeasurement = Array.Empty<byte>(); // REM
    private byte[] _platformToken = Array.Empty<byte>();

    /// <inheritdoc/>
    public override TeeProviderType ProviderType => TeeProviderType.ArmCca;

    /// <inheritdoc/>
    public override void Initialize(TeeOptions options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        // Generate RIM (Realm Initial Measurement) and REM (Realm Extensible Measurement)
        _realmInitialMeasurement = new byte[32];
        _realmExtensibleMeasurement = new byte[32];
        _platformToken = new byte[32];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(_realmInitialMeasurement);
            rng.GetBytes(_realmExtensibleMeasurement);
            rng.GetBytes(_platformToken);
        }

        base.Initialize(options);
    }

    /// <inheritdoc/>
    public override void Destroy()
    {
        if (_realmInitialMeasurement.Length > 0) Array.Clear(_realmInitialMeasurement, 0, _realmInitialMeasurement.Length);
        if (_realmExtensibleMeasurement.Length > 0) Array.Clear(_realmExtensibleMeasurement, 0, _realmExtensibleMeasurement.Length);
        if (_platformToken.Length > 0) Array.Clear(_platformToken, 0, _platformToken.Length);

        _realmInitialMeasurement = Array.Empty<byte>();
        _realmExtensibleMeasurement = Array.Empty<byte>();
        _platformToken = Array.Empty<byte>();

        base.Destroy();
    }

    /// <inheritdoc/>
    public override long GetMaxEnclaveMemory()
    {
        long requested = (long)Options.MaxEnclaveMemoryMb * 1024 * 1024;
        return Math.Min(requested, CcaDefaultMaxBytes);
    }

    /// <inheritdoc/>
    protected override byte[] DeriveSealingKey()
    {
        // ARM CCA: sealing key from RIM (Realm Initial Measurement)
        var salt = System.Text.Encoding.UTF8.GetBytes("ArmCCA-SealingKey-RIM-v1");
        var info = System.Text.Encoding.UTF8.GetBytes("AES-256-GCM-Sealing");
        return HkdfSha256.DeriveKey(_realmInitialMeasurement, salt, info, 32);
    }

    /// <inheritdoc/>
    protected override string ComputeMeasurementHash()
    {
        return BitConverter.ToString(_realmInitialMeasurement).Replace("-", "").ToLowerInvariant();
    }

    /// <inheritdoc/>
    protected override byte[] BuildQuote(byte[] reportData)
    {
        // ARM CCA attestation format (simplified):
        // version(2) + providerType(1) + timestamp(8) + RIM(32) + REM(32) + platformToken(32)
        //   + reportDataLen(4) + reportData + signature(64)
        long timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        int quoteLength = 2 + 1 + 8 + 32 + 32 + 32 + 4 + reportData.Length + 64;
        var quote = new byte[quoteLength];
        int offset = 0;

        // Version
        quote[offset++] = 0x01;
        quote[offset++] = 0x00;

        // Provider type
        quote[offset++] = (byte)TeeProviderType.ArmCca;

        // Timestamp
        var timestampBytes = BitConverter.GetBytes(timestamp);
        Buffer.BlockCopy(timestampBytes, 0, quote, offset, 8);
        offset += 8;

        // RIM (32 bytes)
        Buffer.BlockCopy(_realmInitialMeasurement, 0, quote, offset, 32);
        offset += 32;

        // REM (32 bytes)
        Buffer.BlockCopy(_realmExtensibleMeasurement, 0, quote, offset, 32);
        offset += 32;

        // Platform token (32 bytes)
        Buffer.BlockCopy(_platformToken, 0, quote, offset, 32);
        offset += 32;

        // Report data
        var reportLenBytes = BitConverter.GetBytes(reportData.Length);
        Buffer.BlockCopy(reportLenBytes, 0, quote, offset, 4);
        offset += 4;
        Buffer.BlockCopy(reportData, 0, quote, offset, reportData.Length);
        offset += reportData.Length;

        // Simulated Realm token signature
        using var hmac = new HMACSHA256(_realmInitialMeasurement);
        byte[] sig1 = hmac.ComputeHash(quote, 0, offset);
        Buffer.BlockCopy(sig1, 0, quote, offset, 32);
        offset += 32;

        // Platform token signature
        using var hmac2 = new HMACSHA256(_platformToken);
        byte[] sig2 = hmac2.ComputeHash(quote, 0, offset - 32);
        Buffer.BlockCopy(sig2, 0, quote, offset, 32);

        return quote;
    }
}
