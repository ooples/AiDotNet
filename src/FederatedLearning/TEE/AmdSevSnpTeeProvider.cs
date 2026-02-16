using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// TEE provider for AMD SEV-SNP (Secure Encrypted Virtualization - Secure Nested Paging).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> AMD SEV-SNP encrypts all of a virtual machine's memory using a
/// hardware key that the hypervisor cannot access. "SNP" adds integrity protection — if anyone
/// tries to tamper with encrypted memory, the CPU detects it. This makes it possible to run
/// federated learning aggregation in a VM that even the cloud provider's host OS cannot inspect.</para>
///
/// <para><b>Key concepts:</b></para>
/// <list type="bullet">
/// <item><description><b>VMPL:</b> Virtual Machine Privilege Levels — SNP supports 4 levels within a VM.</description></item>
/// <item><description><b>Launch Digest:</b> Hash of the initial VM image (analogous to SGX MRENCLAVE).</description></item>
/// <item><description><b>VCEK:</b> Versioned Chip Endorsement Key — AMD's per-chip attestation key.</description></item>
/// <item><description><b>Memory encryption:</b> AES-128 per-VM key managed by the AMD Secure Processor.</description></item>
/// </list>
///
/// <para><b>Cloud availability:</b> Azure DCasv5/ECasv5 VMs, Google Cloud C2D, AWS (forthcoming).</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class AmdSevSnpTeeProvider<T> : TeeProviderBase<T>
{
    /// <summary>
    /// SEV-SNP supports full VM memory, default 32 GB for FL workloads.
    /// </summary>
    private const long SevSnpDefaultMaxBytes = 32L * 1024 * 1024 * 1024;

    private byte[] _launchDigest = Array.Empty<byte>(); // VM measurement (analogous to MRENCLAVE)
    private byte[] _vcekIdentity = Array.Empty<byte>(); // Chip endorsement key identity

    /// <inheritdoc/>
    public override TeeProviderType ProviderType => TeeProviderType.SevSnp;

    /// <inheritdoc/>
    public override void Initialize(TeeOptions options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        // Generate launch digest and VCEK identity
        _launchDigest = new byte[48]; // SEV-SNP uses 384-bit measurements
        _vcekIdentity = new byte[32];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(_launchDigest);
            rng.GetBytes(_vcekIdentity);
        }

        base.Initialize(options);
    }

    /// <inheritdoc/>
    public override void Destroy()
    {
        if (_launchDigest.Length > 0) Array.Clear(_launchDigest, 0, _launchDigest.Length);
        if (_vcekIdentity.Length > 0) Array.Clear(_vcekIdentity, 0, _vcekIdentity.Length);

        _launchDigest = Array.Empty<byte>();
        _vcekIdentity = Array.Empty<byte>();

        base.Destroy();
    }

    /// <inheritdoc/>
    public override long GetMaxEnclaveMemory()
    {
        long requested = (long)Options.MaxEnclaveMemoryMb * 1024 * 1024;
        return Math.Min(requested, SevSnpDefaultMaxBytes);
    }

    /// <inheritdoc/>
    protected override byte[] DeriveSealingKey()
    {
        // SEV-SNP: sealing key from launch digest (VM identity)
        var salt = System.Text.Encoding.UTF8.GetBytes("AmdSEV-SNP-SealingKey-LaunchDigest-v1");
        var info = System.Text.Encoding.UTF8.GetBytes("AES-256-GCM-Sealing");
        return HkdfSha256.DeriveKey(_launchDigest, salt, info, 32);
    }

    /// <inheritdoc/>
    protected override string ComputeMeasurementHash()
    {
        return BitConverter.ToString(_launchDigest).Replace("-", "").ToLowerInvariant();
    }

    /// <inheritdoc/>
    protected override byte[] BuildQuote(byte[] reportData)
    {
        // SEV-SNP attestation report format (simplified):
        // version(2) + providerType(1) + timestamp(8) + launchDigest(48) + vcekId(32)
        //   + reportDataLen(4) + reportData + signature(64)
        long timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        int quoteLength = 2 + 1 + 8 + 48 + 32 + 4 + reportData.Length + 64;
        var quote = new byte[quoteLength];
        int offset = 0;

        // Version
        quote[offset++] = 0x02;
        quote[offset++] = 0x00;

        // Provider type
        quote[offset++] = (byte)TeeProviderType.SevSnp;

        // Timestamp
        var timestampBytes = BitConverter.GetBytes(timestamp);
        Buffer.BlockCopy(timestampBytes, 0, quote, offset, 8);
        offset += 8;

        // Launch digest (48 bytes)
        Buffer.BlockCopy(_launchDigest, 0, quote, offset, 48);
        offset += 48;

        // VCEK identity (32 bytes)
        Buffer.BlockCopy(_vcekIdentity, 0, quote, offset, 32);
        offset += 32;

        // Report data
        var reportLenBytes = BitConverter.GetBytes(reportData.Length);
        Buffer.BlockCopy(reportLenBytes, 0, quote, offset, 4);
        offset += 4;
        Buffer.BlockCopy(reportData, 0, quote, offset, reportData.Length);
        offset += reportData.Length;

        // Simulated VCEK signature
        using var hmac = new HMACSHA256(_vcekIdentity);
        byte[] sig1 = hmac.ComputeHash(quote, 0, offset);
        Buffer.BlockCopy(sig1, 0, quote, offset, 32);
        offset += 32;

        using var hmac2 = new HMACSHA256(_launchDigest);
        byte[] sig2 = hmac2.ComputeHash(quote, 0, offset - 32);
        Buffer.BlockCopy(sig2, 0, quote, offset, 32);

        return quote;
    }
}
