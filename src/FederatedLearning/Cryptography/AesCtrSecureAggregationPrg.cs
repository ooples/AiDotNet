using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// AES-256 in counter mode as the secure-aggregation pseudorandom generator.
/// </summary>
/// <remarks>
/// <para>
/// The 32-byte seed is the AES-256 key and the counter starts at zero. Each ring element takes the next 8 keystream
/// bytes as a little-endian 64-bit integer and keeps its low <c>b</c> bits. Because <c>R = 2^b</c> is a power of two,
/// keeping the low bits of a uniform 64-bit value is exactly uniform in <c>Z_R</c>: no rejection sampling is needed.
/// AES in counter mode is a standard PRG (NIST SP 800-90A CTR_DRBG is built on the same construction), and it is what
/// Bonawitz et al. (2017) name for the PRG in their prototype (section 7.1, alongside ECDH over NIST P-256).
/// </para>
/// <para><b>For Beginners:</b> This turns a 32-byte secret into as many random-looking numbers as the model has
/// parameters. Anyone with the same 32 bytes gets the same numbers; anyone without them sees noise.</para>
/// </remarks>
[ComponentType(ComponentType.CryptoPrimitive)]
[PipelineStage(PipelineStage.Training)]
public sealed class AesCtrSecureAggregationPrg : ISecureAggregationPrg
{
    /// <summary>The seed length this PRG requires: an AES-256 key.</summary>
    public const int SeedLength = 32;

    /// <inheritdoc />
    public void Expand(byte[] seed, int bitWidth, ulong[] destination)
    {
        if (seed is null)
        {
            throw new ArgumentNullException(nameof(seed));
        }

        if (seed.Length != SeedLength)
        {
            throw new ArgumentException($"The AES-CTR PRG requires a {SeedLength}-byte seed.", nameof(seed));
        }

        if (bitWidth < 1 || bitWidth > 64)
        {
            throw new ArgumentOutOfRangeException(nameof(bitWidth), "Bit width must be between 1 and 64.");
        }

        if (destination is null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (destination.Length == 0)
        {
            return;
        }

        ulong mask = bitWidth == 64 ? ulong.MaxValue : (1UL << bitWidth) - 1UL;
        var stream = new byte[checked(destination.Length * 8)];
        try
        {
            AesCtr.Keystream(seed, new byte[AesCtr.BlockSize], stream, stream.Length);
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = ReadUInt64LittleEndian(stream, i * 8) & mask;
            }
        }
        finally
        {
            Array.Clear(stream, 0, stream.Length);
        }
    }

    private static ulong ReadUInt64LittleEndian(byte[] buffer, int offset)
    {
        ulong value = 0;
        for (int i = 7; i >= 0; i--)
        {
            value = (value << 8) | buffer[offset + i];
        }

        return value;
    }
}
