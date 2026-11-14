using System.Security.Cryptography;

namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// AES in counter (CTR) mode, built from the AES block cipher in ECB mode so it is available on every target
/// framework (.NET Framework 4.6.2 and .NET Standard 2.0 have neither <c>AesGcm</c> nor a CTR cipher mode).
/// </summary>
/// <remarks>
/// <para>
/// The keystream is <c>AES_K(IV) || AES_K(IV + 1) || ...</c> with the 128-bit counter block incremented as a
/// big-endian integer (NIST SP 800-38A, appendix B.1). XOR with the keystream both encrypts and decrypts.
/// </para>
/// <para><b>For Beginners:</b> AES-CTR turns a block cipher into a stream of random-looking bytes that depends only
/// on the key and the starting counter. It is used here both as the pseudorandom generator that expands mask seeds and
/// as the encryption layer for secret shares.</para>
/// </remarks>
internal static class AesCtr
{
    internal const int BlockSize = 16;

    // Blocks encrypted per ECB call: 4096 * 16 = 64 KiB of keystream at a time.
    private const int BlocksPerBatch = 4096;

    /// <summary>
    /// Writes <paramref name="count"/> bytes of AES-CTR keystream for <paramref name="key"/> starting at the
    /// counter block <paramref name="initialCounter"/> into <paramref name="destination"/>.
    /// </summary>
    internal static void Keystream(byte[] key, byte[] initialCounter, byte[] destination, int count)
    {
        if (key is null)
        {
            throw new ArgumentNullException(nameof(key));
        }

        if (key.Length != 16 && key.Length != 24 && key.Length != 32)
        {
            throw new ArgumentException("AES keys must be 16, 24 or 32 bytes long.", nameof(key));
        }

        if (initialCounter is null)
        {
            throw new ArgumentNullException(nameof(initialCounter));
        }

        if (initialCounter.Length != BlockSize)
        {
            throw new ArgumentException("The initial counter block must be 16 bytes long.", nameof(initialCounter));
        }

        if (destination is null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (count < 0 || count > destination.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "Count must be between 0 and the destination length.");
        }

        if (count == 0)
        {
            return;
        }

        var counter = (byte[])initialCounter.Clone();
        int totalBlocks = (count + BlockSize - 1) / BlockSize;
        int batchBlocks = Math.Min(BlocksPerBatch, totalBlocks);
        var counterBatch = new byte[batchBlocks * BlockSize];
        var keystreamBatch = new byte[batchBlocks * BlockSize];

        using var aes = Aes.Create();
        aes.Mode = CipherMode.ECB;
        aes.Padding = PaddingMode.None;
        aes.Key = key;
        using var encryptor = aes.CreateEncryptor();

        int written = 0;
        int remainingBlocks = totalBlocks;
        try
        {
            while (remainingBlocks > 0)
            {
                int blocks = Math.Min(batchBlocks, remainingBlocks);
                for (int b = 0; b < blocks; b++)
                {
                    Buffer.BlockCopy(counter, 0, counterBatch, b * BlockSize, BlockSize);
                    IncrementBigEndian(counter);
                }

                int bytes = blocks * BlockSize;
                encryptor.TransformBlock(counterBatch, 0, bytes, keystreamBatch, 0);
                int toCopy = Math.Min(bytes, count - written);
                Buffer.BlockCopy(keystreamBatch, 0, destination, written, toCopy);
                written += toCopy;
                remainingBlocks -= blocks;
            }
        }
        finally
        {
            Array.Clear(keystreamBatch, 0, keystreamBatch.Length);
            Array.Clear(counterBatch, 0, counterBatch.Length);
            Array.Clear(counter, 0, counter.Length);
        }
    }

    /// <summary>XORs <paramref name="data"/> in place with the AES-CTR keystream (encrypts or decrypts).</summary>
    internal static void Transform(byte[] key, byte[] initialCounter, byte[] data)
    {
        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var stream = new byte[data.Length];
        try
        {
            Keystream(key, initialCounter, stream, stream.Length);
            for (int i = 0; i < data.Length; i++)
            {
                data[i] ^= stream[i];
            }
        }
        finally
        {
            Array.Clear(stream, 0, stream.Length);
        }
    }

    private static void IncrementBigEndian(byte[] counter)
    {
        for (int i = counter.Length - 1; i >= 0; i--)
        {
            unchecked
            {
                counter[i]++;
            }

            if (counter[i] != 0)
            {
                return;
            }
        }
    }
}
