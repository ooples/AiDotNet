using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Cryptography;

namespace AiDotNet.FederatedLearning.MPC;

/// <summary>
/// Implements base 1-out-of-2 oblivious transfer using symmetric cryptography.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In oblivious transfer (OT), a sender has two messages (m0, m1)
/// and a receiver has a choice bit (0 or 1). After the protocol:</para>
/// <list type="bullet">
/// <item><description>The receiver learns m_choice but NOT the other message.</description></item>
/// <item><description>The sender learns nothing about which message was chosen.</description></item>
/// </list>
///
/// <para>This implementation uses a simplified random-oracle model where the sender and
/// receiver derive keys from a shared random seed (simulating the Diffie-Hellman key
/// exchange that would happen in a real network protocol). In production, this would use
/// actual public-key cryptography over a network channel.</para>
///
/// <para><b>Performance:</b> Each base OT requires public-key operations. Use
/// <see cref="ExtendedObliviousTransfer"/> to amortize this cost for many transfers.</para>
/// </remarks>
public class BaseObliviousTransfer : IObliviousTransfer
{
    private int _baseTransferCount;

    /// <inheritdoc/>
    public int BaseTransferCount => _baseTransferCount;

    /// <summary>
    /// Initializes a new instance of <see cref="BaseObliviousTransfer"/>.
    /// </summary>
    public BaseObliviousTransfer()
    {
        _baseTransferCount = 0;
    }

    /// <inheritdoc/>
    public byte[] Transfer(byte[] message0, byte[] message1, int choiceBit)
    {
        if (message0 is null || message1 is null)
        {
            throw new ArgumentNullException(message0 is null ? nameof(message0) : nameof(message1));
        }

        if (choiceBit != 0 && choiceBit != 1)
        {
            throw new ArgumentOutOfRangeException(nameof(choiceBit), "Choice bit must be 0 or 1.");
        }

        if (message0.Length != message1.Length)
        {
            throw new ArgumentException("Both messages must have the same length.");
        }

        _baseTransferCount++;

        // Simulate OT using random oracle model:
        // 1. Generate random nonce for this transfer
        // 2. Derive two keys from the nonce (one for each message)
        // 3. Encrypt both messages under their respective keys
        // 4. The receiver can only derive the key for its chosen message

        var nonce = new byte[32];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(nonce);
        }

        // Derive keys for message 0 and message 1
        var salt = new byte[] { 0x4F, 0x54, 0x42, 0x41, 0x53, 0x45 }; // "OTBASE"
        var key0 = HkdfSha256.DeriveKey(nonce, salt, new byte[] { 0 }, message0.Length);
        var key1 = HkdfSha256.DeriveKey(nonce, salt, new byte[] { 1 }, message1.Length);

        // Encrypt both messages
        var encrypted0 = XorEncrypt(message0, key0);
        var encrypted1 = XorEncrypt(message1, key1);

        // The receiver decrypts only the chosen message
        byte[] chosenKey = choiceBit == 0 ? key0 : key1;
        byte[] chosenCiphertext = choiceBit == 0 ? encrypted0 : encrypted1;

        // Clear the key we don't need (simulates the receiver not having access to it)
        if (choiceBit == 0)
        {
            Array.Clear(key1, 0, key1.Length);
        }
        else
        {
            Array.Clear(key0, 0, key0.Length);
        }

        return XorEncrypt(chosenCiphertext, chosenKey);
    }

    /// <inheritdoc/>
    public byte[][] BatchTransfer(byte[][] messages0, byte[][] messages1, int[] choiceBits)
    {
        if (messages0 is null || messages1 is null || choiceBits is null)
        {
            throw new ArgumentNullException(
                messages0 is null ? nameof(messages0) :
                messages1 is null ? nameof(messages1) : nameof(choiceBits));
        }

        if (messages0.Length != messages1.Length || messages0.Length != choiceBits.Length)
        {
            throw new ArgumentException("All arrays must have the same length.");
        }

        var results = new byte[messages0.Length][];
        for (int i = 0; i < messages0.Length; i++)
        {
            results[i] = Transfer(messages0[i], messages1[i], choiceBits[i]);
        }

        return results;
    }

    private static byte[] XorEncrypt(byte[] data, byte[] key)
    {
        var result = new byte[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            result[i] = (byte)(data[i] ^ key[i % key.Length]);
        }

        return result;
    }
}
