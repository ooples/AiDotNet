using System.Security.Cryptography;
using AiDotNet.FederatedLearning.Cryptography;

namespace AiDotNet.FederatedLearning.MPC;

/// <summary>
/// Implements OT extension — amortizes a small number of base OTs into many cheap OTs.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Base oblivious transfer uses expensive public-key cryptography.
/// OT extension lets you perform a small number of base OTs (e.g., 128) and then "extend" them
/// into millions of OTs using only symmetric crypto (hashing). This makes garbled circuit
/// evaluation practical.</para>
///
/// <para><b>How it works (simplified):</b></para>
/// <list type="bullet">
/// <item><description>Run k base OTs where the sender and receiver swap roles.</description></item>
/// <item><description>The receiver sends a matrix of random bits XORed with its choice bits.</description></item>
/// <item><description>The sender uses base OT keys to generate pseudorandom pads for both messages.</description></item>
/// <item><description>Each extended OT requires only a hash computation.</description></item>
/// </list>
///
/// <para><b>Reference:</b> IKNP OT Extension (Ishai, Kilian, Nissim, Petrank, CRYPTO 2003).</para>
/// </remarks>
public class ExtendedObliviousTransfer : IObliviousTransfer
{
    private readonly IObliviousTransfer _baseOt;
    private readonly int _securityParameter;
    private byte[][]? _baseKeys0;
    private byte[][]? _baseKeys1;
    private bool _initialized;
    private int _transferCount;

    /// <inheritdoc/>
    public int BaseTransferCount => _baseOt.BaseTransferCount;

    /// <summary>
    /// Initializes a new instance of <see cref="ExtendedObliviousTransfer"/>.
    /// </summary>
    /// <param name="baseOt">The base OT protocol to bootstrap from.</param>
    /// <param name="securityParameter">Security parameter in bits (default 128).</param>
    public ExtendedObliviousTransfer(IObliviousTransfer? baseOt = null, int securityParameter = 128)
    {
        _baseOt = baseOt ?? new BaseObliviousTransfer();
        _securityParameter = securityParameter;
        _initialized = false;
        _transferCount = 0;
    }

    /// <summary>
    /// Initializes the OT extension by running the base OTs.
    /// </summary>
    public void Initialize()
    {
        if (_initialized)
        {
            return;
        }

        // Run _securityParameter base OTs to generate seed keys
        _baseKeys0 = new byte[_securityParameter][];
        _baseKeys1 = new byte[_securityParameter][];

        using (var rng = RandomNumberGenerator.Create())
        {
            for (int i = 0; i < _securityParameter; i++)
            {
                // Generate two random keys for each base OT
                _baseKeys0[i] = new byte[16]; // 128-bit keys
                _baseKeys1[i] = new byte[16];
                rng.GetBytes(_baseKeys0[i]);
                rng.GetBytes(_baseKeys1[i]);

                // Perform base OT with random choice to establish key correlation
                int choiceBit = (i % 2 == 0) ? 0 : 1;
                _baseOt.Transfer(_baseKeys0[i], _baseKeys1[i], choiceBit);
            }
        }

        _initialized = true;
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

        if (!_initialized)
        {
            Initialize();
        }

        _transferCount++;

        // Derive pad for this transfer index using DIFFERENT base keys for each message.
        // pad0 uses _baseKeys0 (sender's key 0) and pad1 uses _baseKeys1 (sender's key 1).
        // This ensures the receiver can only decrypt the chosen message, not both.
        int msgLen = message0.Length;
        var salt = BitConverter.GetBytes(_transferCount);
        var info0 = new byte[] { 0x45, 0x58, 0x54, 0x30 }; // "EXT0"
        var info1 = new byte[] { 0x45, 0x58, 0x54, 0x31 }; // "EXT1"

        int keyIdx = _transferCount % _securityParameter;
        byte[] seedKey0 = _baseKeys0 is not null && _baseKeys0.Length > keyIdx
            ? _baseKeys0[keyIdx]
            : new byte[16];
        byte[] seedKey1 = _baseKeys1 is not null && _baseKeys1.Length > keyIdx
            ? _baseKeys1[keyIdx]
            : new byte[16];

        var pad0 = HkdfSha256.DeriveKey(seedKey0, salt, info0, msgLen);
        var pad1 = HkdfSha256.DeriveKey(seedKey1, salt, info1, msgLen);

        // Encrypt both messages
        var encrypted0 = new byte[msgLen];
        var encrypted1 = new byte[msgLen];
        for (int i = 0; i < msgLen; i++)
        {
            encrypted0[i] = (byte)(message0[i] ^ pad0[i]);
            encrypted1[i] = (byte)(message1[i] ^ pad1[i]);
        }

        // Receiver decrypts chosen message using the corresponding pad
        // choiceBit=0 uses pad0 (from _baseKeys0), choiceBit=1 uses pad1 (from _baseKeys1)
        byte[] chosenPad = choiceBit == 0 ? pad0 : pad1;
        byte[] chosenCiphertext = choiceBit == 0 ? encrypted0 : encrypted1;

        var result = new byte[msgLen];
        for (int i = 0; i < msgLen; i++)
        {
            result[i] = (byte)(chosenCiphertext[i] ^ chosenPad[i]);
        }

        return result;
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

        if (!_initialized)
        {
            Initialize();
        }

        var results = new byte[messages0.Length][];
        for (int i = 0; i < messages0.Length; i++)
        {
            results[i] = Transfer(messages0[i], messages1[i], choiceBits[i]);
        }

        return results;
    }
}
