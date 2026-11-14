namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Round 0 (AdvertiseKeys) message: a client's two P-256 public keys.
/// </summary>
/// <remarks>
/// <para>
/// Bonawitz et al. (2017), figure 4, round 0: client <c>u</c> generates <c>(c_u^PK, c_u^SK)</c> for encrypting shares and
/// <c>(s_u^PK, s_u^SK)</c> for deriving pairwise masks, and sends both public keys to the server, which broadcasts the
/// list to all clients in <c>U_1</c>.
/// </para>
/// <para><b>For Beginners:</b> Public keys can be shown to everyone. Each pair of clients combines them with their own
/// private keys to agree on shared secrets without ever sending those secrets.</para>
/// </remarks>
public sealed class SecureAggregationAdvertisedKeys
{
    /// <summary>Creates the message.</summary>
    public SecureAggregationAdvertisedKeys(int clientId, byte[] cipherPublicKey, byte[] maskingPublicKey)
    {
        ClientId = clientId;
        CipherPublicKey = cipherPublicKey ?? throw new ArgumentNullException(nameof(cipherPublicKey));
        MaskingPublicKey = maskingPublicKey ?? throw new ArgumentNullException(nameof(maskingPublicKey));
    }

    /// <summary>Gets the sending client's identifier <c>u</c>.</summary>
    public int ClientId { get; }

    /// <summary>Gets <c>c_u^PK</c>, the 64-byte (X || Y) P-256 key used to encrypt shares.</summary>
    public byte[] CipherPublicKey { get; }

    /// <summary>Gets <c>s_u^PK</c>, the 64-byte (X || Y) P-256 key used to derive pairwise masks.</summary>
    public byte[] MaskingPublicKey { get; }
}
