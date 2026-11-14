namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Round 1 (ShareKeys) message: client <c>u</c>'s shares for client <c>v</c>, encrypted so only <c>v</c> can read them.
/// </summary>
/// <remarks>
/// <para>
/// Bonawitz et al. (2017), figure 4, round 1: <c>e_{u,v} = AE.enc(KA.agree(c_u^SK, c_v^PK), u || v || s_{u,v}^SK ||
/// b_{u,v})</c>. The server only routes these ciphertexts; it cannot decrypt them.
/// </para>
/// <para><b>For Beginners:</b> This is a sealed envelope from one client to another, carried by the server.</para>
/// </remarks>
public sealed class SecureAggregationEncryptedShare
{
    /// <summary>Creates the message.</summary>
    public SecureAggregationEncryptedShare(int senderId, int recipientId, byte[] ciphertext)
    {
        SenderId = senderId;
        RecipientId = recipientId;
        Ciphertext = ciphertext ?? throw new ArgumentNullException(nameof(ciphertext));
    }

    /// <summary>Gets the sender <c>u</c>.</summary>
    public int SenderId { get; }

    /// <summary>Gets the recipient <c>v</c>.</summary>
    public int RecipientId { get; }

    /// <summary>Gets the authenticated ciphertext <c>e_{u,v}</c>.</summary>
    public byte[] Ciphertext { get; }
}
