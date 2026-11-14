namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// The rounds of the Bonawitz et al. (2017) secure aggregation protocol, in order.
/// </summary>
/// <remarks>
/// <para>
/// "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (CCS 2017), figure 4. The honest-but-curious
/// protocol runs rounds 0, 1, 2 and 4; <see cref="ConsistencyCheck"/> (round 3) belongs to the variant secure against
/// an actively malicious server and is not executed by this library.
/// </para>
/// <para><b>For Beginners:</b> Each value names the next message a participant is waiting to send or receive. A
/// participant that receives a message for the wrong round refuses it.</para>
/// </remarks>
public enum SecureAggregationRound
{
    /// <summary>Round 0: each client publishes two Diffie-Hellman public keys (<c>c_u^PK</c>, <c>s_u^PK</c>).</summary>
    AdvertiseKeys = 0,

    /// <summary>Round 1: each client Shamir-shares its self-mask seed <c>b_u</c> and masking key <c>s_u^SK</c>.</summary>
    ShareKeys = 1,

    /// <summary>Round 2: each client sends its masked input <c>y_u</c>.</summary>
    MaskedInputCollection = 2,

    /// <summary>Round 3: signature-based consistency check of the malicious-server variant (not executed).</summary>
    ConsistencyCheck = 3,

    /// <summary>Round 4: surviving clients send the shares the server needs to remove the masks.</summary>
    Unmasking = 4,

    /// <summary>The protocol instance has finished (or aborted) and holds no further secrets.</summary>
    Completed = 5,
}
