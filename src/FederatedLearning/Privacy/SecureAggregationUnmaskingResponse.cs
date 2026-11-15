namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Round 4 (Unmasking) message: the shares a surviving client releases so the server can remove the masks.
/// </summary>
/// <remarks>
/// <para>
/// Bonawitz et al. (2017), figure 4, round 4: client <c>u</c> sends <c>s_{v,u}^SK</c> for every <c>v</c> in
/// <c>U_2 \ U_3</c> (clients that dropped before sending a masked input, whose pairwise masks must be recomputed) and
/// <c>b_{v,u}</c> for every <c>v</c> in <c>U_3</c> (clients whose self masks must be removed). An honest client never
/// releases both kinds of share for the same <c>v</c>, so the server can never unmask an individual input.
/// </para>
/// <para><b>For Beginners:</b> Each surviving client hands over its pieces of other clients' secrets, but only the
/// pieces needed to cancel masks, never enough to expose anyone's update.</para>
/// </remarks>
public sealed class SecureAggregationUnmaskingResponse
{
    /// <summary>Creates the message.</summary>
    public SecureAggregationUnmaskingResponse(
        int clientId,
        IReadOnlyDictionary<int, byte[]> maskingKeyShares,
        IReadOnlyDictionary<int, byte[]> selfMaskShares)
    {
        ClientId = clientId;
        MaskingKeyShares = maskingKeyShares ?? throw new ArgumentNullException(nameof(maskingKeyShares));
        SelfMaskShares = selfMaskShares ?? throw new ArgumentNullException(nameof(selfMaskShares));
    }

    /// <summary>Gets the sending client <c>u</c>.</summary>
    public int ClientId { get; }

    /// <summary>Gets <c>s_{v,u}^SK</c> keyed by the dropped client <c>v</c>.</summary>
    public IReadOnlyDictionary<int, byte[]> MaskingKeyShares { get; }

    /// <summary>Gets <c>b_{v,u}</c> keyed by the surviving client <c>v</c>.</summary>
    public IReadOnlyDictionary<int, byte[]> SelfMaskShares { get; }
}
