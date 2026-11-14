namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Round 2 (MaskedInputCollection) message: a client's masked input <c>y_u</c>, a vector in <c>Z_R^m</c>.
/// </summary>
/// <remarks>
/// <para>
/// Bonawitz et al. (2017), figure 4, round 2: <c>y_u = x_u + PRG(b_u) + sum_{v in U_2 \ {u}} p_{u,v} (mod R)</c>, with
/// <c>p_{u,v} = +PRG(s_{u,v})</c> when <c>u &lt; v</c> and <c>-PRG(s_{u,v})</c> when <c>u &gt; v</c>. Each element is a
/// ring element stored in the low <see cref="FixedPointRingEncoding.BitWidth"/> bits of a <see cref="ulong"/>.
/// </para>
/// <para><b>For Beginners:</b> This vector looks uniformly random to the server; only the sum over all surviving
/// clients, after unmasking, means anything.</para>
/// </remarks>
public sealed class SecureAggregationMaskedInput
{
    /// <summary>Creates the message.</summary>
    public SecureAggregationMaskedInput(int clientId, ulong[] values)
    {
        ClientId = clientId;
        Values = values ?? throw new ArgumentNullException(nameof(values));
    }

    /// <summary>Gets the sending client <c>u</c>.</summary>
    public int ClientId { get; }

    /// <summary>Gets <c>y_u</c>, one ring element per model parameter.</summary>
    public ulong[] Values { get; }
}
