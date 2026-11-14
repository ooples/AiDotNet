namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// A cryptographic pseudorandom generator (PRG) that expands a short seed into a vector of ring elements, as used by
/// secure aggregation to turn the pairwise seeds <c>s_{u,v}</c> and the self-mask seed <c>b_u</c> into masks.
/// </summary>
/// <remarks>
/// <para>
/// Bonawitz et al. (2017), "Practical Secure Aggregation for Privacy-Preserving Machine Learning", section 4: the
/// masks are <c>PRG(s_{u,v})</c> and <c>PRG(b_u)</c>, vectors in <c>Z_R^m</c>. An implementation must be a secure PRG
/// (indistinguishable from uniform without the seed) and deterministic: the same seed and bit width must always produce
/// the same vector, because the server re-expands reconstructed seeds to remove masks.
/// </para>
/// <para><b>For Beginners:</b> Two clients that share a secret seed can each compute the same long random-looking
/// vector from it without sending that vector anywhere. One adds it, the other subtracts it, so it cancels in the sum.
/// The library default is <see cref="AesCtrSecureAggregationPrg"/>.</para>
/// </remarks>
public interface ISecureAggregationPrg
{
    /// <summary>
    /// Fills <paramref name="destination"/> with elements of <c>Z_R</c>, <c>R = 2^bitWidth</c>, expanded from
    /// <paramref name="seed"/>. Every element must be uniformly distributed in <c>[0, 2^bitWidth)</c>.
    /// </summary>
    /// <param name="seed">The secret seed (32 bytes for the library implementations).</param>
    /// <param name="bitWidth">The ring bit width <c>b</c>, between 1 and 64.</param>
    /// <param name="destination">The vector to fill; its length is the number of ring elements generated.</param>
    void Expand(byte[] seed, int bitWidth, ulong[] destination);
}
