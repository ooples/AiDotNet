using Newtonsoft.Json;

namespace AiDotNet.Helpers;

/// <summary>
/// The signed claim set carried by an <c>aidn2.</c> asymmetric license token.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A license token is a small block of JSON (these fields) plus a
/// cryptographic signature over that exact JSON. Because the signature can only be produced with the
/// server's private key, nobody can change any field (e.g. bump <c>exp</c> or upgrade <c>tier</c>)
/// without invalidating the signature.</para>
/// <para>Field names are intentionally short (JWT-style) to keep tokens compact.</para>
/// </remarks>
internal sealed class LicenseClaims
{
    /// <summary>Subject — the customer / license the token was issued to.</summary>
    [JsonProperty("sub")]
    public string? Sub { get; set; }

    /// <summary>Tier: <c>community</c>, <c>pro</c>, or <c>enterprise</c>.</summary>
    [JsonProperty("tier")]
    public string? Tier { get; set; }

    /// <summary>Number of licensed seats.</summary>
    [JsonProperty("seats")]
    public int Seats { get; set; }

    /// <summary>Issued-at, Unix seconds (UTC).</summary>
    [JsonProperty("iat")]
    public long Iat { get; set; }

    /// <summary>Expiry, Unix seconds (UTC). Bounds offline use — a leaked token self-expires.</summary>
    [JsonProperty("exp")]
    public long Exp { get; set; }

    /// <summary>Key id — selects which embedded public key verifies this token (rotation).</summary>
    [JsonProperty("kid")]
    public string? Kid { get; set; }

    /// <summary>
    /// Signature algorithm identifier. Always <c>"EdDSA"</c> for aidn2 tokens (Ed25519). Recorded in the
    /// claims so a future primitive change / rotation is unambiguous — the verifier rejects any token
    /// whose <c>alg</c> it does not implement rather than silently assuming Ed25519.
    /// </summary>
    [JsonProperty("alg")]
    public string? Alg { get; set; }

    /// <summary>
    /// Serializes the claims to the canonical compact JSON that gets signed. Callers MUST sign/verify
    /// over the exact bytes that appear in the token's middle segment; this helper is used by the
    /// (server/test) signer. The verifier verifies over the raw decoded segment bytes as received.
    /// </summary>
    internal string ToCanonicalJson() => JsonConvert.SerializeObject(this, Formatting.None);

    /// <summary>
    /// Parses claims JSON. Returns null on malformed input.
    /// </summary>
    internal static LicenseClaims? TryParse(string json)
    {
        try
        {
            return JsonConvert.DeserializeObject<LicenseClaims>(json);
        }
        catch (JsonException)
        {
            return null;
        }
    }
}
