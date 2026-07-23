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
    /// Unique token id. The revocation deny-list (CRL) names revoked <c>jti</c> values, so a specific
    /// leaked token can be killed before its <c>exp</c> without revoking the whole signing key. Absent on
    /// legacy tokens (treated as non-revocable-by-jti — only <c>kid</c>/expiry bound them).
    /// </summary>
    [JsonProperty("jti")]
    public string? Jti { get; set; }

    /// <summary>
    /// Explicit capability grants (e.g. <c>model:save</c>, <c>tensors:save</c>, <c>model:encrypt</c>,
    /// <c>offline</c>). Authoritative for offline capability gating: the persistence/encryption guards
    /// check these rather than assuming any <c>Active</c> license unlocks everything. Null/empty means
    /// no explicit grants (tier-derived defaults may still apply during migration).
    /// </summary>
    [JsonProperty("caps")]
    public string[]? Caps { get; set; }

    /// <summary>
    /// Optional machine-binding fingerprint (node-lock). When present, the verifier requires it to equal
    /// this machine's id hash, so a leaked customer token is useless on another machine. Omitted for
    /// non-node-locked tokens (e.g. the CI token, which is ephemeral-machine and uses <c>scope</c> instead).
    /// </summary>
    [JsonProperty("mach")]
    public string? Mach { get; set; }

    /// <summary>
    /// Optional audience/scope binding (e.g. <c>"ci"</c>, <c>"prod"</c>). When present, the verifier
    /// requires it to equal the host's configured expected scope, fencing a scoped token (like the CI
    /// key) off from unintended contexts even if it leaks.
    /// </summary>
    [JsonProperty("scope")]
    public string? Scope { get; set; }

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
