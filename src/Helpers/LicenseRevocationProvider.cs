using Newtonsoft.Json;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Crypto.Signers;

namespace AiDotNet.Helpers;

/// <summary>
/// Offline revocation deny-list (CRL) for <c>aidn2.</c> tokens. Lets a specific leaked token
/// (<c>jti</c>) or a compromised signing key (<c>kid</c>) be killed BEFORE its <c>exp</c>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A signed license can't be "un-signed", so if a token leaks we need a way
/// to reject it before it naturally expires. The server publishes a small <b>signed</b> list of revoked
/// token ids / key ids. Clients ship an embedded copy (refreshed each release) and may also fetch a
/// newer one when they validate online. Because the list is signed with the same private key, nobody can
/// forge or tamper with it — the client verifies it with the embedded PUBLIC key, exactly like a token.</para>
///
/// <para><b>Fail-open by design:</b> revocation is ADDITIVE. A build with no CRL, or a CRL that is
/// expired / unsigned / malformed, revokes nothing — the token's own <c>exp</c> and bindings still bound
/// it. Only a valid (signature-verified, unexpired) CRL can revoke.</para>
///
/// <para><b>Wire format (JWS-style, sign-over-exact-bytes like aidn2):</b></para>
/// <code>
/// { "kid": "&lt;signing kid&gt;",
///   "payload": "&lt;base64url(payload_json)&gt;",   // { "iat":.., "exp":.., "rkids":[..], "rjti":[..] }
///   "sig": "&lt;base64url(ed25519 sig over the decoded payload bytes)&gt;" }
/// </code>
/// The signature is computed over the RAW bytes recovered by base64url-decoding <c>payload</c>, so there
/// is no JSON canonicalization ambiguity (identical guarantee to <see cref="AsymmetricLicenseVerifier"/>).
/// </remarks>
internal static class LicenseRevocationProvider
{
    /// <summary>Embedded manifest resource holding the release-time CRL (may be absent).</summary>
    private const string ResourceName = "AiDotNet.LicenseRevocation";

    private const int Ed25519SignatureSize = 64;

    private static readonly object _lock = new();
    private static bool _embeddedLoaded;
    private static bool _diskCacheLoaded;
    private static Crl? _embedded;   // release-embedded CRL (verified once)
    private static Crl? _fetched;    // newer CRL installed from an online refresh (verified on install)

    /// <summary>
    /// Returns true when a valid (signature-verified, unexpired) CRL revokes this token by its
    /// <paramref name="kid"/> or <paramref name="jti"/>. The most recent valid CRL among the embedded and
    /// the online-fetched copies is consulted. Fails open (returns false) when no valid CRL is present.
    /// </summary>
    internal static bool IsRevoked(string? kid, string? jti)
    {
        var crl = Effective();
        if (crl is null) return false;

        if (kid is { Length: > 0 } && crl.RevokedKids.Contains(kid)) return true;
        if (jti is { Length: > 0 } && crl.RevokedJti.Contains(jti)) return true;
        return false;
    }

    /// <summary>
    /// Installs a CRL obtained from an ONLINE refresh. Verified and expiry-checked here; ignored (kept at
    /// the previous state) if it fails to verify, is expired, or is older than the one already installed.
    /// Returns true when it was accepted and installed.
    /// </summary>
    internal static bool TryInstallFetched(string json, DateTimeOffset nowUtc)
    {
        var candidate = ParseAndVerify(json, nowUtc);
        if (candidate is null) return false;

        lock (_lock)
        {
            // Only move forward in time — a replayed older CRL must never shrink the deny-list.
            if (_fetched is not null && candidate.Iat <= _fetched.Iat) return false;
            _fetched = candidate;
            return true;
        }
    }

    /// <summary>TEST-ONLY: sets the effective fetched CRL (already-parsed) or clears both to a known state.</summary>
    internal static void OverrideForTesting(string? crlJson, DateTimeOffset nowUtc)
    {
        lock (_lock)
        {
            _embeddedLoaded = true;
            // Also mark the disk cache as "handled" so a test's explicit CRL isn't silently overridden by a
            // real ~/.aidotnet/revocations.crl left on the machine — tests must be deterministic.
            _diskCacheLoaded = true;
            _embedded = null;
            _fetched = crlJson is null ? null : ParseAndVerify(crlJson, nowUtc);
        }
    }

    /// <summary>The most recent valid CRL among the embedded and fetched copies (or null).</summary>
    private static Crl? Effective()
    {
        lock (_lock)
        {
            EnsureEmbeddedLoadedNoLock();
            EnsureDiskCacheLoadedNoLock();
            if (_embedded is null) return _fetched;
            if (_fetched is null) return _embedded;
            return _fetched.Iat >= _embedded.Iat ? _fetched : _embedded;
        }
    }

    /// <summary>
    /// Loads the last online-fetched CRL cached on disk (by <see cref="OnlineLicenseServices"/>) once per
    /// process, so revocation is enforced even on a fully-offline start that never refreshes. A later live
    /// refresh via <see cref="TryInstallFetched"/> supersedes it (newer iat wins). Fail-open on any error.
    /// </summary>
    private static void EnsureDiskCacheLoadedNoLock()
    {
        if (_diskCacheLoaded) return;
        _diskCacheLoaded = true;
        try
        {
            string? json = OnlineLicenseServices.ReadCachedCrl();
            if (json is null) return;
            var candidate = ParseAndVerify(json, DateTimeOffset.UtcNow);
            if (candidate is null) return;
            if (_fetched is null || candidate.Iat > _fetched.Iat) _fetched = candidate;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "LicenseRevocationProvider: failed to load cached CRL: " + ex.GetType().Name + ": " + ex.Message);
        }
    }

    private static void EnsureEmbeddedLoadedNoLock()
    {
        if (_embeddedLoaded) return;
        _embeddedLoaded = true;
        try
        {
            var assembly = typeof(LicenseRevocationProvider).Assembly;
            using var stream = assembly.GetManifestResourceStream(ResourceName);
            if (stream is null || stream.Length == 0) return;
            using var reader = new StreamReader(stream, System.Text.Encoding.UTF8);
            _embedded = ParseAndVerify(reader.ReadToEnd(), DateTimeOffset.UtcNow);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "LicenseRevocationProvider: failed to load embedded CRL: " + ex.GetType().Name + ": " + ex.Message);
        }
    }

    /// <summary>
    /// Parses a CRL envelope, verifies its Ed25519 signature against the embedded public key selected by
    /// its <c>kid</c>, and enforces expiry. Returns null on any problem (fail-open: not enforced).
    /// </summary>
    internal static Crl? ParseAndVerify(string json, DateTimeOffset nowUtc)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;

        Envelope? env;
        try { env = JsonConvert.DeserializeObject<Envelope>(json); }
        catch (JsonException) { return null; }
        if (env is null || string.IsNullOrEmpty(env.kid) ||
            string.IsNullOrEmpty(env.payload) || string.IsNullOrEmpty(env.sig))
        {
            return null;
        }

        byte[] payloadBytes, sig;
        try
        {
            payloadBytes = Base64UrlHelper.Decode(env.payload!);
            sig = Base64UrlHelper.Decode(env.sig!);
        }
        catch (FormatException) { return null; }

        if (payloadBytes.Length == 0 || sig.Length != Ed25519SignatureSize) return null;
        if (!LicensePublicKeyProvider.TryGetPublicKey(env.kid!, out byte[] publicKey)) return null;

        try
        {
            var verifier = new Ed25519Signer();
            verifier.Init(false, new Ed25519PublicKeyParameters(publicKey, 0));
            verifier.BlockUpdate(payloadBytes, 0, payloadBytes.Length);
            if (!verifier.VerifySignature(sig)) return null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "LicenseRevocationProvider: CRL signature verification error: " + ex.GetType().Name + ": " + ex.Message);
            return null;
        }

        Payload? p;
        try { p = JsonConvert.DeserializeObject<Payload>(System.Text.Encoding.UTF8.GetString(payloadBytes)); }
        catch (Exception ex) when (ex is JsonException or ArgumentException) { return null; }
        if (p is null) return null;

        // Ignore an expired CRL: it is stale, not authoritative. The client should refetch; offline the
        // token's own exp still applies. (exp==0 => no expiry; treated as always-current.)
        if (p.exp > 0 && DateTimeOffset.FromUnixTimeSeconds(p.exp) < nowUtc) return null;

        return new Crl(
            p.iat,
            new HashSet<string>(p.rkids ?? Array.Empty<string>(), StringComparer.Ordinal),
            new HashSet<string>(p.rjti ?? Array.Empty<string>(), StringComparer.Ordinal));
    }

    internal sealed class Crl
    {
        internal long Iat { get; }
        internal HashSet<string> RevokedKids { get; }
        internal HashSet<string> RevokedJti { get; }
        internal Crl(long iat, HashSet<string> kids, HashSet<string> jti)
        {
            Iat = iat; RevokedKids = kids; RevokedJti = jti;
        }
    }

    private sealed class Envelope
    {
        public string? kid { get; set; }
        public string? payload { get; set; }
        public string? sig { get; set; }
    }

    private sealed class Payload
    {
        public long iat { get; set; }
        public long exp { get; set; }
        public string[]? rkids { get; set; }
        public string[]? rjti { get; set; }
    }
}
