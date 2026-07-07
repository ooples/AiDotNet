using Newtonsoft.Json;

namespace AiDotNet.Helpers;

/// <summary>
/// Provides the PUBLIC license-signing key(s) embedded as a resource in official builds.
/// Used to verify <c>aidn2.</c> asymmetric (Ed25519 / EdDSA) license tokens offline.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The new license format (<c>aidn2.</c>) is signed on the AiDotNet
/// license server with a <b>private</b> key that never leaves the server. Every published DLL only
/// embeds the matching <b>public</b> key. A public key can verify a signature but cannot create one,
/// so — unlike the old symmetric build key — extracting it from the DLL gives an attacker nothing:
/// they still cannot forge a license without the private key.</para>
///
/// <para>The embedded resource <c>AiDotNet.LicensePublicKey</c> is a small JSON document listing one
/// or more Ed25519 public keys, each tagged with a <c>kid</c> (key id). A license token names the
/// <c>kid</c> it was signed with, so the signing key can be <b>rotated</b> (retire a compromised key,
/// issue a new one) without invalidating already-issued licenses that reference an older, still-trusted
/// key.</para>
///
/// <para>Resource schema (JWK OKP — RFC 8037; an Ed25519 public key is the 32-byte <c>x</c> value):</para>
/// <code>
/// { "keys": [ { "kty": "OKP", "crv": "Ed25519", "kid": "2026a", "x": "&lt;base64url 32-byte pubkey&gt;" } ] }
/// </code>
/// </remarks>
internal static class LicensePublicKeyProvider
{
    private const string ResourceName = "AiDotNet.LicensePublicKey";

    /// <summary>Ed25519 raw public keys are exactly 32 bytes.</summary>
    internal const int Ed25519PublicKeySize = 32;

    // kid -> raw 32-byte Ed25519 public key.
    private static Dictionary<string, byte[]>? _cachedKeys;
    private static bool _loaded;
    private static readonly object _lock = new();

    /// <summary>
    /// TEST-ONLY: overrides the embedded public key set so tests can inject an ephemeral test keypair's
    /// public half and exercise the real signature-verification path. Pass <see langword="null"/> to
    /// simulate a dev/fork build (no embedded public key) and assert fail-closed offline behaviour.
    /// </summary>
    internal static void OverrideForTesting(IReadOnlyDictionary<string, byte[]>? keys)
    {
        lock (_lock)
        {
            _cachedKeys = keys is { Count: > 0 }
                ? CloneAll(keys)
                : null;
            _loaded = true;
        }
    }

    /// <summary>
    /// Snapshots the currently-effective public key set (for scoped test overrides), or null if none.
    /// </summary>
    internal static IReadOnlyDictionary<string, byte[]>? CurrentSnapshot()
    {
        lock (_lock)
        {
            EnsureLoadedNoLock();
            return _cachedKeys is { Count: > 0 } ? CloneAll(_cachedKeys) : null;
        }
    }

    /// <summary>
    /// Gets whether this build embeds at least one license public key (i.e. it can verify
    /// <c>aidn2.</c> tokens offline). The asymmetric analogue of <see cref="BuildKeyProvider.IsOfficialBuild"/>.
    /// </summary>
    internal static bool HasAnyKey
    {
        get
        {
            lock (_lock)
            {
                EnsureLoadedNoLock();
                return _cachedKeys is { Count: > 0 };
            }
        }
    }

    /// <summary>
    /// Resolves the raw 32-byte Ed25519 public key for a given <c>kid</c>. Returns false if this build
    /// embeds no matching key.
    /// </summary>
    internal static bool TryGetPublicKey(string kid, out byte[] publicKey)
    {
        publicKey = Array.Empty<byte>();
        if (string.IsNullOrEmpty(kid)) return false;

        lock (_lock)
        {
            EnsureLoadedNoLock();
            if (_cachedKeys is not null && _cachedKeys.TryGetValue(kid, out var found))
            {
                publicKey = (byte[])found.Clone();
                return true;
            }
        }

        return false;
    }

    private static void EnsureLoadedNoLock()
    {
        if (_loaded) return;

        try
        {
            var assembly = typeof(LicensePublicKeyProvider).Assembly;
            using var stream = assembly.GetManifestResourceStream(ResourceName);
            if (stream is null || stream.Length == 0)
            {
                _cachedKeys = null;
                return;
            }

            using var reader = new StreamReader(stream, System.Text.Encoding.UTF8);
            string json = reader.ReadToEnd();
            _cachedKeys = Parse(json);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "LicensePublicKeyProvider: failed to load public key(s): " + ex.GetType().Name + ": " + ex.Message);
            _cachedKeys = null;
        }
        finally
        {
            _loaded = true;
        }
    }

    /// <summary>
    /// Parses the JWK OKP public-key manifest. Returns null when no usable Ed25519 key is present.
    /// Non-Ed25519 / malformed / wrong-length entries are skipped (fail closed).
    /// </summary>
    internal static Dictionary<string, byte[]>? Parse(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;

        var doc = JsonConvert.DeserializeAnonymousType(json, new
        {
            keys = (System.Collections.Generic.List<KeyEntry>?)null
        });

        if (doc?.keys is null || doc.keys.Count == 0) return null;

        var result = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        foreach (var entry in doc.keys)
        {
            string? kid = entry?.kid;
            string? x = entry?.x;
            string? crv = entry?.crv;
            // Explicit is-null checks (not string.IsNullOrEmpty) so the nullable flow analysis narrows
            // uniformly across TFMs — net471's BCL lacks the [NotNullWhen] annotation on IsNullOrEmpty.
            if (kid is null || kid.Length == 0 || x is null || x.Length == 0)
            {
                continue;
            }

            // Only Ed25519 OKP keys are accepted. A missing crv is tolerated for brevity but a present,
            // non-Ed25519 crv is rejected so a future curve cannot be silently treated as Ed25519.
            if (crv is not null && crv.Length > 0 && !string.Equals(crv, "Ed25519", StringComparison.Ordinal))
            {
                System.Diagnostics.Trace.TraceWarning(
                    "LicensePublicKeyProvider: skipping key '" + kid + "' with unsupported crv '" + crv + "'.");
                continue;
            }

            try
            {
                byte[] pub = Base64UrlHelper.Decode(x);
                if (pub.Length != Ed25519PublicKeySize)
                {
                    System.Diagnostics.Trace.TraceWarning(
                        "LicensePublicKeyProvider: skipping key '" + kid + "' with wrong Ed25519 length " + pub.Length + ".");
                    continue;
                }

                result[kid] = pub;
            }
            catch (FormatException ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    "LicensePublicKeyProvider: skipping malformed key '" + kid + "': " + ex.Message);
            }
        }

        return result.Count > 0 ? result : null;
    }

    private static Dictionary<string, byte[]> CloneAll(IReadOnlyDictionary<string, byte[]> src)
    {
        var copy = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        foreach (var kvp in src)
        {
            copy[kvp.Key] = kvp.Value is null ? Array.Empty<byte>() : (byte[])kvp.Value.Clone();
        }

        return copy;
    }

    private sealed class KeyEntry
    {
        public string? kty { get; set; }
        public string? crv { get; set; }
        public string? kid { get; set; }
        public string? x { get; set; }
    }
}
