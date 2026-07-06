using System.Security.Cryptography;
using Newtonsoft.Json;

namespace AiDotNet.Helpers;

/// <summary>
/// Provides the PUBLIC license-signing key(s) embedded as a resource in official builds.
/// Used to verify <c>aidn2.</c> asymmetric license tokens offline.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The new license format (<c>aidn2.</c>) is signed on the AiDotNet
/// license server with a <b>private</b> key that never leaves the server. Every published DLL only
/// embeds the matching <b>public</b> key. A public key can verify a signature but cannot create one,
/// so — unlike the old symmetric build key — extracting it from the DLL gives an attacker nothing:
/// they still cannot forge a license without the private key.</para>
///
/// <para>The embedded resource <c>AiDotNet.LicensePublicKey</c> is a small JSON document listing one
/// or more public keys, each tagged with a <c>kid</c> (key id). A license token names the <c>kid</c>
/// it was signed with, so the signing key can be <b>rotated</b> (retire a compromised key, issue a
/// new one) without invalidating already-issued licenses that reference an older, still-trusted key.</para>
///
/// <para>Resource schema (JWK-like — RSA public keys as base64url modulus <c>n</c> and exponent
/// <c>e</c>, which import identically on net471/net8/net10):</para>
/// <code>
/// { "keys": [ { "kid": "2026a", "n": "&lt;base64url modulus&gt;", "e": "&lt;base64url exponent&gt;" } ] }
/// </code>
/// </remarks>
internal static class LicensePublicKeyProvider
{
    private const string ResourceName = "AiDotNet.LicensePublicKey";

    private static Dictionary<string, RSAParameters>? _cachedKeys;
    private static bool _loaded;
    private static readonly object _lock = new();

    /// <summary>
    /// TEST-ONLY: overrides the embedded public key set so tests can inject an ephemeral test keypair's
    /// public half and exercise the real signature-verification path. Pass <see langword="null"/> to
    /// simulate a dev/fork build (no embedded public key) and assert fail-closed offline behaviour.
    /// </summary>
    internal static void OverrideForTesting(IReadOnlyDictionary<string, RSAParameters>? keys)
    {
        lock (_lock)
        {
            _cachedKeys = keys is { Count: > 0 }
                ? new Dictionary<string, RSAParameters>(CloneAll(keys), StringComparer.Ordinal)
                : null;
            _loaded = true;
        }
    }

    /// <summary>
    /// Snapshots the currently-effective public key set (for scoped test overrides), or null if none.
    /// </summary>
    internal static IReadOnlyDictionary<string, RSAParameters>? CurrentSnapshot()
    {
        lock (_lock)
        {
            EnsureLoadedNoLock();
            return _cachedKeys is { Count: > 0 }
                ? new Dictionary<string, RSAParameters>(CloneAll(_cachedKeys), StringComparer.Ordinal)
                : null;
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
    /// Resolves the public key for a given <c>kid</c>. Returns false if this build embeds no matching key.
    /// </summary>
    internal static bool TryGetPublicKey(string kid, out RSAParameters parameters)
    {
        parameters = default;
        if (string.IsNullOrEmpty(kid)) return false;

        lock (_lock)
        {
            EnsureLoadedNoLock();
            if (_cachedKeys is not null && _cachedKeys.TryGetValue(kid, out var found))
            {
                parameters = Clone(found);
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
    /// Parses the JWK-like public-key manifest. Returns null when no usable key is present.
    /// </summary>
    internal static Dictionary<string, RSAParameters>? Parse(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;

        var doc = JsonConvert.DeserializeAnonymousType(json, new
        {
            keys = (System.Collections.Generic.List<KeyEntry>?)null
        });

        if (doc?.keys is null || doc.keys.Count == 0) return null;

        var result = new Dictionary<string, RSAParameters>(StringComparer.Ordinal);
        foreach (var entry in doc.keys)
        {
            string? kid = entry?.kid;
            string? n = entry?.n;
            string? e = entry?.e;
            // Explicit is-null checks (not string.IsNullOrEmpty) so the nullable flow analysis narrows
            // uniformly across TFMs — net471's BCL lacks the [NotNullWhen] annotation on IsNullOrEmpty.
            if (kid is null || kid.Length == 0 || n is null || n.Length == 0 || e is null || e.Length == 0)
            {
                continue;
            }

            try
            {
                var parameters = new RSAParameters
                {
                    Modulus = Base64UrlHelper.Decode(n),
                    Exponent = Base64UrlHelper.Decode(e)
                };
                result[kid] = parameters;
            }
            catch (FormatException ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    "LicensePublicKeyProvider: skipping malformed key '" + kid + "': " + ex.Message);
            }
        }

        return result.Count > 0 ? result : null;
    }

    private static Dictionary<string, RSAParameters> CloneAll(IReadOnlyDictionary<string, RSAParameters> src)
    {
        var copy = new Dictionary<string, RSAParameters>(StringComparer.Ordinal);
        foreach (var kvp in src)
        {
            copy[kvp.Key] = Clone(kvp.Value);
        }

        return copy;
    }

    private static RSAParameters Clone(RSAParameters p) => new()
    {
        Modulus = p.Modulus is null ? null : (byte[])p.Modulus.Clone(),
        Exponent = p.Exponent is null ? null : (byte[])p.Exponent.Clone()
    };

    private sealed class KeyEntry
    {
        public string? kid { get; set; }
        public string? n { get; set; }
        public string? e { get; set; }
    }
}
