using System.Text;
using AiDotNet.Enums;
using AiDotNet.Models;
using Newtonsoft.Json;

namespace AiDotNet.Helpers;

/// <summary>
/// Client glue for the v2 hybrid online→offline licensing model: fetches the signed revocation list (CRL)
/// and mints/caches short-lived offline <c>aidn2</c> tokens off a SUCCESSFUL online validation, so an
/// <c>AIDN-*</c> server key keeps working — with the correct capabilities and while staying revocable — after
/// the machine goes offline.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> An <c>AIDN-*</c> key normally must phone home to the license server on every
/// run. That breaks air-gapped / offline use. This helper closes the gap: right after a successful online
/// check, it (a) downloads the signed revocation list so leaked keys can be denied even offline, and (b) asks
/// the server for a short-lived, machine-locked offline token and caches it. Next time the server is
/// unreachable, the SDK verifies that cached token locally (against the embedded public key) instead of
/// failing — until the token expires and a fresh online check is needed.</para>
///
/// <para><b>Fail-open / best-effort:</b> every network + disk operation here is wrapped so a failure NEVER
/// breaks or blocks the validation that triggered it. The refresh runs on a background task (off the hot
/// path) and is throttled so it doesn't hit the network on every validation. The offline fallback only ever
/// GRANTS when a genuinely signature-valid, unexpired, machine-matched token is present — it can't weaken
/// security, only preserve availability.</para>
/// </remarks>
internal static class OnlineLicenseServices
{
    /// <summary>Minimum spacing between background refreshes (per machine, tracked by cache-file mtime), so a
    /// tight validate loop doesn't hammer the endpoints. Well inside the offline token's exp so it stays fresh.</summary>
    private static readonly TimeSpan RefreshInterval = TimeSpan.FromHours(12);

    private static readonly object FileLock = new();
    private static string? _cacheDirOverrideForTests;

#if !NET471
    private static readonly System.Net.Http.HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(10) };
#endif

    private static string CacheDir => _cacheDirOverrideForTests ?? Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet");

    private static string CrlPath => Path.Combine(CacheDir, "revocations.crl");

    // One cache file per key (keyed by a short hash of the key, never the plaintext key) so multiple keys on
    // one machine don't clobber each other's offline tokens.
    private static string TokenPath(string licenseKey) =>
        Path.Combine(CacheDir, "offline-" + ShortHash(licenseKey) + ".token");

    /// <summary>TEST-ONLY: redirects the cache directory so tests don't touch the real user profile.</summary>
    internal static IDisposable OverrideCacheDirForTesting(string dir)
    {
        var previous = _cacheDirOverrideForTests;
        _cacheDirOverrideForTests = dir;
        return new CacheDirScope(previous);
    }

    /// <summary>TEST-ONLY: writes an offline token to the same per-key cache path <see cref="RefreshCore"/> uses.</summary>
    internal static void CacheOfflineTokenForTesting(string licenseKey, string token) =>
        WriteAtomic(TokenPath(licenseKey), token);

    /// <summary>TEST-ONLY: writes a CRL to the same cache path a background refresh would.</summary>
    internal static void CacheCrlForTesting(string crl) => WriteAtomic(CrlPath, crl);

    // ───────────────────────── refresh (background, off the hot path) ─────────────────────────

    /// <summary>
    /// Fire-and-forget refresh triggered after a successful ONLINE validation: pulls the latest signed CRL and
    /// (for a server <c>AIDN-*</c> key) mints + caches a fresh offline token. Never blocks the caller and never
    /// throws. Throttled by <see cref="RefreshInterval"/>.
    /// </summary>
    internal static void RefreshInBackground(string validateUrl, string licenseKey, string machineIdHash)
    {
        if (!ShouldRefresh())
        {
            return;
        }

        try
        {
#if NET471
            var thread = new System.Threading.Thread(() => RefreshCore(validateUrl, licenseKey, machineIdHash))
            {
                IsBackground = true,
                Name = "aidotnet-license-refresh"
            };
            thread.Start();
#else
            _ = System.Threading.Tasks.Task.Run(() => RefreshCore(validateUrl, licenseKey, machineIdHash));
#endif
        }
        catch (Exception ex)
        {
            // Even scheduling the work is best-effort — a thread-pool/thread failure must not surface.
            System.Diagnostics.Trace.TraceWarning(
                "OnlineLicenseServices: failed to schedule refresh: " + ex.GetType().Name + ": " + ex.Message);
        }
    }

    private static void RefreshCore(string validateUrl, string licenseKey, string machineIdHash)
    {
        // CRL first: cheapest, benefits every key type (revocation enforcement offline).
        try
        {
            string? crlUrl = DeriveFunctionUrl(validateUrl, "get-revocations");
            if (crlUrl is not null)
            {
                string? crl = HttpGet(crlUrl);
                if (!string.IsNullOrWhiteSpace(crl) &&
                    LicenseRevocationProvider.TryInstallFetched(crl!, DateTimeOffset.UtcNow))
                {
                    WriteAtomic(CrlPath, crl!);
                }
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "OnlineLicenseServices: CRL refresh failed: " + ex.GetType().Name + ": " + ex.Message);
        }

        // Offline token: only meaningful for a server AIDN-* key (aidn./aidn2. keys already verify offline).
        try
        {
            if (LicenseValidator.IsServerValidatedKeyFormat(licenseKey))
            {
                string? issueUrl = DeriveFunctionUrl(validateUrl, "issue-license");
                if (issueUrl is not null)
                {
                    string body = JsonConvert.SerializeObject(new
                    {
                        license_key = licenseKey,
                        machine_id_hash = machineIdHash,
                        package = "AiDotNet"
                    });
                    string? resp = HttpPost(issueUrl, body);
                    string? token = ExtractOfflineToken(resp);
                    // Only cache a token that actually verifies against THIS build's embedded public key — a
                    // token we can't verify is useless offline, and caching it would just fail later.
                    if (token is not null &&
                        AsymmetricLicenseVerifier.Verify(token, DateTimeOffset.UtcNow).Status == LicenseKeyStatus.Active)
                    {
                        WriteAtomic(TokenPath(licenseKey), token);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "OnlineLicenseServices: offline-token issue failed: " + ex.GetType().Name + ": " + ex.Message);
        }
    }

    // ───────────────────────── offline consumption ─────────────────────────

    /// <summary>
    /// Offline fallback used when the license server is unreachable: returns the cached offline <c>aidn2</c>
    /// token's validation result if one exists and verifies (Active) against the embedded public key on THIS
    /// machine, else null. Because it delegates to <see cref="AsymmetricLicenseVerifier"/>, all v2 bindings
    /// (signature, exp, machine-lock, CRL revocation) are enforced — a leaked or expired cached token yields
    /// a non-Active result and is ignored by the caller.
    /// </summary>
    internal static LicenseValidationResult? TryValidateCachedOfflineToken(string licenseKey)
    {
        try
        {
            string path = TokenPath(licenseKey);
            string token;
            lock (FileLock)
            {
                if (!File.Exists(path))
                {
                    return null;
                }

                token = File.ReadAllText(path).Trim();
            }

            if (!AsymmetricLicenseVerifier.IsAsymmetricKeyFormat(token))
            {
                return null;
            }

            var result = AsymmetricLicenseVerifier.Verify(token, DateTimeOffset.UtcNow);
            return result.Status == LicenseKeyStatus.Active ? result : null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "OnlineLicenseServices: cached offline token read failed: " + ex.GetType().Name + ": " + ex.Message);
            return null;
        }
    }

    /// <summary>Reads the last-fetched CRL from disk (or null). Used by <see cref="LicenseRevocationProvider"/>
    /// to enforce the most recent online revocation list even on a fully offline start.</summary>
    internal static string? ReadCachedCrl()
    {
        try
        {
            lock (FileLock)
            {
                return File.Exists(CrlPath) ? File.ReadAllText(CrlPath) : null;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "OnlineLicenseServices: cached CRL read failed: " + ex.GetType().Name + ": " + ex.Message);
            return null;
        }
    }

    // ───────────────────────── helpers ─────────────────────────

    /// <summary>
    /// Derives a sibling edge-function URL (e.g. get-revocations, issue-license) from the validate-license URL
    /// by swapping the trailing function segment. Returns null when the URL isn't the expected
    /// <c>…/validate-license</c> shape (e.g. a custom test stub), so callers simply skip the feature.
    /// </summary>
    internal static string? DeriveFunctionUrl(string validateUrl, string functionName)
    {
        if (string.IsNullOrEmpty(validateUrl))
        {
            return null;
        }

        const string marker = "/validate-license";
        int i = validateUrl.LastIndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (i < 0)
        {
            return null;
        }

        // Preserve anything after the segment (e.g. a query string) though normally there is none.
        string tail = validateUrl[(i + marker.Length)..];
        return validateUrl[..i] + "/" + functionName + tail;
    }

    private static string? ExtractOfflineToken(string? responseJson)
    {
        if (string.IsNullOrWhiteSpace(responseJson))
        {
            return null;
        }

        try
        {
            var obj = JsonConvert.DeserializeAnonymousType(responseJson!, new { valid = false, offline_token = (string?)null });
            return obj is { valid: true } && !string.IsNullOrWhiteSpace(obj.offline_token) ? obj.offline_token : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>True when no cache file exists yet or the freshest one is older than <see cref="RefreshInterval"/>.</summary>
    private static bool ShouldRefresh()
    {
        try
        {
            DateTime newest = DateTime.MinValue;
            lock (FileLock)
            {
                if (Directory.Exists(CacheDir))
                {
                    if (File.Exists(CrlPath))
                    {
                        newest = File.GetLastWriteTimeUtc(CrlPath);
                    }

                    foreach (string f in Directory.EnumerateFiles(CacheDir, "offline-*.token"))
                    {
                        DateTime t = File.GetLastWriteTimeUtc(f);
                        if (t > newest)
                        {
                            newest = t;
                        }
                    }
                }
            }

            return DateTime.UtcNow - newest >= RefreshInterval;
        }
        catch
        {
            // If we can't tell how fresh the cache is, err toward refreshing (still throttled by the fact that
            // it only runs after a successful online validation).
            return true;
        }
    }

    private static void WriteAtomic(string path, string content)
    {
        lock (FileLock)
        {
            string? dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir))
            {
                Directory.CreateDirectory(dir);
            }

            // Write to a temp file then move, so a concurrent reader never sees a half-written file.
            string tmp = path + ".tmp";
            File.WriteAllText(tmp, content);
            if (File.Exists(path))
            {
                File.Delete(path);
            }

            File.Move(tmp, path);
        }
    }

    private static string ShortHash(string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes("license-cache:" + value);
#if NET471
        using var sha = System.Security.Cryptography.SHA256.Create();
        byte[] hash = sha.ComputeHash(bytes);
#else
        byte[] hash = System.Security.Cryptography.SHA256.HashData(bytes);
#endif
        var sb = new StringBuilder(16);
        for (int i = 0; i < 8; i++)
        {
            sb.Append(hash[i].ToString("x2"));
        }

        return sb.ToString();
    }

    private static string? HttpGet(string url)
    {
#if NET471
        using var client = new System.Net.WebClient();
        return client.DownloadString(url);
#else
        using var resp = Http.GetAsync(url).ConfigureAwait(false).GetAwaiter().GetResult();
        return resp.Content.ReadAsStringAsync().ConfigureAwait(false).GetAwaiter().GetResult();
#endif
    }

    private static string? HttpPost(string url, string json)
    {
#if NET471
        using var client = new System.Net.WebClient();
        client.Headers[System.Net.HttpRequestHeader.ContentType] = "application/json";
        try
        {
            return client.UploadString(url, "POST", json);
        }
        catch (System.Net.WebException ex) when (ex.Response is System.Net.HttpWebResponse http)
        {
            // A 403 (e.g. invalid_key) still carries a JSON body we want to inspect/ignore cleanly.
            using var reader = new System.IO.StreamReader(http.GetResponseStream()!);
            return reader.ReadToEnd();
        }
#else
        using var content = new System.Net.Http.StringContent(json, Encoding.UTF8, "application/json");
        using var resp = Http.PostAsync(url, content).ConfigureAwait(false).GetAwaiter().GetResult();
        return resp.Content.ReadAsStringAsync().ConfigureAwait(false).GetAwaiter().GetResult();
#endif
    }

    private sealed class CacheDirScope : IDisposable
    {
        private readonly string? _previous;
        public CacheDirScope(string? previous) => _previous = previous;
        public void Dispose() => _cacheDirOverrideForTests = _previous;
    }
}
