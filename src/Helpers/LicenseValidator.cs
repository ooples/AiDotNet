using System.Security.Cryptography;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Models;
using Newtonsoft.Json;

namespace AiDotNet.Helpers;

/// <summary>
/// Client-side license validator that contacts the AiDotNet license server (Supabase Edge Function)
/// and caches the result for an offline grace period.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a license has a <see cref="AiDotNetLicenseKey.ServerUrl"/>
/// (or uses the default server), this validator contacts the server to check whether the key is valid,
/// what tier it belongs to, and whether the machine activation limit has been reached.
/// If the server is unreachable, the validator uses its cached result until the offline grace
/// period expires.</para>
///
/// <para>When no server URL is configured (offline-only mode), the validator performs offline
/// format/signature validation and returns <see cref="LicenseKeyStatus.Active"/> for valid keys.</para>
///
/// <para><b>Default server:</b> The default license validation endpoint is
/// <c>https://yfkqwpgjahoamlgckjib.supabase.co/functions/v1/validate-license</c>.
/// Set <see cref="AiDotNetLicenseKey.ServerUrl"/> to override.</para>
/// </remarks>
internal sealed class LicenseValidator
{
    /// <summary>
    /// Default license validation endpoint (Supabase Edge Function).
    /// </summary>
    internal const string DefaultServerUrl =
        "https://yfkqwpgjahoamlgckjib.supabase.co/functions/v1/validate-license";

    private readonly AiDotNetLicenseKey _licenseKey;
    private LicenseValidationResult? _cached;
    private readonly object _cacheLock = new();

#if !NET471
    private static readonly System.Net.Http.HttpClient SharedHttpClient = new()
    {
        Timeout = TimeSpan.FromSeconds(15)
    };
#endif

    /// <summary>
    /// Gets the most recent cached validation result, or null if no validation has been performed.
    /// </summary>
    internal LicenseValidationResult? CachedResult
    {
        get
        {
            lock (_cacheLock)
            {
                return _cached;
            }
        }
    }

    public LicenseValidator(AiDotNetLicenseKey licenseKey)
    {
        _licenseKey = licenseKey ?? throw new ArgumentNullException(nameof(licenseKey));
    }

    /// <summary>
    /// Validates the license key. Uses the default AiDotNet license server unless
    /// <see cref="AiDotNetLicenseKey.ServerUrl"/> is set to a custom URL or explicitly
    /// set to empty string for offline-only mode.
    /// </summary>
    /// <returns>A <see cref="LicenseValidationResult"/> describing the current key status.</returns>
    public LicenseValidationResult Validate()
    {
        // Determine validation strategy based on key format and server configuration:
        // - ServerUrl explicitly "" → offline-only (premium keys with HMAC signatures)
        // - ServerUrl null → use DefaultServerUrl for online validation
        // - ServerUrl set → use custom URL for online validation
        //
        // Key format determines offline eligibility:
        // - aidn.{id}.{sig} → signed key, eligible for offline HMAC validation (premium)
        // - AIDN-*-{hex}    → server-validated key, requires online validation (community/CI)

        bool explicitOfflineOnly = _licenseKey.ServerUrl is not null
            && _licenseKey.ServerUrl.Trim().Length == 0;

        // A signed key (aidn.{id}.{sig}) can be verified LOCALLY by HMAC — but ONLY when this build
        // actually embeds the official build key (the HMAC secret). When it does, validate the signature
        // offline and skip the network entirely: this removes the blocking online round-trip (up to the
        // 15s HttpClient timeout when the license server is slow/unreachable) at model-save time WITHOUT
        // weakening security — a forged signature fails the constant-time HMAC compare in ValidateOffline().
        // When the build key is ABSENT we must NOT accept an unverifiable signature offline (ValidateOffline
        // would otherwise fail open), so such keys still go ONLINE, where the server is the source of truth.
        // explicitOfflineOnly remains its own deliberate air-gapped opt-in.
        bool buildKeyEmbedded = BuildKeyProvider.GetBuildKey().Length > 0;
        // Only the DEFAULT server config (ServerUrl == null) opportunistically validates a signed key
        // offline; if the caller set an explicit custom ServerUrl they want online validation (e.g. for
        // revocation), so respect it. ServerUrl == "" is the explicit offline-only opt-in handled above.
        bool serverUrlIsDefault = _licenseKey.ServerUrl is null;
        bool useOfflineValidation = explicitOfflineOnly
            || (serverUrlIsDefault && IsSignedKeyFormat(_licenseKey.Key) && buildKeyEmbedded);

        if (useOfflineValidation)
        {
            // Explicit offline mode — only HMAC-signed keys (aidn.{id}.{sig}) are
            // accepted. Server-validated keys (AIDN-PROD-*, AIDN-DEV-*, etc.) MUST
            // go through online validation against the license server because they
            // carry no cryptographic signature the SDK can verify locally — the
            // server is the only source of truth for their valid-vs-revoked status.
            //
            // Previously the offline path ALSO accepted AIDN-* keys (treated them
            // as "valid format" and returned LicenseKeyStatus.Active without any
            // cryptographic check), which is the security gap PR #1256 review
            // flagged: anyone who guessed/forged a well-formed AIDN-PROD-* key
            // would be granted Active status indefinitely in offline-only mode.
            //
            // Future work: add Ed25519-signed AIDN-{ENV}-{TIER}-{V}-{ID}-{SIG}
            // format that CAN be validated offline against a SDK-shipped public
            // key. Until that lands, AIDN-* keys are server-only.
            if (!IsSignedKeyFormat(_licenseKey.Key))
            {
                // Cache the rejection so the sync path matches ValidateAsync()
                // (line ~461) and CachedResult is non-null after the first call.
                // Without this, repeated Validate() calls allocate a fresh
                // Invalid result each time and CachedResult stays null even
                // though we've already decided this key can't validate.
                lock (_cacheLock)
                {
                    if (_cached is not null) return _cached;
                    var rejected = new LicenseValidationResult(
                        LicenseKeyStatus.Invalid,
                        message: "Offline-only mode requires a signed license key (aidn.{id}.{signature} format). " +
                                 "Server-validated keys (AIDN-PROD-*, AIDN-DEV-*) require online validation — set ServerUrl " +
                                 "to null (default endpoint) or to a custom URL. To enable air-gapped operation, " +
                                 "request a signed license key from support.");
                    _cached = rejected;
                    return rejected;
                }
            }

            // Return the cached result on repeat calls so reference equality holds —
            // tests (and consumers) rely on Assert.Same(result1, result2) to
            // distinguish "fresh validation" from "cache hit".
            lock (_cacheLock)
            {
                if (_cached is not null) return _cached;
            }

            var offlineResult = ValidateOffline();
            lock (_cacheLock)
            {
                // Double-checked locking: another thread may have populated the
                // cache while we were validating. Honour their instance to keep
                // reference equality stable across concurrent first calls.
                _cached ??= offlineResult;
                return _cached;
            }
        }

        // Check cache: if still within the grace period, return cached result
        lock (_cacheLock)
        {
            if (_cached is not null &&
                _cached.Status == LicenseKeyStatus.Active &&
                _cached.ValidatedAt + _licenseKey.OfflineGracePeriod > DateTimeOffset.UtcNow)
            {
                return _cached;
            }
        }

        // Attempt online validation
        try
        {
            var result = ValidateOnline();
            lock (_cacheLock)
            {
                _cached = result;
            }

            return result;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "LicenseValidator: online validation failed: " + ex.GetType().Name + ": " + ex.Message);

            // Network failure: check if we have a valid cached result within grace period
            lock (_cacheLock)
            {
                if (_cached is not null &&
                    _cached.ValidatedAt + _licenseKey.OfflineGracePeriod > DateTimeOffset.UtcNow)
                {
                    return _cached;
                }
            }

            // No valid cache — if we have a stale cached result, return expired status
            lock (_cacheLock)
            {
                if (_cached is not null)
                {
                    // Had a previously valid key but it's now past grace period
                    var expired = new LicenseValidationResult(
                        LicenseKeyStatus.Expired,
                        tier: _cached.Tier,
                        message: "License server unreachable and grace period exceeded.");
                    _cached = expired;
                    return expired;
                }
            }

            // Never validated before — return pending for initial grace window only
            var pending = new LicenseValidationResult(
                LicenseKeyStatus.ValidationPending,
                message: "License server unreachable. Initial validation pending.");

            lock (_cacheLock)
            {
                _cached = pending;
            }

            return pending;
        }
    }

    /// <summary>
    /// Validates the license key offline using format checks and HMAC signature verification
    /// when a build key is available. This prevents garbage or empty keys from being accepted.
    /// </summary>
    private LicenseValidationResult ValidateOffline()
    {
        // Reject empty or whitespace-only keys
        if (string.IsNullOrWhiteSpace(_licenseKey.Key))
        {
            return new LicenseValidationResult(
                LicenseKeyStatus.Invalid,
                message: "License key is empty or missing.");
        }

        // Validate the key format: must be aidn.{id}.{signature}
        if (!ValidateKeyFormat(_licenseKey.Key))
        {
            return new LicenseValidationResult(
                LicenseKeyStatus.Invalid,
                message: "License key format is invalid. Expected format: aidn.{id}.{signature}");
        }

        // Offline validation is CRYPTOGRAPHIC and FAILS CLOSED: a signed key is granted Active offline
        // ONLY if its HMAC-SHA256 signature verifies against the embedded build key. Two former fail-open
        // paths are removed: (1) when no build key was embedded, and (2) when the key had no signature
        // segment, the method previously fell through to Active. Both are now rejected. Without the embedded
        // build key the SDK cannot verify a signature, so it must NOT trust a well-formed-but-unverified
        // string — this is precisely what makes air-gapped operation safe: an air-gapped build MUST embed
        // the official build key to grant licenses offline.
        var buildKey = BuildKeyProvider.GetBuildKey();
        if (buildKey.Length == 0)
        {
            return new LicenseValidationResult(
                LicenseKeyStatus.Invalid,
                message: "Offline license verification requires a build that embeds the official build key; " +
                         "this build cannot verify a signature locally. Use online validation, or deploy an " +
                         "official signed build for air-gapped operation.");
        }

        // The key MUST carry a signature segment: payload.signature (base64url-encoded).
        var dotIndex = _licenseKey.Key.LastIndexOf('.');
        if (dotIndex <= 0 || dotIndex >= _licenseKey.Key.Length - 1)
        {
            return new LicenseValidationResult(
                LicenseKeyStatus.Invalid,
                message: "License key has no verifiable signature (expected payload.signature).");
        }

        string payloadPart = _licenseKey.Key[..dotIndex];
        string signaturePart = _licenseKey.Key[(dotIndex + 1)..];

        try
        {
            byte[] payloadBytes = Encoding.UTF8.GetBytes(payloadPart);
            // Convert Base64URL (RFC 4648) to standard Base64 before decoding.
            // Base64URL uses '-' instead of '+' and '_' instead of '/'.
            string standardBase64 = signaturePart
                .Replace('-', '+')
                .Replace('_', '/');
            // Restore padding if stripped
            switch (standardBase64.Length % 4)
            {
                case 2: standardBase64 += "=="; break;
                case 3: standardBase64 += "="; break;
            }
            byte[] expectedSignature = Convert.FromBase64String(standardBase64);

            using var hmac = new System.Security.Cryptography.HMACSHA256(buildKey);
            byte[] computedSignature = hmac.ComputeHash(payloadBytes);

            if (!CryptographicEquals(computedSignature, expectedSignature))
            {
                return new LicenseValidationResult(
                    LicenseKeyStatus.Invalid,
                    message: "License key signature verification failed.");
            }
        }
        catch (FormatException)
        {
            return new LicenseValidationResult(
                LicenseKeyStatus.Invalid,
                message: "License key signature is malformed.");
        }

        // Signature verified against the embedded build key.
        return new LicenseValidationResult(LicenseKeyStatus.Active, message: "Offline validation succeeded (HMAC verified).");
    }

    /// <summary>
    /// Returns true if the key is in signed offline format: aidn.{id}.{signature}.
    /// The id and signature segments are base64url-encoded (RFC 4648 §5):
    /// alphanumeric plus '-' and '_' (no padding). These keys can be validated
    /// offline via HMAC (premium/enterprise only).
    /// </summary>
    /// <remarks>
    /// Accepting the base64url URL-safe alphabet matters because signed keys
    /// elsewhere in the codebase (issuer / Stripe webhook) emit base64url-
    /// encoded HMAC-SHA256 tags via WebEncoders.Base64UrlEncode — those tags
    /// routinely contain '-' / '_'. A stricter alphanumeric-only check would
    /// misclassify those legitimate signed keys as "not signed" and route them
    /// through online validation instead of the offline HMAC path. Originally
    /// landed in PR #1256 follow-up.
    /// </remarks>
    internal static bool IsSignedKeyFormat(string key)
    {
        if (key is null) return false;
        var parts = key.Split('.');
        if (parts.Length != 3 || parts[0] != "aidn" || parts[1].Length == 0 || parts[2].Length == 0)
            return false;

        for (int i = 0; i < parts[1].Length; i++)
            if (!IsBase64UrlChar(parts[1][i])) return false;
        for (int i = 0; i < parts[2].Length; i++)
            if (!IsBase64UrlChar(parts[2][i])) return false;

        return true;
    }

    /// <summary>
    /// Base64url alphabet (RFC 4648 §5): A–Z, a–z, 0–9, '-', '_'. No padding.
    /// </summary>
    private static bool IsBase64UrlChar(char c)
    {
        return (c >= 'A' && c <= 'Z')
            || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9')
            || c == '-'
            || c == '_';
    }

    /// <summary>
    /// Returns true if the key is in server-validated format: AIDN-{segments}-{hex}
    /// These keys require online validation against the license server.
    /// </summary>
    internal static bool IsServerValidatedKeyFormat(string key)
    {
        var parts = key.Split('-');
        if (parts.Length < 4 || !parts[0].Equals("AIDN", StringComparison.OrdinalIgnoreCase))
            return false;

        // Last segment must be at least 8 hex characters
        string lastPart = parts[^1];
        if (lastPart.Length < 8) return false;

        for (int i = 0; i < lastPart.Length; i++)
        {
            char c = lastPart[i];
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Validates key format — accepts either signed (offline) or server-validated (online) format.
    /// </summary>
    internal static bool ValidateKeyFormat(string key)
    {
        return IsSignedKeyFormat(key) || IsServerValidatedKeyFormat(key);
    }

    /// <summary>
    /// Constant-time comparison to prevent timing attacks on signature verification.
    /// </summary>
    private static bool CryptographicEquals(byte[] a, byte[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        int result = 0;
        for (int i = 0; i < a.Length; i++)
        {
            result |= a[i] ^ b[i];
        }

        return result == 0;
    }

    private LicenseValidationResult ValidateOnline()
    {
        string url = _licenseKey.ServerUrl ?? DefaultServerUrl;

        var requestBody = BuildRequestBody();
        string json = JsonConvert.SerializeObject(requestBody);

#if NET471
        return ValidateOnlineNet471(url, json);
#else
        return ValidateOnlineModern(url, json);
#endif
    }

    /// <summary>
    /// Builds the request body for the license validation endpoint.
    /// Includes machine_id_hash (always) and optional hostname/os_description (when telemetry is enabled).
    /// Also tags the request with <c>package=AiDotNet</c> so server-side analytics
    /// can break validation traffic down by client SDK origin (issue #1195 §2d).
    /// AiDotNet.Tensors sends <c>package=AiDotNet.Tensors</c> from its own validator.
    /// </summary>
    internal Dictionary<string, string?> BuildRequestBody()
    {
        var body = new Dictionary<string, string?>
        {
            ["license_key"] = _licenseKey.Key,
            ["machine_id_hash"] = GetMachineIdHash(),
            // Issue #1195 §2d: tag this request with the package name so the
            // server's analytics can attribute validation calls to AiDotNet
            // vs AiDotNet.Tensors. The server does NOT gate on this field —
            // it always returns the full capability set the user's tier
            // authorises, regardless of which package asked.
            ["package"] = "AiDotNet"
        };

        if (_licenseKey.EnableTelemetry)
        {
            try { body["hostname"] = System.Environment.MachineName; }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning("LicenseValidator: unable to read MachineName: " + ex.Message);
            }

            try
            {
                body["os_description"] = System.Runtime.InteropServices.RuntimeInformation.OSDescription;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning("LicenseValidator: unable to read OSDescription: " + ex.Message);
            }
        }

        return body;
    }

    /// <summary>
    /// Generates a one-way SHA-256 hash of the machine fingerprint for license activation tracking.
    /// Uses a different salt than telemetry to prevent correlation between the two systems.
    /// </summary>
    internal static string GetMachineIdHash()
    {
        string rawId = MachineFingerprint.GetMachineId();
        byte[] bytes = Encoding.UTF8.GetBytes("license-validation:" + rawId);

#if NET471
        using var sha = SHA256.Create();
        byte[] hash = sha.ComputeHash(bytes);
#else
        byte[] hash = SHA256.HashData(bytes);
#endif

        var sb = new StringBuilder(hash.Length * 2);
        for (int i = 0; i < hash.Length; i++)
        {
            sb.Append(hash[i].ToString("x2"));
        }

        return sb.ToString();
    }

#if !NET471
    private async System.Threading.Tasks.Task<LicenseValidationResult> ValidateOnlineAsync(
        System.Threading.CancellationToken cancellationToken = default)
    {
        string url = _licenseKey.ServerUrl ?? DefaultServerUrl;

        var requestBody = BuildRequestBody();
        string json = JsonConvert.SerializeObject(requestBody);
        var content = new System.Net.Http.StringContent(json, Encoding.UTF8, "application/json");
        var response = await SharedHttpClient.PostAsync(url, content, cancellationToken).ConfigureAwait(false);
        string responseJson = await response.Content.ReadAsStringAsync().ConfigureAwait(false);

        return ParseResponse(responseJson, (int)response.StatusCode);
    }

    /// <summary>
    /// Asynchronously validates the configured license key, avoiding sync-over-async.
    /// Prefer this over <see cref="Validate"/> in async contexts to prevent thread pool starvation.
    /// </summary>
    public async System.Threading.Tasks.Task<LicenseValidationResult> ValidateAsync(
        System.Threading.CancellationToken cancellationToken = default)
    {
        // Offline validation path — mirrors Validate(): taken for explicit offline-only mode
        // (ServerUrl=="") OR for a signed key when this build embeds the build key (so the HMAC signature
        // is verifiable locally). A signed key WITHOUT an embedded build key falls through to online
        // validation (the server is the source of truth) — it is never accepted unverified. AIDN-*
        // server-validated keys always require online validation. ValidateOffline() itself fails closed.
        bool explicitOfflineOnly = _licenseKey.ServerUrl is not null && _licenseKey.ServerUrl.Trim().Length == 0;
        bool buildKeyEmbedded = BuildKeyProvider.GetBuildKey().Length > 0;
        bool serverUrlIsDefault = _licenseKey.ServerUrl is null;
        if (explicitOfflineOnly || (serverUrlIsDefault && IsSignedKeyFormat(_licenseKey.Key) && buildKeyEmbedded))
        {
            if (!IsSignedKeyFormat(_licenseKey.Key))
            {
                var rejected = new LicenseValidationResult(
                    LicenseKeyStatus.Invalid,
                    message: "Offline-only mode requires a signed license key (aidn.{id}.{signature} format). " +
                             "Server-validated keys (AIDN-PROD-*) require online validation.");
                lock (_cacheLock) { _cached = rejected; }
                return rejected;
            }
            var offlineResult = ValidateOffline();
            lock (_cacheLock) { _cached = offlineResult; }
            return offlineResult;
        }

        lock (_cacheLock)
        {
            if (_cached is not null &&
                _cached.Status == LicenseKeyStatus.Active &&
                _cached.ValidatedAt + _licenseKey.OfflineGracePeriod > DateTimeOffset.UtcNow)
            {
                return _cached;
            }
        }

        try
        {
            var result = await ValidateOnlineAsync(cancellationToken).ConfigureAwait(false);
            lock (_cacheLock) { _cached = result; }
            return result;
        }
        catch
        {
            lock (_cacheLock)
            {
                if (_cached is not null &&
                    _cached.ValidatedAt + _licenseKey.OfflineGracePeriod > DateTimeOffset.UtcNow)
                {
                    return _cached;
                }

                if (_cached is not null)
                {
                    var expired = new LicenseValidationResult(
                        LicenseKeyStatus.Expired, tier: _cached.Tier,
                        message: "License server unreachable and grace period exceeded.");
                    _cached = expired;
                    return expired;
                }
            }

            var pending = new LicenseValidationResult(
                LicenseKeyStatus.ValidationPending,
                message: "License server unreachable. Initial validation pending.");
            lock (_cacheLock) { _cached = pending; }
            return pending;
        }
    }
#endif

#if NET471
    private LicenseValidationResult ValidateOnlineNet471(string url, string json)
    {
        using var client = new System.Net.WebClient();
        client.Headers[System.Net.HttpRequestHeader.ContentType] = "application/json";
        try
        {
            string responseJson = client.UploadString(url, "POST", json);
            return ParseResponse(responseJson, 200);
        }
        catch (System.Net.WebException ex) when (ex.Response is System.Net.HttpWebResponse httpResponse)
        {
            using var reader = new System.IO.StreamReader(httpResponse.GetResponseStream()!);
            string errorJson = reader.ReadToEnd();
            return ParseResponse(errorJson, (int)httpResponse.StatusCode);
        }
    }
#else
    private LicenseValidationResult ValidateOnlineModern(string url, string json)
    {
        var content = new System.Net.Http.StringContent(json, Encoding.UTF8, "application/json");
        var response = SharedHttpClient.PostAsync(url, content).ConfigureAwait(false).GetAwaiter().GetResult();
        string responseJson = response.Content.ReadAsStringAsync().ConfigureAwait(false).GetAwaiter().GetResult();

        return ParseResponse(responseJson, (int)response.StatusCode);
    }
#endif

    /// <summary>
    /// Parses the JSON response from the Supabase Edge Function validate-license endpoint.
    /// The response schema is: { valid: bool, tier?: string, error?: string, message?: string,
    /// license_id?: string, activation_id?: string, current_activations?: int, max_activations?: int,
    /// capabilities?: string[] }
    /// </summary>
    /// <remarks>
    /// <para>The <c>capabilities</c> field was introduced in issue #1195. Older
    /// server deployments don't include it; this parser tolerates both shapes
    /// and surfaces an empty list rather than failing — that way a stale
    /// server doesn't break validation, it just doesn't grant any tensor-side
    /// capabilities until it's updated.</para>
    /// </remarks>
    internal static LicenseValidationResult ParseResponse(string responseJson, int httpStatusCode)
    {
        var obj = JsonConvert.DeserializeAnonymousType(responseJson, new
        {
            valid = false,
            tier = (string?)null,
            error = (string?)null,
            message = (string?)null,
            license_id = (string?)null,
            activation_id = (string?)null,
            current_activations = (int?)null,
            max_activations = (int?)null,
            capabilities = (string[]?)null
        });

        if (obj is null)
        {
            return new LicenseValidationResult(LicenseKeyStatus.Invalid, message: "Invalid server response.");
        }

        // Map the Edge Function response to our LicenseKeyStatus enum
        LicenseKeyStatus status;
        if (obj.valid)
        {
            status = LicenseKeyStatus.Active;
        }
        else
        {
            status = obj.error switch
            {
                "invalid_key" => LicenseKeyStatus.Invalid,
                "license_expired" => LicenseKeyStatus.Expired,
                "license_revoked" => LicenseKeyStatus.Revoked,
                "license_suspended" => LicenseKeyStatus.Revoked,
                "activation_limit" => LicenseKeyStatus.SeatLimitReached,
                "server_error" => LicenseKeyStatus.ValidationPending,
                "missing_fields" => LicenseKeyStatus.Invalid,
                "method_not_allowed" => LicenseKeyStatus.Invalid,
                _ => httpStatusCode >= 500 ? LicenseKeyStatus.ValidationPending : LicenseKeyStatus.Invalid
            };
        }

        return new LicenseValidationResult(
            status,
            tier: obj.tier,
            seatsUsed: obj.current_activations ?? 0,
            seatsMax: obj.max_activations,
            validatedAt: DateTimeOffset.UtcNow,
            message: obj.message,
            capabilities: obj.capabilities);
    }
}
