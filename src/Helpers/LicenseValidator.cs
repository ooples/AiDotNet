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
        // Offline-only mode: when ServerUrl is explicitly set to empty string
        if (_licenseKey.ServerUrl is not null && _licenseKey.ServerUrl.Trim().Length == 0)
        {
            var offlineResult = ValidateOffline();
            lock (_cacheLock)
            {
                _cached = offlineResult;
            }

            return offlineResult;
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

        // When an official build key is available, verify the license key's HMAC signature.
        // The key is expected to be in the format: payload.signature (base64url-encoded).
        var buildKey = BuildKeyProvider.GetBuildKey();
        if (buildKey.Length > 0)
        {
            var dotIndex = _licenseKey.Key.LastIndexOf('.');
            if (dotIndex > 0 && dotIndex < _licenseKey.Key.Length - 1)
            {
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
            }
            // Keys without a dot separator are accepted for backwards compatibility
            // when no server URL is configured (legacy keys).
        }

        return new LicenseValidationResult(LicenseKeyStatus.Active, message: "Offline-only mode.");
    }

    /// <summary>
    /// Validates that the license key has the expected format: aidn.{id}.{signature}
    /// where both id and signature are non-empty alphanumeric strings.
    /// </summary>
    private static bool ValidateKeyFormat(string key)
    {
        var parts = key.Split('.');
        return parts.Length == 3 &&
               parts[0] == "aidn" &&
               parts[1].Length > 0 &&
               parts[2].Length > 0;
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
    /// </summary>
    private Dictionary<string, string?> BuildRequestBody()
    {
        var body = new Dictionary<string, string?>
        {
            ["license_key"] = _licenseKey.Key,
            ["machine_id_hash"] = GetMachineIdHash()
        };

        if (_licenseKey.EnableTelemetry)
        {
            try { body["hostname"] = System.Environment.MachineName; } catch { /* best effort */ }
            try
            {
                body["os_description"] = System.Runtime.InteropServices.RuntimeInformation.OSDescription;
            }
            catch { /* best effort */ }
        }

        return body;
    }

    /// <summary>
    /// Generates a one-way SHA-256 hash of the machine fingerprint for license activation tracking.
    /// Uses a different salt than telemetry to prevent correlation between the two systems.
    /// </summary>
    private static string GetMachineIdHash()
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
        // Offline-only mode: when ServerUrl is explicitly set to empty string
        if (_licenseKey.ServerUrl is not null && _licenseKey.ServerUrl.Trim().Length == 0)
        {
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
    /// license_id?: string, activation_id?: string, current_activations?: int, max_activations?: int }
    /// </summary>
    private static LicenseValidationResult ParseResponse(string responseJson, int httpStatusCode)
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
            max_activations = (int?)null
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
            message: obj.message);
    }
}
