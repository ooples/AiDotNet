using System.Text;
using AiDotNet.Enums;
using AiDotNet.Models;
using Newtonsoft.Json;

namespace AiDotNet.Helpers;

/// <summary>
/// Client-side license validator that contacts an optional license server
/// and caches the result for an offline grace period.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a license has a <see cref="AiDotNetLicenseKey.ServerUrl"/>,
/// this validator will contact the server to check whether the key is valid, how many seats are
/// used, and whether it has expired. If the server is unreachable, the validator uses its cached
/// result until the offline grace period expires.</para>
///
/// <para>When no server URL is configured (offline-only mode), the validator always returns
/// <see cref="LicenseKeyStatus.Active"/>.</para>
/// </remarks>
internal sealed class LicenseValidator
{
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
    /// Validates the license key, contacting the server if configured.
    /// </summary>
    /// <returns>A <see cref="LicenseValidationResult"/> describing the current key status.</returns>
    public LicenseValidationResult Validate()
    {
        // Offline-only mode: validate key format before accepting
        if (string.IsNullOrWhiteSpace(_licenseKey.ServerUrl))
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
        catch
        {
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

        // Reject keys that are too short to be valid (minimum 16 characters)
        if (_licenseKey.Key.Length < 16)
        {
            return new LicenseValidationResult(
                LicenseKeyStatus.Invalid,
                message: "License key format is invalid (too short).");
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
                    byte[] expectedSignature = Convert.FromBase64String(signaturePart);

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
        string serverUrl = _licenseKey.ServerUrl ?? string.Empty;
        string url = serverUrl.TrimEnd('/') + "/api/licenses/validate";

        var requestBody = new
        {
            key = _licenseKey.Key,
            machineId = _licenseKey.EnableTelemetry ? MachineFingerprint.GetMachineId() : null as string,
            machineName = _licenseKey.EnableTelemetry ? System.Environment.MachineName : null as string,
            environment = _licenseKey.Environment
        };

        string json = JsonConvert.SerializeObject(requestBody);

#if NET471
        return ValidateOnlineNet471(url, json);
#else
        return ValidateOnlineModern(url, json);
#endif
    }

#if NET471
    private LicenseValidationResult ValidateOnlineNet471(string url, string json)
    {
        using var client = new System.Net.WebClient();
        client.Headers[System.Net.HttpRequestHeader.ContentType] = "application/json";
        string responseJson = client.UploadString(url, "POST", json);
        return ParseResponse(responseJson);
    }
#else
    private LicenseValidationResult ValidateOnlineModern(string url, string json)
    {
        var content = new System.Net.Http.StringContent(json, Encoding.UTF8, "application/json");
        var response = SharedHttpClient.PostAsync(url, content).ConfigureAwait(false).GetAwaiter().GetResult();
        string responseJson = response.Content.ReadAsStringAsync().ConfigureAwait(false).GetAwaiter().GetResult();

        if (!response.IsSuccessStatusCode)
        {
            return new LicenseValidationResult(
                LicenseKeyStatus.Invalid,
                message: $"Server returned HTTP {(int)response.StatusCode}.");
        }

        return ParseResponse(responseJson);
    }
#endif

    private static LicenseValidationResult ParseResponse(string responseJson)
    {
        var obj = JsonConvert.DeserializeAnonymousType(responseJson, new
        {
            status = string.Empty,
            tier = (string?)null,
            expiresAt = (DateTimeOffset?)null,
            seatsUsed = 0,
            seatsMax = (int?)null,
            message = (string?)null,
            decryptionToken = (string?)null
        });

        if (obj is null)
        {
            return new LicenseValidationResult(LicenseKeyStatus.Invalid, message: "Invalid server response.");
        }

        if (!Enum.TryParse<LicenseKeyStatus>(obj.status, ignoreCase: true, out var status))
        {
            status = LicenseKeyStatus.Invalid;
        }

        // Parse decryption token from base64
        byte[]? tokenBytes = null;
        if (!string.IsNullOrWhiteSpace(obj.decryptionToken))
        {
            try
            {
                byte[] decoded = Convert.FromBase64String(obj.decryptionToken);
                // Validate minimum key length (at least 16 bytes / 128 bits)
                if (decoded.Length >= 16)
                {
                    tokenBytes = decoded;
                }
            }
            catch (FormatException)
            {
                // Ignore malformed token — will use null
            }
        }

        return new LicenseValidationResult(
            status,
            tier: obj.tier,
            expiresAt: obj.expiresAt,
            seatsUsed: obj.seatsUsed,
            seatsMax: obj.seatsMax,
            validatedAt: DateTimeOffset.UtcNow,
            message: obj.message,
            decryptionToken: tokenBytes);
    }
}
