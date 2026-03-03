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
        // Offline-only mode: no server URL means always active
        if (string.IsNullOrWhiteSpace(_licenseKey.ServerUrl))
        {
            var offlineResult = new LicenseValidationResult(LicenseKeyStatus.Active, message: "Offline-only mode.");
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

            // No valid cache — return ValidationPending (allow use, don't block)
            var pending = new LicenseValidationResult(
                LicenseKeyStatus.ValidationPending,
                message: "License server unreachable. Operating in grace mode.");

            lock (_cacheLock)
            {
                _cached = pending;
            }

            return pending;
        }
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
        using var httpClient = new System.Net.Http.HttpClient { Timeout = TimeSpan.FromSeconds(15) };
        var content = new System.Net.Http.StringContent(json, Encoding.UTF8, "application/json");
        var response = httpClient.PostAsync(url, content).ConfigureAwait(false).GetAwaiter().GetResult();
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
            message = (string?)null
        });

        if (obj is null)
        {
            return new LicenseValidationResult(LicenseKeyStatus.Invalid, message: "Invalid server response.");
        }

        if (!Enum.TryParse<LicenseKeyStatus>(obj.status, ignoreCase: true, out var status))
        {
            status = LicenseKeyStatus.Invalid;
        }

        return new LicenseValidationResult(
            status,
            tier: obj.tier,
            expiresAt: obj.expiresAt,
            seatsUsed: obj.seatsUsed,
            seatsMax: obj.seatsMax,
            validatedAt: DateTimeOffset.UtcNow,
            message: obj.message);
    }
}
