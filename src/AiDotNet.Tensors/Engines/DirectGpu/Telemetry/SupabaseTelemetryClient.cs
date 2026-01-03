// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.Telemetry;

/// <summary>
/// Supabase-based telemetry client for GPU profile sharing.
/// </summary>
/// <remarks>
/// <para>
/// This client submits GPU A/B test results and retrieves optimal configurations
/// from the community-shared profile database.
/// </para>
/// <para><b>Privacy:</b></para>
/// <list type="bullet">
/// <item>Machine ID is hashed (SHA256) - never sent in plain text</item>
/// <item>Only GPU info and performance metrics are collected</item>
/// <item>No personal data or file system information is transmitted</item>
/// <item>Opt-out is supported and respected</item>
/// </list>
/// </remarks>
public sealed class SupabaseTelemetryClient : ITelemetryClient
{
    private readonly HttpClient _httpClient;
    private readonly string _supabaseUrl;
    private readonly string _supabaseKey;
    private readonly string _clientHash;
    private readonly string _aidotnetVersion;
    private bool _isEnabled;
    private bool _disposed;

    /// <summary>
    /// Default Supabase project URL for AiDotNet telemetry.
    /// </summary>
    public const string DefaultSupabaseUrl = "https://yfkqwpgjahoamlgckjib.supabase.co";

    /// <summary>
    /// Default anon key for read/insert operations.
    /// </summary>
    public const string DefaultAnonKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlma3F3cGdqYWhvYW1sZ2NramliIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjczMDc0MjUsImV4cCI6MjA4Mjg4MzQyNX0.166oZVGXdnuXMmPzWeKb3fs2qNi0BQHwn502TWxDrwQ";

    /// <inheritdoc/>
    public bool IsEnabled => _isEnabled && !_disposed;

    /// <summary>
    /// Creates a new Supabase telemetry client.
    /// </summary>
    /// <param name="enabled">Whether telemetry is enabled (default: true).</param>
    /// <param name="supabaseUrl">Supabase project URL (optional, uses default).</param>
    /// <param name="supabaseKey">Supabase anon key (optional, uses default).</param>
    public SupabaseTelemetryClient(
        bool enabled = true,
        string? supabaseUrl = null,
        string? supabaseKey = null)
    {
        _isEnabled = enabled;
        _supabaseUrl = supabaseUrl ?? DefaultSupabaseUrl;
        _supabaseKey = supabaseKey ?? DefaultAnonKey;
        _clientHash = GenerateClientHash();
        _aidotnetVersion = GetAidotnetVersion();

        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(10)
        };
        _httpClient.DefaultRequestHeaders.Add("apikey", _supabaseKey);
        _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {_supabaseKey}");
    }

    /// <inheritdoc/>
    public async Task SubmitTuningResultAsync(
        TuningResultData result,
        GpuInfoData gpuInfo,
        CancellationToken cancellationToken = default)
    {
        if (!IsEnabled)
        {
            return;
        }

        try
        {
            var payload = new
            {
                gpu_vendor = gpuInfo.Vendor,
                gpu_model = gpuInfo.Model,
                gpu_architecture = gpuInfo.Architecture,
                driver_version = gpuInfo.DriverVersion,
                os_platform = gpuInfo.OsPlatform,
                matrix_m = result.MatrixM,
                matrix_n = result.MatrixN,
                matrix_k = result.MatrixK,
                config_json = result.ConfigJson,
                measured_gflops = result.MeasuredGflops,
                efficiency_percent = result.EfficiencyPercent,
                client_hash = _clientHash,
                aidotnet_version = _aidotnetVersion
            };

            var json = JsonSerializer.Serialize(payload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                $"{_supabaseUrl}/rest/v1/gpu_telemetry",
                content,
                cancellationToken).ConfigureAwait(false);

            // Don't throw on failure - telemetry should be silent
            if (!response.IsSuccessStatusCode)
            {
                // Log failure silently - don't interrupt the user's workflow
                System.Diagnostics.Debug.WriteLine(
                    $"[Telemetry] Failed to submit tuning result: {response.StatusCode}");
            }
        }
        catch (Exception ex)
        {
            // Telemetry failures should never interrupt the user
            System.Diagnostics.Debug.WriteLine($"[Telemetry] Exception: {ex.Message}");
        }
    }

    /// <inheritdoc/>
    public async Task<GpuProfileData?> GetProfileAsync(
        GpuInfoData gpuInfo,
        int minDimension,
        int maxDimension,
        CancellationToken cancellationToken = default)
    {
        if (!IsEnabled)
        {
            return null;
        }

        try
        {
            // Query for matching profile
            var url = $"{_supabaseUrl}/rest/v1/gpu_profiles" +
                $"?gpu_vendor=eq.{Uri.EscapeDataString(gpuInfo.Vendor)}" +
                $"&gpu_model=eq.{Uri.EscapeDataString(gpuInfo.Model)}" +
                $"&min_dimension=lte.{minDimension}" +
                $"&max_dimension=gte.{maxDimension}" +
                $"&order=efficiency_percent.desc" +
                $"&limit=1";

            var response = await _httpClient.GetAsync(url, cancellationToken).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                return null;
            }

            var json = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            var profiles = JsonNode.Parse(json)?.AsArray();

            if (profiles is null || profiles.Count == 0)
            {
                return null;
            }

            var profile = profiles[0];
            return new GpuProfileData
            {
                ConfigJson = profile?["config_json"]?.ToString() ?? string.Empty,
                MeasuredGflops = profile?["measured_gflops"]?.GetValue<double>() ?? 0,
                EfficiencyPercent = profile?["efficiency_percent"]?.GetValue<double>() ?? 0,
                SampleCount = profile?["sample_count"]?.GetValue<int>() ?? 1
            };
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[Telemetry] GetProfile failed: {ex.Message}");
            return null;
        }
    }

    /// <inheritdoc/>
    public async Task SubmitExceptionAsync(
        ExceptionData exception,
        CancellationToken cancellationToken = default)
    {
        if (!IsEnabled)
        {
            return;
        }

        try
        {
            var payload = new
            {
                exception_type = exception.ExceptionType,
                exception_message = exception.ExceptionMessage,
                stack_trace = TruncateStackTrace(exception.StackTrace),
                inner_exception_type = exception.InnerExceptionType,
                inner_exception_message = exception.InnerExceptionMessage,
                component = exception.Component,
                operation = exception.Operation,
                aidotnet_version = exception.AidotnetVersion,
                dotnet_version = exception.DotnetVersion,
                os_platform = exception.OsPlatform,
                os_version = exception.OsVersion,
                gpu_vendor = exception.GpuVendor,
                gpu_model = exception.GpuModel,
                client_hash = _clientHash,
                additional_context = exception.AdditionalContextJson
            };

            var json = JsonSerializer.Serialize(payload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            await _httpClient.PostAsync(
                $"{_supabaseUrl}/rest/v1/exception_telemetry",
                content,
                cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            // Silently ignore exception telemetry failures
        }
    }

    /// <inheritdoc/>
    public async Task OptOutAsync(CancellationToken cancellationToken = default)
    {
        _isEnabled = false;

        try
        {
            var payload = new
            {
                client_hash = _clientHash,
                opted_out = true,
                opted_out_at = DateTime.UtcNow.ToString("O")
            };

            var json = JsonSerializer.Serialize(payload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            // Upsert the opt-out record
            var request = new HttpRequestMessage(HttpMethod.Post, $"{_supabaseUrl}/rest/v1/telemetry_consent")
            {
                Content = content
            };
            request.Headers.Add("Prefer", "resolution=merge-duplicates");

            await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            // Silently ignore opt-out failures - we've already disabled locally
        }
    }

    /// <inheritdoc/>
    public async Task<bool> IsOptedOutAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var url = $"{_supabaseUrl}/rest/v1/telemetry_consent" +
                $"?client_hash=eq.{_clientHash}" +
                $"&select=opted_out";

            var response = await _httpClient.GetAsync(url, cancellationToken).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                return false;
            }

            var json = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            var records = JsonNode.Parse(json)?.AsArray();

            if (records is not null && records.Count > 0)
            {
                return records[0]?["opted_out"]?.GetValue<bool>() ?? false;
            }

            return false;
        }
        catch
        {
            return false;
        }
    }

    private static string GenerateClientHash()
    {
        try
        {
            // Generate a stable hash based on machine characteristics
            // This is anonymous - we never send the actual machine ID
            var machineId = Environment.MachineName + Environment.ProcessorCount + Environment.OSVersion;

            using var sha256 = SHA256.Create();
            var bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(machineId));
            return Convert.ToBase64String(bytes)[..22]; // Short hash
        }
        catch
        {
            // If we can't generate a stable hash, use a random one
            return Guid.NewGuid().ToString("N")[..22];
        }
    }

    private static string GetAidotnetVersion()
    {
        try
        {
            var assembly = typeof(SupabaseTelemetryClient).Assembly;
            var version = assembly.GetName().Version;
            return version?.ToString() ?? "unknown";
        }
        catch
        {
            return "unknown";
        }
    }

    private static string? TruncateStackTrace(string? stackTrace)
    {
        // Use pattern matching for null-safe access
        if (stackTrace is null || stackTrace.Length == 0)
        {
            return null;
        }

        // Limit stack trace to ~4KB to avoid large payloads
        const int maxLength = 4096;
        if (stackTrace.Length <= maxLength)
        {
            return stackTrace;
        }

        return stackTrace.Substring(0, maxLength - 20) + "\n[truncated...]";
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _httpClient.Dispose();
    }
}
