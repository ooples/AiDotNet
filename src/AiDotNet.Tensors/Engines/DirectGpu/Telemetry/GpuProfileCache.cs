// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace AiDotNet.Tensors.Engines.DirectGpu.Telemetry;

/// <summary>
/// Local cache for GPU profiles to avoid repeated network requests.
/// </summary>
/// <remarks>
/// <para>
/// This cache stores optimal GPU configurations locally, reducing network
/// dependency and providing instant access to previously tuned configurations.
/// </para>
/// <para><b>Cache Location:</b></para>
/// <list type="bullet">
/// <item>Windows: %LOCALAPPDATA%\AiDotNet\gpu_profiles.json</item>
/// <item>Linux: ~/.local/share/AiDotNet/gpu_profiles.json</item>
/// <item>macOS: ~/Library/Application Support/AiDotNet/gpu_profiles.json</item>
/// </list>
/// </remarks>
public sealed class GpuProfileCache
{
    private readonly string _cacheFilePath;
    private readonly object _lock = new();
    private Dictionary<string, CachedProfile> _profiles;
    private bool _isDirty;

    /// <summary>
    /// Gets the default cache directory for the current platform.
    /// </summary>
    public static string DefaultCacheDirectory
    {
        get
        {
            // Use platform detection that works on both .NET Framework and .NET Core
            var platform = Environment.OSVersion.Platform;
            bool isWindows = platform == PlatformID.Win32NT || platform == PlatformID.Win32S ||
                             platform == PlatformID.Win32Windows || platform == PlatformID.WinCE;

            if (isWindows)
            {
                var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                return Path.Combine(localAppData, "AiDotNet");
            }
            else
            {
                var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                // Check for macOS by looking for Library folder
                var macPath = Path.Combine(home, "Library");
                if (Directory.Exists(macPath))
                {
                    return Path.Combine(home, "Library", "Application Support", "AiDotNet");
                }
                else
                {
                    // Linux and others
                    return Path.Combine(home, ".local", "share", "AiDotNet");
                }
            }
        }
    }

    /// <summary>
    /// Creates a new GPU profile cache.
    /// </summary>
    /// <param name="cacheDirectory">Custom cache directory, or null for default.</param>
    public GpuProfileCache(string? cacheDirectory = null)
    {
        var directory = cacheDirectory ?? DefaultCacheDirectory;
        _cacheFilePath = Path.Combine(directory, "gpu_profiles.json");
        _profiles = new Dictionary<string, CachedProfile>(StringComparer.OrdinalIgnoreCase);

        LoadFromDisk();
    }

    /// <summary>
    /// Gets a cached profile for the specified GPU and dimension range.
    /// </summary>
    /// <param name="gpuVendor">GPU vendor.</param>
    /// <param name="gpuModel">GPU model name.</param>
    /// <param name="minDimension">Minimum matrix dimension.</param>
    /// <param name="maxDimension">Maximum matrix dimension.</param>
    /// <returns>The cached profile, or null if not found.</returns>
    public CachedProfile? GetProfile(string gpuVendor, string gpuModel, int minDimension, int maxDimension)
    {
        var key = GenerateKey(gpuVendor, gpuModel, minDimension, maxDimension);

        lock (_lock)
        {
            if (_profiles.TryGetValue(key, out var profile))
            {
                // Check if cache is still valid (7 days)
                if (profile.CachedAt > DateTime.UtcNow.AddDays(-7))
                {
                    return profile;
                }
            }
        }

        return null;
    }

    /// <summary>
    /// Stores a profile in the cache.
    /// </summary>
    /// <param name="gpuVendor">GPU vendor.</param>
    /// <param name="gpuModel">GPU model name.</param>
    /// <param name="minDimension">Minimum matrix dimension.</param>
    /// <param name="maxDimension">Maximum matrix dimension.</param>
    /// <param name="configJson">The optimal configuration as JSON.</param>
    /// <param name="gflops">Measured performance in GFLOPS.</param>
    /// <param name="efficiencyPercent">Measured efficiency percentage.</param>
    public void SetProfile(
        string gpuVendor,
        string gpuModel,
        int minDimension,
        int maxDimension,
        string configJson,
        double gflops,
        double efficiencyPercent)
    {
        var key = GenerateKey(gpuVendor, gpuModel, minDimension, maxDimension);
        var profile = new CachedProfile
        {
            GpuVendor = gpuVendor,
            GpuModel = gpuModel,
            MinDimension = minDimension,
            MaxDimension = maxDimension,
            ConfigJson = configJson,
            MeasuredGflops = gflops,
            EfficiencyPercent = efficiencyPercent,
            CachedAt = DateTime.UtcNow
        };

        lock (_lock)
        {
            _profiles[key] = profile;
            _isDirty = true;
        }

        // Save asynchronously to not block the caller
        SaveToDiskAsync();
    }

    /// <summary>
    /// Gets all cached profiles.
    /// </summary>
    public IReadOnlyList<CachedProfile> GetAllProfiles()
    {
        lock (_lock)
        {
            return new List<CachedProfile>(_profiles.Values);
        }
    }

    /// <summary>
    /// Clears all cached profiles.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _profiles.Clear();
            _isDirty = true;
        }

        SaveToDiskAsync();
    }

    /// <summary>
    /// Forces a save to disk.
    /// </summary>
    public void Flush()
    {
        lock (_lock)
        {
            if (_isDirty)
            {
                SaveToDiskSync();
            }
        }
    }

    private void LoadFromDisk()
    {
        try
        {
            if (!File.Exists(_cacheFilePath))
            {
                return;
            }

            var json = File.ReadAllText(_cacheFilePath);
            var loaded = JsonSerializer.Deserialize<Dictionary<string, CachedProfile>>(json);

            if (loaded is not null)
            {
                lock (_lock)
                {
                    _profiles = loaded;
                }
            }
        }
        catch
        {
            // Ignore cache loading errors - start fresh
            _profiles = new Dictionary<string, CachedProfile>(StringComparer.OrdinalIgnoreCase);
        }
    }

    private void SaveToDiskAsync()
    {
        // Fire and forget save
        System.Threading.ThreadPool.QueueUserWorkItem(_ => SaveToDiskSync());
    }

    private void SaveToDiskSync()
    {
        try
        {
            Dictionary<string, CachedProfile> snapshot;
            lock (_lock)
            {
                if (!_isDirty)
                {
                    return;
                }

                snapshot = new Dictionary<string, CachedProfile>(_profiles, StringComparer.OrdinalIgnoreCase);
                _isDirty = false;
            }

            var directory = Path.GetDirectoryName(_cacheFilePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(snapshot, options);
            File.WriteAllText(_cacheFilePath, json);
        }
        catch
        {
            // Ignore save errors - the cache is in memory
        }
    }

    private static string GenerateKey(string vendor, string model, int minDim, int maxDim)
    {
        return $"{vendor}|{model}|{minDim}-{maxDim}";
    }
}

/// <summary>
/// A cached GPU profile entry.
/// </summary>
public sealed class CachedProfile
{
    public string GpuVendor { get; init; } = string.Empty;
    public string GpuModel { get; init; } = string.Empty;
    public int MinDimension { get; init; }
    public int MaxDimension { get; init; }
    public string ConfigJson { get; init; } = string.Empty;
    public double MeasuredGflops { get; init; }
    public double EfficiencyPercent { get; init; }
    public DateTime CachedAt { get; init; }
}
