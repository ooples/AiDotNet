// Copyright (c) AiDotNet. All rights reserved.
// Persistent storage for GEMM tuning results.

using System;
using System.Collections.Generic;
using System.IO;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// Persistent storage for GEMM tuning results.
/// Saves best configurations to disk for reuse across application restarts.
/// </summary>
internal sealed class GemmTuningDatabase : IDisposable
{
    private readonly string _databasePath;
    private readonly Dictionary<string, (GemmConfig Config, double GFlops)> _cache;
    private readonly object _lock = new();
    private bool _isDirty;
    private bool _disposed;

    public GemmTuningDatabase(string? customPath = null)
    {
        _cache = new Dictionary<string, (GemmConfig, double)>();

        // Use app data folder for persistence
        var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var aiDotNetPath = Path.Combine(appDataPath, "AiDotNet", "GpuTuning");
        Directory.CreateDirectory(aiDotNetPath);

        _databasePath = customPath ?? Path.Combine(aiDotNetPath, "gemm_tuning.json");

        LoadFromDisk();
    }

    /// <summary>
    /// Gets the cached configuration for a given matrix size, if available.
    /// </summary>
    public GemmConfig? GetBestConfig(int M, int N, int K)
    {
        lock (_lock)
        {
            var key = FormattableString.Invariant($"{M}x{N}x{K}");
            if (_cache.TryGetValue(key, out var entry))
                return entry.Config;
            return null;
        }
    }

    /// <summary>
    /// Stores a tuning result if it's better than the current best.
    /// </summary>
    public void StoreResult(int M, int N, int K, GemmConfig config, double gflops)
    {
        lock (_lock)
        {
            var key = FormattableString.Invariant($"{M}x{N}x{K}");
            if (!_cache.TryGetValue(key, out var existing) || gflops > existing.GFlops)
            {
                _cache[key] = (config, gflops);
                _isDirty = true;
            }
        }
    }

    /// <summary>
    /// Gets all stored tuning results for analysis.
    /// </summary>
    public IReadOnlyDictionary<string, (GemmConfig Config, double GFlops)> GetAllResults()
    {
        lock (_lock)
        {
            return new Dictionary<string, (GemmConfig, double)>(_cache);
        }
    }

    /// <summary>
    /// Persists any changes to disk.
    /// </summary>
    public void Save()
    {
        lock (_lock)
        {
            if (!_isDirty) return;
            SaveToDisk();
            _isDirty = false;
        }
    }

    private void LoadFromDisk()
    {
        try
        {
            if (!File.Exists(_databasePath)) return;

            var json = File.ReadAllText(_databasePath);
            var entries = ParseJsonDatabase(json);
            foreach (var (key, config, gflops) in entries)
            {
                _cache[key] = (config, gflops);
            }
        }
        catch (Exception)
        {
            // Silently ignore load failures - start fresh
        }
    }

    private void SaveToDisk()
    {
        try
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("{");

            bool first = true;
            foreach (var kvp in _cache)
            {
                if (!first) sb.AppendLine(",");
                first = false;

                var key = kvp.Key;
                var config = kvp.Value.Config;
                var gflops = kvp.Value.GFlops;

                sb.Append(FormattableString.Invariant($"  \"{key}\": {{"));
                sb.Append(FormattableString.Invariant($"\"TileM\": {config.TileM}, "));
                sb.Append(FormattableString.Invariant($"\"TileN\": {config.TileN}, "));
                sb.Append(FormattableString.Invariant($"\"TileK\": {config.TileK}, "));
                sb.Append(FormattableString.Invariant($"\"ThreadTileM\": {config.ThreadTileM}, "));
                sb.Append(FormattableString.Invariant($"\"ThreadTileN\": {config.ThreadTileN}, "));
                sb.Append(FormattableString.Invariant($"\"VectorWidthM\": {config.VectorWidthM}, "));
                sb.Append(FormattableString.Invariant($"\"VectorWidthN\": {config.VectorWidthN}, "));
                sb.Append(FormattableString.Invariant($"\"UseDoubleBuffering\": {config.UseDoubleBuffering.ToString().ToLower()}, "));
                sb.Append(FormattableString.Invariant($"\"UseVectorizedLoads\": {config.UseVectorizedLoads.ToString().ToLower()}, "));
                sb.Append(FormattableString.Invariant($"\"KernelName\": \"{config.KernelName}\", "));
                sb.Append(FormattableString.Invariant($"\"GFlops\": {gflops:F2}}}"));
            }

            sb.AppendLine();
            sb.AppendLine("}");

            File.WriteAllText(_databasePath, sb.ToString());
        }
        catch (Exception)
        {
            // Silently ignore save failures
        }
    }

    private static IEnumerable<(string Key, GemmConfig Config, double GFlops)> ParseJsonDatabase(string json)
    {
        var results = new List<(string, GemmConfig, double)>();

        // Simple regex-based JSON parsing for our specific format
        var entryPattern = @"""([^""]+)""\s*:\s*\{([^}]+)\}";
        var matches = System.Text.RegularExpressions.Regex.Matches(json, entryPattern,
            System.Text.RegularExpressions.RegexOptions.None, TimeSpan.FromSeconds(5));

        foreach (System.Text.RegularExpressions.Match match in matches)
        {
            try
            {
                var key = match.Groups[1].Value;
                var content = match.Groups[2].Value;

                int GetIntValue(string name)
                {
                    var pattern = string.Concat("\"", name, "\"\\s*:\\s*(\\d+)");
                    var m = System.Text.RegularExpressions.Regex.Match(content, pattern,
                        System.Text.RegularExpressions.RegexOptions.None, TimeSpan.FromSeconds(1));
                    return m.Success ? int.Parse(m.Groups[1].Value) : 0;
                }

                bool GetBoolValue(string name)
                {
                    var pattern = string.Concat("\"", name, "\"\\s*:\\s*(true|false)");
                    var m = System.Text.RegularExpressions.Regex.Match(content, pattern,
                        System.Text.RegularExpressions.RegexOptions.IgnoreCase, TimeSpan.FromSeconds(1));
                    return m.Success && m.Groups[1].Value.Equals("true", StringComparison.OrdinalIgnoreCase);
                }

                string GetStringValue(string name)
                {
                    var pattern = string.Concat("\"", name, "\"\\s*:\\s*\"([^\"]*)\"");
                    var m = System.Text.RegularExpressions.Regex.Match(content, pattern,
                        System.Text.RegularExpressions.RegexOptions.None, TimeSpan.FromSeconds(1));
                    return m.Success ? m.Groups[1].Value : "";
                }

                double GetDoubleValue(string name)
                {
                    var pattern = string.Concat("\"", name, "\"\\s*:\\s*([\\d.]+)");
                    var m = System.Text.RegularExpressions.Regex.Match(content, pattern,
                        System.Text.RegularExpressions.RegexOptions.None, TimeSpan.FromSeconds(1));
                    return m.Success ? double.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture) : 0;
                }

                var config = new GemmConfig
                {
                    TileM = GetIntValue("TileM"),
                    TileN = GetIntValue("TileN"),
                    TileK = GetIntValue("TileK"),
                    ThreadTileM = GetIntValue("ThreadTileM"),
                    ThreadTileN = GetIntValue("ThreadTileN"),
                    VectorWidthM = GetIntValue("VectorWidthM"),
                    VectorWidthN = GetIntValue("VectorWidthN"),
                    UseDoubleBuffering = GetBoolValue("UseDoubleBuffering"),
                    UseVectorizedLoads = GetBoolValue("UseVectorizedLoads"),
                    KernelName = GetStringValue("KernelName")
                };

                var gflops = GetDoubleValue("GFlops");

                if (config.TileM > 0 && config.TileN > 0)
                {
                    results.Add((key, config, gflops));
                }
            }
            catch
            {
                // Skip malformed entries
            }
        }

        return results;
    }

    public void Dispose()
    {
        if (_disposed) return;
        Save();
        _disposed = true;
    }
}
