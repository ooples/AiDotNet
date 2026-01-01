// Copyright (c) AiDotNet. All rights reserved.
// Persistent storage for GEMM tuning results.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// Persistent storage for GEMM tuning results.
/// Saves best configurations to disk for reuse across application restarts.
/// Also tracks ALL tested configurations to avoid duplicate work in Bayesian optimization.
/// </summary>
internal sealed class GemmTuningDatabase : IDisposable
{
    private readonly string _signature;
    private readonly string _databasePath;
    private readonly string _historyPath;  // Tracks ALL tested configs, not just best
    private readonly Dictionary<string, (GemmConfig Config, double GFlops)> _cache;
    private readonly HashSet<string> _testedConfigs;  // ConfigKey -> already tested
    private readonly Dictionary<string, double> _testedGflops;
    private readonly Dictionary<string, int> _matrixKeyCounts;
    private readonly object _lock = new();
    private bool _isDirty;
    private bool _disposed;

    public GemmTuningDatabase(string? customPath = null, string? deviceSignature = null)
    {
        _cache = new Dictionary<string, (GemmConfig, double)>();
        _testedConfigs = new HashSet<string>();
        _testedGflops = new Dictionary<string, double>();
        _matrixKeyCounts = new Dictionary<string, int>();
        _signature = NormalizeSignature(deviceSignature);

        // Use app data folder for persistence
        var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var aiDotNetPath = Path.Combine(appDataPath, "AiDotNet", "GpuTuning");
        Directory.CreateDirectory(aiDotNetPath);

        string suffix = string.IsNullOrEmpty(_signature) ? string.Empty : $"_{_signature}";
        string databaseFileName = $"gemm_tuning{suffix}.json";
        string historyFileName = $"gemm_history{suffix}.txt";
        string? normalizedCustomPath = NormalizeCustomPath(customPath, databaseFileName);

        if (string.IsNullOrEmpty(normalizedCustomPath))
        {
            _databasePath = Path.Combine(aiDotNetPath, databaseFileName);
            _historyPath = Path.Combine(aiDotNetPath, historyFileName);
        }
        else
        {
            _databasePath = normalizedCustomPath;
            var historyDir = Path.GetDirectoryName(_databasePath);
            if (string.IsNullOrWhiteSpace(historyDir))
                historyDir = aiDotNetPath;
            _historyPath = Path.Combine(historyDir, historyFileName);
        }

        EnsureDirectoryExists(_databasePath);
        EnsureDirectoryExists(_historyPath);

        LoadFromDisk();
        LoadHistoryFromDisk();
    }

    /// <summary>
    /// Checks if a specific configuration has already been tested for a matrix size.
    /// Used by Bayesian optimization to avoid re-testing configurations.
    /// </summary>
    public bool HasBeenTested(int M, int N, int K, GemmConfig config)
    {
        lock (_lock)
        {
            var historyKey = BuildHistoryKey(M, N, K, config);
            return _testedConfigs.Contains(historyKey);
        }
    }

    /// <summary>
    /// Gets the cached GFLOPS for a specific configuration, if it was tested before.
    /// Returns null if the config hasn't been tested yet.
    /// </summary>
    public double? GetCachedGflops(int M, int N, int K, GemmConfig config)
    {
        lock (_lock)
        {
            var historyKey = BuildHistoryKey(M, N, K, config);
            if (_testedGflops.TryGetValue(historyKey, out var gflops))
                return gflops;
            return null;
        }
    }

    /// <summary>
    /// Marks a configuration as tested (even if it failed or wasn't the best).
    /// This is crucial for Bayesian optimization efficiency.
    /// </summary>
    public void MarkAsTested(int M, int N, int K, GemmConfig config, double gflops)
    {
        lock (_lock)
        {
            var historyKey = BuildHistoryKey(M, N, K, config);
            if (_testedConfigs.Add(historyKey))
                IncrementMatrixCount(BuildMatrixKey(M, N, K));

            _testedGflops[historyKey] = NormalizeGflops(gflops);
            _isDirty = true;
        }
    }

    /// <summary>
    /// Gets the count of configurations already tested for a specific matrix size.
    /// </summary>
    public int GetTestedCount(int M, int N, int K)
    {
        lock (_lock)
        {
            var key = BuildMatrixKey(M, N, K);
            return _matrixKeyCounts.TryGetValue(key, out var count) ? count : 0;
        }
    }

    /// <summary>
    /// Gets the cached configuration for a given matrix size, if available.
    /// </summary>
    public GemmConfig? GetBestConfig(int M, int N, int K)
    {
        lock (_lock)
        {
            var key = BuildMatrixKey(M, N, K);
            if (_cache.TryGetValue(key, out var entry))
                return entry.Config;
            return null;
        }
    }

    /// <summary>
    /// Gets the cached configuration AND its stored GFLOPS for a given matrix size.
    /// Returns null if no cached entry exists.
    /// </summary>
    public (GemmConfig Config, double GFlops)? GetBestConfigWithGflops(int M, int N, int K)
    {
        lock (_lock)
        {
            var key = BuildMatrixKey(M, N, K);
            if (_cache.TryGetValue(key, out var entry))
                return (entry.Config, entry.GFlops);
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
            if (!IsFiniteGflops(gflops) || gflops < 0)
                return;

            var key = BuildMatrixKey(M, N, K);
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
    /// Persists any changes to disk (including test history).
    /// </summary>
    public void Save()
    {
        lock (_lock)
        {
            if (!_isDirty) return;
            SaveToDisk();
            SaveHistoryToDisk();
            _isDirty = false;
        }
    }

    private void LoadFromDisk()
    {
        try
        {
            if (string.IsNullOrWhiteSpace(_databasePath) || !File.Exists(_databasePath))
                return;

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

    private void LoadHistoryFromDisk()
    {
        try
        {
            if (string.IsNullOrWhiteSpace(_historyPath) || !File.Exists(_historyPath))
                return;

            foreach (var line in File.ReadLines(_historyPath))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var parsed = ParseHistoryLine(line);
                if (parsed == null)
                    continue;

                if (_testedConfigs.Add(parsed.Value.Key))
                {
                    var matrixKey = ExtractMatrixKey(parsed.Value.Key);
                    if (!string.IsNullOrEmpty(matrixKey))
                        IncrementMatrixCount(matrixKey);
                }

                _testedGflops[parsed.Value.Key] = NormalizeGflops(parsed.Value.Gflops);
            }
        }
        catch (Exception)
        {
            // Silently ignore load failures
        }
    }

    /// <summary>
    /// Clears all cached results and test history, and removes persisted files.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _cache.Clear();
            _testedConfigs.Clear();
            _testedGflops.Clear();
            _matrixKeyCounts.Clear();
            _isDirty = true;
        }

        try
        {
            if (File.Exists(_databasePath))
                File.Delete(_databasePath);
        }
        catch (Exception)
        {
            // Silently ignore delete failures
        }

        try
        {
            if (File.Exists(_historyPath))
                File.Delete(_historyPath);
        }
        catch (Exception)
        {
            // Silently ignore delete failures
        }
    }

    private void SaveHistoryToDisk()
    {
        try
        {
            if (string.IsNullOrWhiteSpace(_historyPath))
                return;

            EnsureDirectoryExists(_historyPath);
            var lines = new List<string>(_testedConfigs.Count);
            foreach (var key in _testedConfigs)
            {
                if (_testedGflops.TryGetValue(key, out var gflops))
                {
                    lines.Add(FormattableString.Invariant($"{key}\t{gflops}"));
                }
                else
                {
                    lines.Add(key);
                }
            }

            File.WriteAllLines(_historyPath, lines);
        }
        catch (Exception)
        {
            // Silently ignore save failures
        }
    }

    private string BuildMatrixKey(int M, int N, int K)
    {
        var baseKey = FormattableString.Invariant($"{M}x{N}x{K}");
        return string.IsNullOrEmpty(_signature) ? baseKey : $"{_signature}|{baseKey}";
    }

    private string BuildHistoryKey(int M, int N, int K, GemmConfig config)
    {
        return FormattableString.Invariant($"{BuildMatrixKey(M, N, K)}|{config.ToKey()}");
    }

    private static string NormalizeSignature(string? signature)
    {
        if (string.IsNullOrWhiteSpace(signature))
            return string.Empty;

        var sb = new StringBuilder(signature.Length);
        foreach (var ch in signature)
        {
            if (ch <= 0x7F && (char.IsLetterOrDigit(ch) || ch == '-' || ch == '_'))
                sb.Append(ch);
            else
                sb.Append('_');
        }

        var normalized = sb.ToString().Trim('_');
        return normalized.Length > 80 ? normalized[..80] : normalized;
    }

    private static string? NormalizeCustomPath(string? customPath, string defaultFileName)
    {
        if (string.IsNullOrWhiteSpace(customPath))
            return null;

        try
        {
            var fullPath = Path.GetFullPath(customPath);
            if (!Path.HasExtension(fullPath))
                fullPath = Path.Combine(fullPath, defaultFileName);
            return fullPath;
        }
        catch
        {
            return null;
        }
    }

    private static void EnsureDirectoryExists(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return;

        var directory = Path.GetDirectoryName(path);
        if (string.IsNullOrWhiteSpace(directory))
            return;

        Directory.CreateDirectory(directory);
    }

    private static bool IsFiniteGflops(double gflops)
    {
        return !double.IsNaN(gflops) && !double.IsInfinity(gflops);
    }

    private static double NormalizeGflops(double gflops)
    {
        if (!IsFiniteGflops(gflops) || gflops < 0)
            return 0.0;
        return gflops;
    }

    private void IncrementMatrixCount(string matrixKey)
    {
        if (_matrixKeyCounts.TryGetValue(matrixKey, out var count))
            _matrixKeyCounts[matrixKey] = count + 1;
        else
            _matrixKeyCounts[matrixKey] = 1;
    }

    private static string? ExtractMatrixKey(string historyKey)
    {
        if (string.IsNullOrWhiteSpace(historyKey))
            return null;

        int lastSeparator = historyKey.LastIndexOf('|');
        if (lastSeparator <= 0)
            return null;

        return historyKey[..lastSeparator];
    }

    private static (string Key, double Gflops)? ParseHistoryLine(string line)
    {
        var trimmed = line.Trim();
        if (trimmed.Length == 0)
            return null;

        var parts = trimmed.Split('\t');
        var key = parts[0].Trim();
        if (key.Length == 0)
            return null;

        if (parts.Length > 1 &&
            double.TryParse(parts[1].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var gflops))
        {
            return (key, gflops);
        }

        return (key, 0.0);
    }

    private void SaveToDisk()
    {
        try
        {
            if (string.IsNullOrWhiteSpace(_databasePath))
                return;

            EnsureDirectoryExists(_databasePath);
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
                // CLBlast-style parameters
                sb.Append(FormattableString.Invariant($"\"KReg\": {config.KReg}, "));
                sb.Append(FormattableString.Invariant($"\"KUnroll\": {config.KUnroll}, "));
                sb.Append(FormattableString.Invariant($"\"UseSubgroupOps\": {config.UseSubgroupOps.ToString().ToLower()}, "));
                sb.Append(FormattableString.Invariant($"\"StrideM\": {config.StrideM.ToString().ToLower()}, "));
                sb.Append(FormattableString.Invariant($"\"StrideN\": {config.StrideN.ToString().ToLower()}, "));
                sb.Append(FormattableString.Invariant($"\"CacheA\": {config.CacheA.ToString().ToLower()}, "));
                sb.Append(FormattableString.Invariant($"\"CacheB\": {config.CacheB.ToString().ToLower()}, "));
                sb.Append(FormattableString.Invariant($"\"MdimaSize\": {config.MdimaSize}, "));
                sb.Append(FormattableString.Invariant($"\"NdimbSize\": {config.NdimbSize}, "));
                sb.Append(FormattableString.Invariant($"\"KernelName\": \"{config.KernelName}\", "));
                sb.Append(FormattableString.Invariant($"\"UseTrueVectorLDS\": {config.UseTrueVectorLDS.ToString().ToLower()}, "));
                sb.Append(FormattableString.Invariant($"\"UseColumnMajorA\": {config.UseColumnMajorA.ToString().ToLower()}, "));
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
        if (string.IsNullOrWhiteSpace(json))
            return results;

        try
        {
            using var document = JsonDocument.Parse(json);
            if (document.RootElement.ValueKind != JsonValueKind.Object)
                return results;

            foreach (var entry in document.RootElement.EnumerateObject())
            {
                if (entry.Value.ValueKind != JsonValueKind.Object)
                    continue;

                var obj = entry.Value;

                int GetIntValue(string name)
                {
                    if (!obj.TryGetProperty(name, out var value))
                        return 0;
                    if (value.TryGetInt32(out var intValue))
                        return intValue;
                    return value.TryGetDouble(out var doubleValue) ? (int)doubleValue : 0;
                }

                bool GetBoolValue(string name)
                {
                    if (!obj.TryGetProperty(name, out var value))
                        return false;
                    return value.ValueKind switch
                    {
                        JsonValueKind.True => true,
                        JsonValueKind.False => false,
                        JsonValueKind.Number => value.TryGetInt32(out var intValue) && intValue != 0,
                        JsonValueKind.String => bool.TryParse(value.GetString(), out var result) && result,
                        _ => false
                    };
                }

                string GetStringValue(string name)
                {
                    if (!obj.TryGetProperty(name, out var value))
                        return string.Empty;
                    return value.ValueKind == JsonValueKind.String ? value.GetString() ?? string.Empty : string.Empty;
                }

                double GetDoubleValue(string name)
                {
                    if (!obj.TryGetProperty(name, out var value))
                        return 0.0;
                    return value.TryGetDouble(out var doubleValue) ? doubleValue : 0.0;
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
                    // CLBlast-style parameters
                    KReg = GetIntValue("KReg"),
                    KUnroll = GetIntValue("KUnroll"),
                    UseSubgroupOps = GetBoolValue("UseSubgroupOps"),
                    StrideM = GetBoolValue("StrideM"),
                    StrideN = GetBoolValue("StrideN"),
                    CacheA = GetBoolValue("CacheA"),
                    CacheB = GetBoolValue("CacheB"),
                    MdimaSize = GetIntValue("MdimaSize"),
                    NdimbSize = GetIntValue("NdimbSize"),
                    KernelName = GetStringValue("KernelName"),
                    UseTrueVectorLDS = GetBoolValue("UseTrueVectorLDS"),
                    UseColumnMajorA = GetBoolValue("UseColumnMajorA")
                };

                var gflops = NormalizeGflops(GetDoubleValue("GFlops"));

                if (config.TileM > 0 && config.TileN > 0)
                {
                    results.Add((entry.Name, config, gflops));
                }
            }
        }
        catch (JsonException)
        {
            // Skip malformed entries
        }
        catch
        {
            // Skip malformed entries
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
