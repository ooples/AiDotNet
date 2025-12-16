using System;
using System.Collections.Concurrent;

namespace AiDotNet.Helpers;

/// <summary>
/// Internal diagnostics for inference decisions (non-user-facing).
/// Enable by setting env var AIDOTNET_DIAGNOSTICS=1.
/// </summary>
internal static class InferenceDiagnostics
{
    private const int MaxEntries = 1024;

    private static readonly bool Enabled =
        string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_DIAGNOSTICS"), "1", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_DIAGNOSTICS"), "true", StringComparison.OrdinalIgnoreCase);

    private static readonly ConcurrentQueue<InferenceDiagnosticEntry> Entries = new();

    internal static void RecordDecision(string area, string feature, bool enabled, string reason)
    {
        if (!Enabled)
            return;

        Entries.Enqueue(new InferenceDiagnosticEntry(
            TimestampUtc: DateTime.UtcNow,
            Area: area ?? string.Empty,
            Feature: feature ?? string.Empty,
            Enabled: enabled,
            Reason: reason ?? string.Empty,
            ExceptionType: null,
            ExceptionMessage: null));

        TrimIfNeeded();
    }

    internal static void RecordException(string area, string feature, Exception ex, string reason)
    {
        if (!Enabled)
            return;

        Entries.Enqueue(new InferenceDiagnosticEntry(
            TimestampUtc: DateTime.UtcNow,
            Area: area ?? string.Empty,
            Feature: feature ?? string.Empty,
            Enabled: false,
            Reason: reason ?? string.Empty,
            ExceptionType: ex.GetType().FullName ?? ex.GetType().Name,
            ExceptionMessage: ex.Message));

        TrimIfNeeded();
    }

    // Intentionally internal-only: serving can use InternalsVisibleTo to read these if needed later.
    internal static InferenceDiagnosticEntry[] Snapshot()
    {
        if (!Enabled)
            return Array.Empty<InferenceDiagnosticEntry>();

        return Entries.ToArray();
    }

    private static void TrimIfNeeded()
    {
        // Best-effort: bound memory use when diagnostics are enabled.
        while (Entries.Count > MaxEntries && Entries.TryDequeue(out _))
        {
        }
    }

    internal readonly record struct InferenceDiagnosticEntry(
        DateTime TimestampUtc,
        string Area,
        string Feature,
        bool Enabled,
        string Reason,
        string? ExceptionType,
        string? ExceptionMessage);
}
