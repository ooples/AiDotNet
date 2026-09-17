using System.Security.Cryptography;
using System.Text;

namespace AttributionRuntime;

public enum TrialScopeState { Open, Complete, Rejected }
public enum TrialScopeLedgerState { Recorded, Invalid }
public sealed record TrialScopeObservation(string Owner, string PathHash, string RootHash, string PreviousHash, string PreviousPathHash,
    TrialScopeState State);
public sealed record TrialScopeReport(TrialScopeLedgerState State, TrialScopeObservation[] Scopes);

internal enum TrialPathState { Absent, Present, Unavailable }
internal sealed record TrialPathSample(string PathHash, string RootHash, TrialPathState State);

// Observations only. No pathname leaves the process, and absence at the two
// boundaries does not prove that arbitrary test code performed no intermediate
// I/O. The static lifecycle/body contracts remain mandatory for reuse.
internal sealed class TrialPathScopes
{
    private readonly Dictionary<string, TrialScopeObservation> scopes = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> paths = new(StringComparer.Ordinal);
    private TrialScopeLedgerState state;

    internal void Invalidate() => state = TrialScopeLedgerState.Invalid;

    internal void Begin(string owner, TrialPathSample sample, string previousHash, string activeHash, string? previousPathHash = null)
    {
        if (string.IsNullOrWhiteSpace(owner) || scopes.ContainsKey(owner)) { Invalidate(); return; }
        previousPathHash ??= previousHash;
        bool accepted = sample.State == TrialPathState.Absent && sample.PathHash == activeHash && previousPathHash != sample.PathHash;
        if (!paths.TryAdd(sample.PathHash, owner))
        {
            accepted = false;
            string other = paths[sample.PathHash];
            scopes[other] = scopes[other] with { State = TrialScopeState.Rejected };
        }
        scopes.Add(owner, new(owner, sample.PathHash, sample.RootHash, previousHash, previousPathHash,
            accepted ? TrialScopeState.Open : TrialScopeState.Rejected));
    }

    internal void End(string owner, TrialPathSample sample, string restoredHash)
    {
        if (!scopes.TryGetValue(owner, out TrialScopeObservation? observation)) { Invalidate(); return; }
        bool accepted = observation.State == TrialScopeState.Open && sample.State == TrialPathState.Absent &&
            sample.PathHash == observation.PathHash && sample.RootHash == observation.RootHash && restoredHash == observation.PreviousHash;
        scopes[owner] = observation with { State = accepted ? TrialScopeState.Complete : TrialScopeState.Rejected };
    }

    internal TrialScopeReport Snapshot() => new(state, scopes.Values.OrderBy(scope => scope.Owner, StringComparer.Ordinal).ToArray());

    internal static string ValueHash(string? value) => Convert.ToHexStringLower(SHA256.HashData(
        Encoding.UTF8.GetBytes(value is null ? "null:" : "value:" + value)));

    internal static string PathHash(string? path)
    {
        if (path is null) return ValueHash(null);
        try
        {
            if (!Path.IsPathFullyQualified(path)) return ValueHash("relative:" + path);
            string full = Path.GetFullPath(path);
            return ValueHash(OperatingSystem.IsWindows() ? full.ToUpperInvariant() : full);
        }
        catch (Exception error) when (error is IOException or ArgumentException or NotSupportedException)
        {
            return ValueHash("invalid:" + path);
        }
    }

    internal static TrialPathSample Inspect(string? path, string? temporaryRoot = null)
    {
        string pathHash = PathHash(path);
        try
        {
            if (path is null || !Path.IsPathFullyQualified(path)) return new(pathHash, "", TrialPathState.Unavailable);
            string root = Path.GetFullPath(Path.Combine(temporaryRoot ?? Path.GetTempPath(), "aidotnet-trial-tests"));
            string rootHash = PathHash(root);
            string full = Path.GetFullPath(path);
            string name = Path.GetFileName(full);
            if (OperatingSystem.IsWindows() && full.StartsWith("\\\\", StringComparison.Ordinal) ||
                !string.Equals(Path.GetDirectoryName(full), root, OperatingSystem.IsWindows()
                    ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal) ||
                !name.EndsWith(".json", StringComparison.Ordinal) || !Guid.TryParseExact(name[..^5], "N", out _))
                return new(pathHash, rootHash, TrialPathState.Unavailable);
            // A missing leaf/directory is normal. Links in any existing ancestor
            // (including a broken leaf link) must not inherit containment proof.
            for (FileSystemInfo? entry = new FileInfo(full); entry is not null;
                 entry = entry is FileInfo file ? file.Directory : ((DirectoryInfo)entry).Parent)
                if (entry.LinkTarget is not null || entry.Exists && (entry.Attributes & FileAttributes.ReparsePoint) != 0)
                    return new(pathHash, rootHash, TrialPathState.Unavailable);
            if (new FileInfo(full + ".tombstone").LinkTarget is not null) return new(pathHash, rootHash, TrialPathState.Unavailable);
            return new(pathHash, rootHash, File.Exists(full) || Directory.Exists(full) || File.Exists(full + ".tombstone") ||
                Directory.Exists(full + ".tombstone") ? TrialPathState.Present : TrialPathState.Absent);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            return new(pathHash, "", TrialPathState.Unavailable);
        }
    }
}
