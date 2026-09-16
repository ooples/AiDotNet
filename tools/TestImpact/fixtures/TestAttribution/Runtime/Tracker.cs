using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AttributionRuntime;

public enum AttributionFault { LateHit, UnclosedScope, InvalidBoundary, IncompleteWorker }
public enum AttributionProcessKind { TestHost, Worker }
public sealed record WorkerTicket(string Run, string Token, string Owner, bool Completed);
public sealed record MethodHits(string Owner, string[] Methods);
public sealed record AttributionReport(int Schema, string Run, string Token,
    AttributionProcessKind Kind, int ProcessId, string? WorkerOwner, int PeakScopes,
    MethodHits[] Hits, string[] CompletedOwners, AttributionFault[] Faults, WorkerTicket[] Workers);

// Prototype only: unknown-context hits apply to the entire test-host execution group.
// AsyncLocal identifies ownership; a shared scope object detects hits after its owner closes.
public static class Tracker
{
    public const string SharedOwner = "<execution-group>";
    private sealed class Scope(string owner)
    {
        public string Owner { get; } = owner;
        public bool Closed { get; set; }
    }

    private static readonly object Gate = new();
    private static readonly AsyncLocal<Scope?> Current = new();
    private static readonly Dictionary<string, HashSet<string>> Hits = new(StringComparer.Ordinal);
    private static readonly HashSet<string> Completed = new(StringComparer.Ordinal);
    private static readonly HashSet<Scope> Open = new();
    private static readonly HashSet<AttributionFault> Faults = new();
    private static readonly List<WorkerTicket> Workers = new();
    private static readonly string? DirectoryPath = OptionalEnvironment("ATTRIBUTION_OUTPUT");
    private static readonly string Run = OptionalEnvironment("ATTRIBUTION_RUN") ?? "";
    private static readonly string? WorkerOwner = OptionalEnvironment("ATTRIBUTION_OWNER");
    private static readonly string Token = OptionalEnvironment("ATTRIBUTION_TOKEN") ?? Guid.NewGuid().ToString("N");
    private static readonly AttributionProcessKind Kind = WorkerOwner is null ? AttributionProcessKind.TestHost : AttributionProcessKind.Worker;
    private static int peakScopes;

    private static string? OptionalEnvironment(string name)
    {
        string? value = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrEmpty(value) ? null : value;
    }

    static Tracker()
    {
        if (DirectoryPath is not null)
        {
            if (!Guid.TryParseExact(Run, "N", out _) || !Guid.TryParseExact(Token, "N", out _))
                throw new InvalidOperationException("Invalid prototype run or process token.");
            AppDomain.CurrentDomain.ProcessExit += (_, _) => Flush();
        }
    }

    public static void Begin(string owner)
    {
        if (DirectoryPath is null) return;
        ArgumentException.ThrowIfNullOrWhiteSpace(owner);
        lock (Gate)
        {
            if (Current.Value is not null)
            {
                Faults.Add(AttributionFault.InvalidBoundary);
                throw new InvalidOperationException("Nested or stale attribution boundary.");
            }
            var scope = new Scope(owner);
            Current.Value = scope;
            Open.Add(scope);
            peakScopes = Math.Max(peakScopes, Open.Count);
        }
    }

    public static void End(string owner)
    {
        if (DirectoryPath is null) return;
        lock (Gate)
        {
            Scope? scope = Current.Value;
            if (scope is null || scope.Closed || scope.Owner != owner)
            {
                Faults.Add(AttributionFault.InvalidBoundary);
                throw new InvalidOperationException("Attribution boundary does not match its owner.");
            }
            scope.Closed = true;
            if (Workers.Any(worker => worker.Owner == owner && !worker.Completed))
                Faults.Add(AttributionFault.IncompleteWorker);
            Open.Remove(scope);
            Completed.Add(owner);
            Current.Value = null;
        }
    }

    public static void Hit(string method)
    {
        if (DirectoryPath is null) return;
        lock (Gate)
        {
            Scope? scope = Current.Value;
            string owner = scope?.Owner ?? SharedOwner;
            if (scope?.Closed == true)
            {
                Faults.Add(AttributionFault.LateHit);
                owner = SharedOwner;
            }
            if (!Hits.TryGetValue(owner, out HashSet<string>? methods))
                Hits.Add(owner, methods = new(StringComparer.Ordinal));
            methods.Add(method);
        }
    }

    public static WorkerTicket AttachWorker(ProcessStartInfo start)
    {
        if (DirectoryPath is null) throw new InvalidOperationException("Attribution is disabled.");
        lock (Gate)
        {
            Scope scope = Current.Value ?? throw new InvalidOperationException("Worker has no test owner.");
            if (scope.Closed) throw new InvalidOperationException("Worker owner already closed.");
            var ticket = new WorkerTicket(Run, Guid.NewGuid().ToString("N"), scope.Owner, false);
            Workers.Add(ticket);
            start.Environment["ATTRIBUTION_OUTPUT"] = DirectoryPath;
            start.Environment["ATTRIBUTION_RUN"] = ticket.Run;
            start.Environment["ATTRIBUTION_TOKEN"] = ticket.Token;
            start.Environment["ATTRIBUTION_OWNER"] = ticket.Owner;
            return ticket;
        }
    }

    public static void CompleteWorker(WorkerTicket ticket, int exitCode)
    {
        lock (Gate)
        {
            Scope? scope = Current.Value;
            int index = Workers.FindIndex(worker => worker == ticket);
            if (scope is null || scope.Closed || scope.Owner != ticket.Owner || index < 0 || exitCode != 0)
            {
                Faults.Add(AttributionFault.IncompleteWorker);
                throw new InvalidOperationException("Worker did not complete within its owning test.");
            }
            Workers[index] = ticket with { Completed = true };
        }
    }

    // The worker executable must explicitly participate; a missing report is rejected by validation.
    public static string BeginWorker()
    {
        string owner = WorkerOwner ?? throw new InvalidOperationException("Missing worker ownership.");
        Begin(owner);
        return owner;
    }

    private static void Flush()
    {
        if (DirectoryPath is null) return;
        lock (Gate)
        {
            if (Open.Count != 0) Faults.Add(AttributionFault.UnclosedScope);
            var report = new AttributionReport(1, Run, Token, Kind, Environment.ProcessId,
                WorkerOwner, peakScopes,
                Hits.OrderBy(pair => pair.Key, StringComparer.Ordinal)
                    .Select(pair => new MethodHits(pair.Key, pair.Value.Order(StringComparer.Ordinal).ToArray())).ToArray(),
                Completed.Order(StringComparer.Ordinal).ToArray(), Faults.Order().ToArray(), Workers.ToArray());
            Directory.CreateDirectory(DirectoryPath);
            string destination = Path.Combine(DirectoryPath, $"{Token}.json");
            string pending = destination + ".pending";
            var options = new JsonSerializerOptions { WriteIndented = true };
            options.Converters.Add(new JsonStringEnumConverter());
            // Missing, truncated, or pending-only files cannot satisfy the validator.
            using (var stream = new FileStream(pending, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                JsonSerializer.Serialize(stream, report, options);
            File.Move(pending, destination, overwrite: false);
        }
    }
}
