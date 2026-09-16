using System.Diagnostics;
using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Serialization;
using AiDotNet.TestImpact;

namespace AttributionRuntime;

public enum AttributionFault { LateHit, UnclosedScope, InvalidBoundary, IncompleteWorker, UnjoinedTask, UntrackedProcess, UntrackedConcurrency, InvalidInventory, IncompleteCase, CaseResultsMismatch, UnsuccessfulCase, ExecutionContextChanged }
public enum AttributionProcessKind { TestHost, Worker }
public enum HitCollectionMode { Serialized, Cached }
public enum DiscoveredCaseKind { Enumerated, DeferredOrCustom }
public enum ObservedOutcome { Passed, Failed, Skipped }
public sealed record DiscoveredCase(string Id, string Owner, string DisplayName, DiscoveredCaseKind Kind);
public sealed record ObservedCaseResult(string DisplayName, ObservedOutcome Outcome);
public sealed record CaseExecutionReport(DiscoveredCase Case, bool Finished, ObservedCaseResult[] Results);
public sealed record WorkerTicket(string Run, string Token, string Owner, bool Started, bool Completed);
public sealed record MethodHits(string Owner, string[] Methods);
public sealed record AttributionReport(int Schema, string Run, string Token,
    AttributionProcessKind Kind, int ProcessId, string? WorkerOwner, int PeakScopes, HitCollectionMode CollectionMode,
    MethodHits[] Hits, string[] CompletedOwners, AttributionFault[] Faults, WorkerTicket[] Workers, CaseExecutionReport[] Cases,
    ExecutionPlan? Plan);

// Prototype only: unknown-context hits apply to the entire test-host execution group.
// AsyncLocal identifies ownership; a shared scope object detects hits after its owner closes.
public static class Tracker
{
    private readonly record struct MethodIdentity(string Module, int Token)
    {
        public string Serialize() => Token == 0 ? Module : $"{Module}:{Token:X8}";
    }
    public static bool IsEnabled => DirectoryPath is not null;
    public const string SharedOwner = "<execution-group>";
    private sealed class Scope(string owner)
    {
        public string Owner { get; } = owner;
        public bool Closed;
        public ConcurrentDictionary<MethodIdentity, byte> Recorded { get; } = new();
        public HashSet<Task> Tasks { get; } = new(ReferenceEqualityComparer.Instance);
    }

    private static readonly object Gate = new();
    private static readonly AsyncLocal<Scope?> Current = new();
    private static readonly Dictionary<string, HashSet<MethodIdentity>> Hits = new(StringComparer.Ordinal);
    private static readonly HashSet<string> Completed = new(StringComparer.Ordinal);
    private static readonly HashSet<Scope> Open = new();
    private static readonly HashSet<AttributionFault> Faults = new();
    private static readonly List<WorkerTicket> Workers = new();
    private static readonly HashSet<Task> GroupTasks = new(ReferenceEqualityComparer.Instance);
    private sealed class CaseExecution(DiscoveredCase descriptor)
    {
        public DiscoveredCase Descriptor { get; } = descriptor;
        public List<ObservedCaseResult> Results { get; } = [];
        public bool Finished;
    }
    private static readonly Dictionary<string, CaseExecution> Cases = new(StringComparer.Ordinal);
    private static bool inventoryRegistered;
    private static ExecutionPlan? executionPlan;
    private static readonly string? DirectoryPath = OptionalEnvironment("ATTRIBUTION_OUTPUT");
    private static readonly string Run = OptionalEnvironment("ATTRIBUTION_RUN") ?? "";
    private static readonly string? WorkerOwner = OptionalEnvironment("ATTRIBUTION_OWNER");
    private static readonly string Token = OptionalEnvironment("ATTRIBUTION_TOKEN") ?? Guid.NewGuid().ToString("N");
    private static readonly AttributionProcessKind Kind = WorkerOwner is null ? AttributionProcessKind.TestHost : AttributionProcessKind.Worker;
    private static int peakScopes;
    private static bool published;
    private static bool revoked;
    private static readonly HitCollectionMode CollectionMode = ReadCollectionMode();

    private static HitCollectionMode ReadCollectionMode()
    {
        string? value = OptionalEnvironment("ATTRIBUTION_HIT_MODE");
        if (value is null) return HitCollectionMode.Cached;
        return Enum.TryParse(value, out HitCollectionMode mode) && Enum.IsDefined(mode)
            ? mode : throw new InvalidOperationException("Invalid hit collection mode.");
    }

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

    public static void ConfigurePlan(ExecutionPlan plan)
    {
        if (DirectoryPath is null) throw new InvalidOperationException("Planned execution requires attribution.");
        lock (Gate)
        {
            RevokePublishedReport();
            if (executionPlan is not null || inventoryRegistered) throw new InvalidOperationException("Execution plan is already fixed.");
            executionPlan = plan with { RequiredCases = plan.RequiredCases.ToArray() };
        }
    }

    public static void InvalidateExecutionContext()
    {
        lock (Gate)
        {
            RevokePublishedReport();
            Faults.Add(AttributionFault.ExecutionContextChanged);
        }
    }

    public static void RegisterCases(IEnumerable<DiscoveredCase> inventory)
    {
        if (DirectoryPath is null) return;
        lock (Gate)
        {
            RevokePublishedReport();
            if (inventoryRegistered) Faults.Add(AttributionFault.InvalidInventory);
            inventoryRegistered = true;
            foreach (DiscoveredCase descriptor in inventory)
            {
                if (string.IsNullOrWhiteSpace(descriptor.Id) || string.IsNullOrWhiteSpace(descriptor.Owner) ||
                    string.IsNullOrWhiteSpace(descriptor.DisplayName) || !Enum.IsDefined(descriptor.Kind) ||
                    !Cases.TryAdd(descriptor.Id, new(descriptor))) Faults.Add(AttributionFault.InvalidInventory);
            }
            if (Cases.Count == 0) Faults.Add(AttributionFault.InvalidInventory);
        }
    }

    public static void RecordCaseResult(string id, string displayName, ObservedOutcome outcome)
    {
        if (DirectoryPath is null) return;
        lock (Gate)
        {
            RevokePublishedReport();
            if (!Cases.TryGetValue(id, out CaseExecution? execution) || execution.Finished ||
                string.IsNullOrWhiteSpace(displayName) || !Enum.IsDefined(outcome))
            {
                Faults.Add(AttributionFault.CaseResultsMismatch);
                return;
            }
            execution.Results.Add(new(displayName, outcome));
            if (outcome != ObservedOutcome.Passed) Faults.Add(AttributionFault.UnsuccessfulCase);
        }
    }

    public static void FinishCase(string id, int testsRun, int failed, int skipped)
    {
        if (DirectoryPath is null) return;
        lock (Gate)
        {
            RevokePublishedReport();
            if (!Cases.TryGetValue(id, out CaseExecution? execution) || execution.Finished)
            {
                Faults.Add(AttributionFault.CaseResultsMismatch);
                return;
            }
            execution.Finished = true;
            if (testsRun <= 0 || testsRun != execution.Results.Count ||
                failed != execution.Results.Count(result => result.Outcome == ObservedOutcome.Failed) ||
                skipped != execution.Results.Count(result => result.Outcome == ObservedOutcome.Skipped) ||
                (execution.Descriptor.Kind == DiscoveredCaseKind.Enumerated && testsRun != 1))
                Faults.Add(AttributionFault.CaseResultsMismatch);
        }
    }

    public static void Begin(string owner)
    {
        if (DirectoryPath is null) return;
        ArgumentException.ThrowIfNullOrWhiteSpace(owner);
        lock (Gate)
        {
            RevokePublishedReport();
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
            RevokePublishedReport();
            Scope? scope = Current.Value;
            if (scope is null || scope.Closed || scope.Owner != owner)
            {
                Faults.Add(AttributionFault.InvalidBoundary);
                throw new InvalidOperationException("Attribution boundary does not match its owner.");
            }
            Volatile.Write(ref scope.Closed, true);
            if (scope.Tasks.Any(task => !task.IsCompleted)) Faults.Add(AttributionFault.UnjoinedTask);
            if (Workers.Any(worker => worker.Owner == owner && !worker.Completed))
                Faults.Add(AttributionFault.IncompleteWorker);
            Open.Remove(scope);
            Completed.Add(owner);
            Current.Value = null;
        }
    }

    public static void ObserveTask(Task? task)
    {
        if (DirectoryPath is null || task is null || task.IsCompleted) return;
        lock (Gate)
        {
            RevokePublishedReport();
            Scope? scope = Current.Value;
            if (scope is null) GroupTasks.Add(task);
            else if (scope.Closed) Faults.Add(AttributionFault.LateHit);
            else scope.Tasks.Add(task);
        }
    }

    public static void CheckProcessStart(ProcessStartInfo start)
    {
        if (DirectoryPath is null) return;
        lock (Gate)
        {
            RevokePublishedReport();
            Scope? scope = Current.Value;
            start.Environment.TryGetValue("ATTRIBUTION_TOKEN", out string? token);
            start.Environment.TryGetValue("ATTRIBUTION_RUN", out string? run);
            start.Environment.TryGetValue("ATTRIBUTION_OWNER", out string? owner);
            int index = Workers.FindIndex(worker => worker.Token == token && worker.Run == run && worker.Owner == owner);
            if (scope is null || scope.Closed || run != Run || owner != scope.Owner ||
                index < 0 || Workers[index].Started || Workers[index].Completed)
                Faults.Add(AttributionFault.UntrackedProcess);
            else Workers[index] = Workers[index] with { Started = true };
        }
    }

    public static void UntrackedProcess()
    {
        if (DirectoryPath is not null) lock (Gate)
        {
            RevokePublishedReport();
            Faults.Add(AttributionFault.UntrackedProcess);
        }
    }

    public static void UntrackedConcurrency()
    {
        if (DirectoryPath is not null) lock (Gate)
        {
            RevokePublishedReport();
            Faults.Add(AttributionFault.UntrackedConcurrency);
        }
    }

    // One shared module string plus an integer token avoids adding a long unique
    // user string to the assembly metadata for every instrumented method. The
    // canonical report key is formatted only during publication, not on the hot
    // path. Typed identities avoid a second global lookup for every hit.
    public static void HitMethod(string module, int token) => HitCore(new(module, token));

    public static void Hit(string method) => HitCore(new(method, 0));

    private static void HitCore(MethodIdentity method)
    {
        if (DirectoryPath is null) return;
        Scope? scope = Current.Value;
        // Only already-published hits may bypass the global lock. Check closure
        // before the cache, so repeated hits from detached tasks still poison evidence.
        if (CollectionMode == HitCollectionMode.Cached && scope is not null &&
            !Volatile.Read(ref published) && !Volatile.Read(ref scope.Closed) && scope.Recorded.ContainsKey(method)) return;
        lock (Gate)
        {
            RevokePublishedReport();
            string owner = scope?.Owner ?? SharedOwner;
            if (scope?.Closed == true)
            {
                Faults.Add(AttributionFault.LateHit);
                owner = SharedOwner;
            }
            if (!Hits.TryGetValue(owner, out HashSet<MethodIdentity>? methods))
                Hits.Add(owner, methods = new());
            methods.Add(method);
            if (scope is not null && !scope.Closed) scope.Recorded.TryAdd(method, 0);
        }
    }

    public static WorkerTicket AttachWorker(ProcessStartInfo start)
    {
        if (DirectoryPath is null) throw new InvalidOperationException("Attribution is disabled.");
        lock (Gate)
        {
            RevokePublishedReport();
            Scope scope = Current.Value ?? throw new InvalidOperationException("Worker has no test owner.");
            if (scope.Closed) throw new InvalidOperationException("Worker owner already closed.");
            var ticket = new WorkerTicket(Run, Guid.NewGuid().ToString("N"), scope.Owner, false, false);
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
            RevokePublishedReport();
            Scope? scope = Current.Value;
            int index = Workers.FindIndex(worker => worker.Run == ticket.Run && worker.Token == ticket.Token && worker.Owner == ticket.Owner);
            if (scope is null || scope.Closed || scope.Owner != ticket.Owner || index < 0 ||
                !Workers[index].Started || Workers[index].Completed || exitCode != 0)
            {
                Faults.Add(AttributionFault.IncompleteWorker);
                throw new InvalidOperationException("Worker did not complete within its owning test.");
            }
            Workers[index] = Workers[index] with { Completed = true };
        }
    }

    // The worker executable must explicitly participate; a missing report is rejected by validation.
    public static string BeginWorker()
    {
        string owner = WorkerOwner ?? throw new InvalidOperationException("Missing worker ownership.");
        Begin(owner);
        return owner;
    }

    // Called only under Gate. Validation must wait for process termination and
    // reject this marker, even when the original report was otherwise complete.
    private static void RevokePublishedReport()
    {
        if (!published || revoked || DirectoryPath is null) return;
        using var marker = new FileStream(Path.Combine(DirectoryPath, $"{Token}.json.invalid"),
            FileMode.CreateNew, FileAccess.Write, FileShare.None);
        revoked = true;
    }

    public static void CompleteTestHost()
    {
        if (Kind != AttributionProcessKind.TestHost) throw new InvalidOperationException("Only the test host can complete the assembly report.");
        Flush();
    }

    private static void Flush()
    {
        if (DirectoryPath is null) return;
        lock (Gate)
        {
            if (published) return;
            if (Open.Count != 0) Faults.Add(AttributionFault.UnclosedScope);
            if (GroupTasks.Any(task => !task.IsCompleted)) Faults.Add(AttributionFault.UnjoinedTask);
            if (Kind == AttributionProcessKind.TestHost && (!inventoryRegistered || Cases.Count == 0))
                Faults.Add(AttributionFault.InvalidInventory);
            if (Cases.Values.Any(execution => !execution.Finished)) Faults.Add(AttributionFault.IncompleteCase);
            var report = new AttributionReport(3, Run, Token, Kind, Environment.ProcessId,
                WorkerOwner, peakScopes, CollectionMode,
                Hits.OrderBy(pair => pair.Key, StringComparer.Ordinal)
                    .Select(pair => new MethodHits(pair.Key, pair.Value.Select(method => method.Serialize()).Order(StringComparer.Ordinal).ToArray())).ToArray(),
                Completed.Order(StringComparer.Ordinal).ToArray(), Faults.Order().ToArray(), Workers.ToArray(),
                Cases.OrderBy(pair => pair.Key, StringComparer.Ordinal).Select(pair => new CaseExecutionReport(
                    pair.Value.Descriptor, pair.Value.Finished, pair.Value.Results.ToArray())).ToArray(), executionPlan);
            Directory.CreateDirectory(DirectoryPath);
            string destination = Path.Combine(DirectoryPath, $"{Token}.json");
            string pending = destination + ".pending";
            var options = new JsonSerializerOptions { WriteIndented = true };
            options.Converters.Add(new JsonStringEnumConverter());
            // Missing, truncated, or pending-only files cannot satisfy the validator.
            using (var stream = new FileStream(pending, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                JsonSerializer.Serialize(stream, report, options);
            File.Move(pending, destination, overwrite: false);
            Volatile.Write(ref published, true);
        }
    }
}
