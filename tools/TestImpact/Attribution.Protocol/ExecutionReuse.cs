namespace AiDotNet.TestImpact;

public sealed record BoundDependencySnapshot(ExecutionContextIdentity Context, DependencySnapshot Graph);

// Protocol consistency only: these inputs are NOT authenticated by this library.
// The production caller must prove graph completeness and the complete change
// set before using this partition to skip execution. No certificate is issued.
public static class ExecutionReuse
{
    public static ReusePartition Prepare(VerifiedExecution baseline, string workload,
        TestCaseIdentity[] currentInventory, ExecutionContextIdentity currentContext,
        BoundDependencySnapshot before, BoundDependencySnapshot after,
        string[] changedNodes, bool unmappedChange)
    {
        ArgumentNullException.ThrowIfNull(baseline);
        ArgumentNullException.ThrowIfNull(before);
        ArgumentNullException.ThrowIfNull(after);
        if (!baseline.CanReplaceFullBaseline)
            throw new EvidenceException(EvidenceFailure.Scope, "A partial run cannot seed changed-base reuse.");
        if (baseline.Workload != workload)
            throw new EvidenceException(EvidenceFailure.Plan, "Baseline belongs to another workload.");
        if (before.Context != baseline.Context || after.Context != currentContext)
            throw new EvidenceException(EvidenceFailure.Context, "Dependency graphs belong to different revisions or builds.");
        ExecutionPlan full = ExecutionEvidence.CreatePlan(workload, currentInventory, [], ValidationScope.FullWorkload, currentContext);
        TestCaseIdentity[] current = full.RequiredCases;
        RequireExactMethods(before.Graph, baseline.Cases);
        RequireExactMethods(after.Graph, current);
        HashSet<string> selected = DependencySelection.Select(before.Graph, after.Graph, changedNodes, unmappedChange)
            .Select(method => method.MethodId).ToHashSet(StringComparer.Ordinal);
        var oldMethods = baseline.Cases.GroupBy(test => test.MethodId, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Select(test => test.CaseId).ToHashSet(StringComparer.Ordinal), StringComparer.Ordinal);
        foreach (var method in current.GroupBy(test => test.MethodId, StringComparer.Ordinal))
        {
            // Discovery changes include added AND removed theory rows. Run the
            // complete surviving method; don't reuse only a convenient subset.
            if (baseline.Context.ProfileFingerprint != currentContext.ProfileFingerprint ||
                !oldMethods.TryGetValue(method.Key, out HashSet<string>? oldCases) ||
                !oldCases.SetEquals(method.Select(test => test.CaseId))) selected.Add(method.Key);
        }
        TestCaseIdentity[] reused = current.Where(test => !selected.Contains(test.MethodId)).ToArray();
        ExecutionPlan? run = selected.Count == 0 ? null : selected.Count == after.Graph.Tests.Length
            ? full : ExecutionEvidence.CreatePlan(workload, current, selected.ToArray(), ValidationScope.SelectedMethods, currentContext);
        return new ReusePartition(full, run, reused, baseline);
    }

    public static VerifiedReusePartition Complete(ReusePartition partition, VerifiedExecution? executed)
    {
        ArgumentNullException.ThrowIfNull(partition);
        ExecutionPlan? required = partition.Execution;
        if (required is null)
        {
            if (executed is not null) throw new EvidenceException(EvidenceFailure.Plan, "Reuse-only partition cannot consume an unrelated execution.");
        }
        else if (executed is null || !ExecutionEvidence.CanReuseIdenticalExecution(executed, required))
            throw new EvidenceException(EvidenceFailure.Plan, "Current execution does not satisfy the exact partition.");
        return new VerifiedReusePartition(partition, executed);
    }

    private static void RequireExactMethods(DependencySnapshot graph, IEnumerable<TestCaseIdentity> inventory)
    {
        ArgumentNullException.ThrowIfNull(graph);
        if (graph.Tests is null || graph.Tests.Any(test => test is null) ||
            !inventory.Select(test => test.MethodId).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal)
                .SequenceEqual(graph.Tests.Select(test => test.MethodId).Order(StringComparer.Ordinal)))
            throw new EvidenceException(EvidenceFailure.Inventory, "Graph roots do not cover exactly the discovered test methods.");
    }
}

public sealed class ReusePartition
{
    private readonly ExecutionPlan? execution;
    internal ReusePartition(ExecutionPlan full, ExecutionPlan? run, TestCaseIdentity[] reused, VerifiedExecution baseline)
    {
        Context = full.Context;
        Workload = full.Workload;
        InventoryHash = full.InventoryHash;
        Cases = Array.AsReadOnly(full.RequiredCases.ToArray());
        ReusedCases = Array.AsReadOnly(reused.ToArray());
        execution = run;
        Baseline = baseline;
    }
    public ExecutionContextIdentity Context { get; }
    public string Workload { get; }
    public string InventoryHash { get; }
    public IReadOnlyList<TestCaseIdentity> Cases { get; }
    public IReadOnlyList<TestCaseIdentity> ReusedCases { get; }
    public VerifiedExecution Baseline { get; }
    // ExecutionPlan contains an array, so never expose our stored partition.
    public ExecutionPlan? Execution => execution is null ? null : execution with { RequiredCases = execution.RequiredCases.ToArray() };
}

public sealed class VerifiedReusePartition
{
    internal VerifiedReusePartition(ReusePartition partition, VerifiedExecution? executed)
    {
        Partition = partition;
        Executed = executed;
    }
    public ReusePartition Partition { get; }
    public VerifiedExecution? Executed { get; }
    // A mixed/reused result is not a freshly observed full-workload baseline.
    public bool CanReplaceFullBaseline => Partition.ReusedCases.Count == 0 && Executed?.CanReplaceFullBaseline == true;
}
