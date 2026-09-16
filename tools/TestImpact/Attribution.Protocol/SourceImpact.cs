namespace AiDotNet.TestImpact;

public enum SourceMapStatus { Verified, Unverifiable }
public sealed record SourceSpan(string Path, int FirstLine, int LastLine);
public sealed record SourceMethod(DependencyNode Dependency, string Owner, string BodyHash, SourceSpan[] Spans, bool Lifecycle);
public sealed record SourceSnapshot(int Schema, string SourceTree, string AssemblyFile, string AssemblyHash, string PdbHash, string ConfigurationHash,
    SourceMapStatus Status, SourceMethod[] Methods);
public sealed record ChangedLines(string Path, int Start, int Count);
public sealed record SourceDelta(string BeforeTree, string AfterTree, ChangedLines[] Before, ChangedLines[] After, bool Unmapped);
public sealed record SourceSelection(SelectedMethod[] Methods, bool FullFallback, string[] ChangedNodes);

// Source snapshots must be produced from checksum-verified PDBs. Unknown source
// or metadata edits broaden execution; observed hits alone never authorize skips.
public static class SourceImpact
{
    public static ReusePartition PrepareReuse(VerifiedExecution baseline, SourceSnapshot before, SourceSnapshot after,
        DiscoveryManifest currentInventory, SourceDelta delta)
    {
        if (currentInventory.Schema != 1 || before.SourceTree != baseline.Context.SourceTree ||
            after.SourceTree != currentInventory.Context.SourceTree || before.Status != SourceMapStatus.Verified || after.Status != SourceMapStatus.Verified)
            throw new EvidenceException(EvidenceFailure.Context, "Reuse snapshots do not match verified execution sources.");
        SourceSelection selection = Select(before, after, baseline.Cases.ToArray(), currentInventory.Cases, delta);
        bool fallback = selection.FullFallback;
        DependencySnapshot oldGraph = Bind(before, baseline.Cases.ToArray(), ref fallback);
        DependencySnapshot newGraph = Bind(after, currentInventory.Cases, ref fallback);
        return ExecutionReuse.Prepare(baseline, currentInventory.Workload, currentInventory.Cases, currentInventory.Context,
            new(baseline.Context, oldGraph), new(currentInventory.Context, newGraph), selection.ChangedNodes, fallback);
    }

    public static SourceSelection Select(SourceSnapshot before, SourceSnapshot after,
        TestCaseIdentity[] oldInventory, TestCaseIdentity[] currentInventory, SourceDelta delta)
    {
        Validate(before);
        Validate(after);
        if (before.Schema != 1 || after.Schema != 1 || before.SourceTree != delta.BeforeTree || after.SourceTree != delta.AfterTree)
            throw new EvidenceException(EvidenceFailure.Context, "Source snapshots do not match the diff revisions.");
        var changed = new HashSet<string>(StringComparer.Ordinal);
        bool fallback = delta.Unmapped || before.Status != SourceMapStatus.Verified || after.Status != SourceMapStatus.Verified ||
            before.ConfigurationHash != after.ConfigurationHash;
        DependencySnapshot oldGraph = Bind(before, oldInventory, ref fallback);
        DependencySnapshot newGraph = Bind(after, currentInventory, ref fallback);
        fallback |= !Map(before, delta.Before, changed) | !Map(after, delta.After, changed);
        var oldBodies = before.Methods.ToDictionary(method => method.Dependency.Id, method => method.BodyHash, StringComparer.Ordinal);
        var newBodies = after.Methods.ToDictionary(method => method.Dependency.Id, method => method.BodyHash, StringComparer.Ordinal);
        foreach ((string id, string body) in oldBodies)
            if (!newBodies.TryGetValue(id, out string? next) || next != body) changed.Add(id);
        foreach ((string id, string body) in newBodies)
            if (!oldBodies.TryGetValue(id, out string? previousBody) || previousBody != body) changed.Add(id);
        // Discovery changes must rerun complete methods, including removed rows.
        var previous = oldInventory.GroupBy(test => test.MethodId, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Select(test => test.CaseId).ToHashSet(StringComparer.Ordinal), StringComparer.Ordinal);
        var selected = DependencySelection.Select(oldGraph, newGraph, changed.ToArray(), fallback)
            .ToDictionary(method => method.MethodId, StringComparer.Ordinal);
        foreach (var method in currentInventory.GroupBy(test => test.MethodId, StringComparer.Ordinal))
            if (!previous.TryGetValue(method.Key, out HashSet<string>? cases) || !cases.SetEquals(method.Select(test => test.CaseId)))
                selected.TryAdd(method.Key, new(method.Key, SelectionReason.NewTest));
        return new(selected.Values.OrderBy(method => method.MethodId, StringComparer.Ordinal).ToArray(), fallback,
            changed.Order(StringComparer.Ordinal).ToArray());
    }

    private static void Validate(SourceSnapshot snapshot)
    {
        static bool Hash(string value, int length) => value is not null && value.Length == length &&
            value.All(character => character is >= '0' and <= '9' or >= 'a' and <= 'f');
        if (snapshot is null || !Hash(snapshot.SourceTree, 40) || !Hash(snapshot.AssemblyHash, 64) ||
            !Hash(snapshot.PdbHash, 64) || !Hash(snapshot.ConfigurationHash, 64) || !Enum.IsDefined(snapshot.Status) || snapshot.Methods is null ||
            snapshot.Methods.Any(method => method is null || method.Dependency is null || method.Spans is null ||
                string.IsNullOrWhiteSpace(method.Owner) || !Hash(method.BodyHash, 64)))
            throw new EvidenceException(EvidenceFailure.Format, "Incomplete source snapshot.");
    }

    private static DependencySnapshot Bind(SourceSnapshot snapshot, TestCaseIdentity[] inventory, ref bool fallback)
    {
        if (snapshot.Methods is null || inventory is null || inventory.Length == 0 ||
            inventory.Any(test => test is null || string.IsNullOrWhiteSpace(test.CaseId) || string.IsNullOrWhiteSpace(test.MethodId)) ||
            inventory.Select(test => test.CaseId).Distinct(StringComparer.Ordinal).Count() != inventory.Length)
            throw new EvidenceException(EvidenceFailure.Inventory, "Missing or duplicate source selection inventory.");
        var owners = snapshot.Methods.GroupBy(method => method.Owner, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Select(method => method.Dependency.Id).ToArray(), StringComparer.Ordinal);
        var roots = new List<TestDependencyRoots>();
        var testOwners = inventory.Select(test => test.MethodId).ToHashSet(StringComparer.Ordinal);
        foreach (string owner in testOwners)
        {
            if (!owners.TryGetValue(owner, out string[]? nodes)) { fallback = true; nodes = ["unmapped:" + owner]; }
            roots.Add(new(owner, nodes));
        }
        // Test assembly helpers can run during discovery or shared fixtures even
        // when they have no dynamically observed test owner. Keep them group-wide.
        string[] groups = snapshot.Methods.Where(method => method.Lifecycle || !testOwners.Contains(method.Owner))
            .Select(method => method.Dependency.Id).ToArray();
        return new(snapshot.Methods.Select(method => method.Dependency).ToArray(), roots.ToArray(), groups);
    }

    private static bool Map(SourceSnapshot snapshot, ChangedLines[] edits, HashSet<string> changed)
    {
        bool complete = true;
        foreach (ChangedLines edit in edits)
        {
            if (string.IsNullOrWhiteSpace(edit.Path) || edit.Start < 0 || edit.Count < 0 || (long)edit.Start + edit.Count > int.MaxValue)
                throw new EvidenceException(EvidenceFailure.Format, "Invalid source diff range.");
            if (edit.Count == 0) continue;
            int end = edit.Start + edit.Count - 1;
            var spans = new List<SourceSpan>();
            foreach (SourceMethod method in snapshot.Methods)
                foreach (SourceSpan span in method.Spans)
                {
                    if (span.FirstLine <= 0 || span.LastLine < span.FirstLine)
                        throw new EvidenceException(EvidenceFailure.Format, "Invalid mapped source span.");
                    if (span.Path != edit.Path || span.LastLine < edit.Start || span.FirstLine > end) continue;
                    changed.Add(method.Dependency.Id);
                    spans.Add(span);
                }
            // Prove every changed line is covered without iterating huge ranges.
            long next = edit.Start;
            foreach (SourceSpan span in spans.OrderBy(span => span.FirstLine))
            {
                if (span.FirstLine > next) break;
                next = Math.Max(next, (long)span.LastLine + 1);
            }
            if (next <= end) complete = false;
        }
        return complete;
    }
}
