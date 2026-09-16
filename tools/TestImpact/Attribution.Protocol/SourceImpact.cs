namespace AiDotNet.TestImpact;

public enum SourceMapStatus { Verified, Unverifiable }
public enum SourceMethodScope { TestAssembly, DependencyAssembly }
public sealed record SourceSpan(string Path, int FirstLine, int LastLine);
public sealed record SourceMethod(DependencyNode Dependency, string Owner, string BodyHash, SourceSpan[] Spans, bool Lifecycle,
    SourceMethodScope Scope = SourceMethodScope.TestAssembly);
public sealed record SourceTestLifecycle(string Owner, string[] Roots, bool Complete);
public sealed record SourceLifecycleMap(int Schema, SourceTestLifecycle[] Tests, string[] GroupRoots);
public enum ManagedBinaryOrigin { Bundle, Runtime }
public sealed record SourceDependencyBinary(string File, string Hash, ManagedBinaryOrigin Origin);
public sealed record SourceManagedDependencies(int Schema, SourceDependencyBinary[] Files, SourceMethod[] Methods, bool Truncated);
public sealed record SourceSnapshot(int Schema, string SourceTree, string AssemblyFile, string AssemblyHash, string PdbHash, string ConfigurationHash,
    SourceMapStatus Status, SourceMethod[] Methods, SourceLifecycleMap? LifecycleMap = null, SourceManagedDependencies? ManagedDependencies = null);
public sealed record ChangedLines(string Path, int Start, int Count);
public sealed record SourceDelta(string BeforeTree, string AfterTree, ChangedLines[] Before, ChangedLines[] After, bool Unmapped);
public sealed record SourceSelection(SelectedMethod[] Methods, bool FullFallback, string[] ChangedNodes);
public sealed record SourceBundleSnapshot(int Schema, string SourceTree, string TestAssembly, SourceSnapshot[] Assemblies);

// Source snapshots must be produced from checksum-verified PDBs. Unknown source
// or metadata edits broaden execution; observed hits alone never authorize skips.
public static class SourceImpact
{
    private sealed record Graph(string SourceTree, string ConfigurationHash, SourceMapStatus Status, SourceMethod[] Methods,
        SourceLifecycleMap? LifecycleMap);

    public static ReusePartition PrepareReuse(VerifiedExecution baseline, SourceSnapshot before, SourceSnapshot after,
        DiscoveryManifest currentInventory, SourceDelta delta)
        => PrepareReuse(baseline, ToGraph(before), ToGraph(after), currentInventory, delta);

    public static ReusePartition PrepareReuse(VerifiedExecution baseline, SourceBundleSnapshot before, SourceBundleSnapshot after,
        DiscoveryManifest currentInventory, SourceDelta delta)
        => PrepareReuse(baseline, ToGraph(before), ToGraph(after), currentInventory, delta);

    private static ReusePartition PrepareReuse(VerifiedExecution baseline, Graph before, Graph after,
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
        => Select(ToGraph(before), ToGraph(after), oldInventory, currentInventory, delta);

    public static SourceSelection Select(SourceBundleSnapshot before, SourceBundleSnapshot after,
        TestCaseIdentity[] oldInventory, TestCaseIdentity[] currentInventory, SourceDelta delta)
        => Select(ToGraph(before), ToGraph(after), oldInventory, currentInventory, delta);

    private static SourceSelection Select(Graph before, Graph after,
        TestCaseIdentity[] oldInventory, TestCaseIdentity[] currentInventory, SourceDelta delta)
    {
        if (before.SourceTree != delta.BeforeTree || after.SourceTree != delta.AfterTree)
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
        if (snapshot is null || snapshot.Schema != 1 || !Hash(snapshot.SourceTree, 40) || !Hash(snapshot.AssemblyHash, 64) ||
            !Hash(snapshot.PdbHash, 64) || !Hash(snapshot.ConfigurationHash, 64) || !Enum.IsDefined(snapshot.Status) || snapshot.Methods is null ||
            snapshot.Methods.Any(method => method is null || method.Dependency is null || method.Spans is null ||
                string.IsNullOrWhiteSpace(method.Owner) || string.IsNullOrWhiteSpace(method.Dependency.Id) ||
                !Hash(method.BodyHash, 64) || !Enum.IsDefined(method.Scope) || !Enum.IsDefined(method.Dependency.Boundary) ||
                method.Dependency.Calls is null || method.Dependency.SharedState is null ||
                method.Dependency.Calls.Any(string.IsNullOrWhiteSpace) || method.Dependency.SharedState.Any(string.IsNullOrWhiteSpace) ||
                method.Dependency.Calls.Distinct(StringComparer.Ordinal).Count() != method.Dependency.Calls.Length ||
                method.Dependency.SharedState.Distinct(StringComparer.Ordinal).Count() != method.Dependency.SharedState.Length ||
                method.Spans.Any(span => span is null)) ||
            snapshot.Methods.Select(method => method.Dependency.Id).Distinct(StringComparer.Ordinal).Count() != snapshot.Methods.Length)
            throw new EvidenceException(EvidenceFailure.Format, "Incomplete source snapshot.");
        if (snapshot.LifecycleMap is SourceLifecycleMap lifecycle &&
            (lifecycle.Schema != 1 || lifecycle.Tests is null || lifecycle.GroupRoots is null ||
             lifecycle.Tests.Any(test => test is null || string.IsNullOrWhiteSpace(test.Owner) || test.Roots is null ||
                 test.Roots.Length == 0 || test.Roots.Any(string.IsNullOrWhiteSpace) ||
                 test.Roots.Distinct(StringComparer.Ordinal).Count() != test.Roots.Length) ||
             lifecycle.Tests.Select(test => test.Owner).Distinct(StringComparer.Ordinal).Count() != lifecycle.Tests.Length ||
             lifecycle.GroupRoots.Any(string.IsNullOrWhiteSpace) ||
             lifecycle.GroupRoots.Distinct(StringComparer.Ordinal).Count() != lifecycle.GroupRoots.Length))
            throw new EvidenceException(EvidenceFailure.Format, "Incomplete lifecycle root map.");
        if (snapshot.ManagedDependencies is SourceManagedDependencies managed)
        {
            if (managed.Schema != 1 || managed.Files is null || managed.Methods is null ||
                managed.Files.Any(binary => binary is null || !Hash(binary.Hash, 64) || !Enum.IsDefined(binary.Origin) || string.IsNullOrWhiteSpace(binary.File)) ||
                managed.Files.Select(binary => (binary.Origin, binary.File.ToUpperInvariant())).Distinct().Count() != managed.Files.Length)
                throw new EvidenceException(EvidenceFailure.Format, "Incomplete managed dependency manifest.");
            foreach (SourceDependencyBinary binary in managed.Files) _ = ArtifactArchive.ResolveContained(Path.GetTempPath().TrimEnd(Path.DirectorySeparatorChar), binary.File);
            Validate(snapshot with { Methods = managed.Methods, LifecycleMap = null, ManagedDependencies = null });
            if (managed.Methods.Any(method => method.Spans.Length != 0 || !managed.Files.Any(binary =>
                    method.Dependency.Id.StartsWith(Path.GetFileNameWithoutExtension(binary.File) + ":", StringComparison.Ordinal))))
                throw new EvidenceException(EvidenceFailure.Format, "Managed method is not bound to a dependency binary.");
        }
    }

    private static Graph ToGraph(SourceSnapshot snapshot)
    {
        Validate(snapshot);
        return new(snapshot.SourceTree, Configuration(snapshot), snapshot.Status,
            MergeMethods(Methods(snapshot, SourceMethodScope.TestAssembly)), snapshot.LifecycleMap);
    }

    private static Graph ToGraph(SourceBundleSnapshot bundle)
    {
        if (bundle is null || bundle.Schema != 1 || bundle.Assemblies is null || bundle.Assemblies.Length == 0 ||
            string.IsNullOrWhiteSpace(bundle.TestAssembly))
            throw new EvidenceException(EvidenceFailure.Format, "Incomplete source bundle.");
        foreach (SourceSnapshot snapshot in bundle.Assemblies) Validate(snapshot);
        if (bundle.Assemblies.Any(snapshot => snapshot.SourceTree != bundle.SourceTree) ||
            bundle.Assemblies.Select(snapshot => snapshot.AssemblyFile).Distinct(StringComparer.OrdinalIgnoreCase).Count() != bundle.Assemblies.Length ||
            bundle.Assemblies.Count(snapshot => snapshot.AssemblyFile == bundle.TestAssembly) != 1)
            throw new EvidenceException(EvidenceFailure.Context, "Source bundle revisions or assembly identities disagree.");
        // Derive scope from the bundle's test assembly, not a caller-provided per-method flag.
        SourceMethod[] methods = MergeMethods(bundle.Assemblies.SelectMany(snapshot => Methods(snapshot,
            snapshot.AssemblyFile == bundle.TestAssembly ? SourceMethodScope.TestAssembly : SourceMethodScope.DependencyAssembly)));
        foreach (var binaries in bundle.Assemblies.SelectMany(snapshot => snapshot.ManagedDependencies?.Files ?? [])
                     .GroupBy(binary => (binary.Origin, binary.File.ToUpperInvariant())))
            if (binaries.Select(binary => (binary.Hash, binary.Origin)).Distinct().Count() != 1)
                throw new EvidenceException(EvidenceFailure.Context, "Conflicting managed dependency binaries.");
        string configuration = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(
            System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(new { bundle.TestAssembly,
                Assemblies = bundle.Assemblies.OrderBy(snapshot => snapshot.AssemblyFile, StringComparer.Ordinal)
                    .Select(snapshot => new { snapshot.AssemblyFile, ConfigurationHash = Configuration(snapshot) }).ToArray() })));
        return new(bundle.SourceTree, configuration,
            bundle.Assemblies.All(snapshot => snapshot.Status == SourceMapStatus.Verified) ? SourceMapStatus.Verified : SourceMapStatus.Unverifiable,
            methods, bundle.Assemblies.Single(snapshot => snapshot.AssemblyFile == bundle.TestAssembly).LifecycleMap);
    }

    private static IEnumerable<SourceMethod> Methods(SourceSnapshot snapshot, SourceMethodScope scope) =>
        snapshot.Methods.Select(method => method with { Scope = scope }).Concat(
            (snapshot.ManagedDependencies?.Methods ?? []).Select(method => method with { Scope = SourceMethodScope.DependencyAssembly }));

    private static string Configuration(SourceSnapshot snapshot) => snapshot.ManagedDependencies is null ? snapshot.ConfigurationHash :
        Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(new
        {
            snapshot.ConfigurationHash,
            Binaries = snapshot.ManagedDependencies.Files.OrderBy(binary => binary.Origin).ThenBy(binary => binary.File, StringComparer.Ordinal).ToArray()
        })));

    private static SourceMethod[] MergeMethods(IEnumerable<SourceMethod> methods)
    {
        var result = new List<SourceMethod>();
        foreach (var group in methods.GroupBy(method => method.Dependency.Id, StringComparer.Ordinal))
        {
            SourceMethod[] definitions = group.ToArray();
            SourceMethod first = definitions[0];
            if (definitions.Length == 1) { result.Add(first); continue; }
            if (definitions.Any(method => method.Scope != SourceMethodScope.DependencyAssembly || method.Spans.Length != 0 ||
                    method.Owner != first.Owner || method.BodyHash != first.BodyHash))
                throw new EvidenceException(EvidenceFailure.Format, "Duplicate or conflicting cross-assembly method identity.");
            // The same managed method may be reached from both source modules.
            // Budget-limited/open copies must never be replaced by a less
            // conservative copy from the other traversal.
            result.Add(first with { Dependency = first.Dependency with
            {
                Calls = definitions.SelectMany(method => method.Dependency.Calls).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray(),
                SharedState = definitions.SelectMany(method => method.Dependency.SharedState).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray(),
                Boundary = definitions.All(method => method.Dependency.Boundary == DependencyBoundary.Closed)
                    ? DependencyBoundary.Closed : DependencyBoundary.Unresolved
            } });
        }
        return result.ToArray();
    }

    private static DependencySnapshot Bind(Graph snapshot, TestCaseIdentity[] inventory, ref bool fallback)
    {
        if (snapshot.Methods is null || inventory is null || inventory.Length == 0 ||
            inventory.Any(test => test is null || string.IsNullOrWhiteSpace(test.CaseId) || string.IsNullOrWhiteSpace(test.MethodId)) ||
            inventory.Select(test => test.CaseId).Distinct(StringComparer.Ordinal).Count() != inventory.Length)
            throw new EvidenceException(EvidenceFailure.Inventory, "Missing or duplicate source selection inventory.");
        var owners = snapshot.Methods.Where(method => method.Scope == SourceMethodScope.TestAssembly).GroupBy(method => method.Owner, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Select(method => method.Dependency.Id).ToArray(), StringComparer.Ordinal);
        var roots = new List<TestDependencyRoots>();
        var testOwners = inventory.Select(test => test.MethodId).ToHashSet(StringComparer.Ordinal);
        if (snapshot.LifecycleMap is SourceLifecycleMap lifecycle)
        {
            var mapped = lifecycle.Tests.ToDictionary(test => test.Owner, StringComparer.Ordinal);
            foreach (string owner in testOwners)
            {
                if (!mapped.TryGetValue(owner, out SourceTestLifecycle? test))
                    roots.Add(new(owner, ["unmapped:xunit:" + owner]));
                else roots.Add(new(owner, test.Complete ? test.Roots : [.. test.Roots, "unresolved:xunit:" + owner]));
            }
            return new(snapshot.Methods.Select(method => method.Dependency).ToArray(), roots.ToArray(), lifecycle.GroupRoots);
        }
        foreach (string owner in testOwners)
        {
            if (!owners.TryGetValue(owner, out string[]? nodes)) { fallback = true; nodes = ["unmapped:" + owner]; }
            roots.Add(new(owner, nodes));
        }
        // Test assembly helpers can run during discovery or shared fixtures even
        // when they have no dynamically observed test owner. Keep them group-wide.
        string[] groups = snapshot.Methods.Where(method => method.Scope == SourceMethodScope.TestAssembly &&
                (method.Lifecycle || !testOwners.Contains(method.Owner)))
            .Select(method => method.Dependency.Id).ToArray();
        return new(snapshot.Methods.Select(method => method.Dependency).ToArray(), roots.ToArray(), groups);
    }

    private static bool Map(Graph snapshot, ChangedLines[] edits, HashSet<string> changed)
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
