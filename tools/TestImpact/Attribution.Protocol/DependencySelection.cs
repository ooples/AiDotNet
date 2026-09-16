namespace AiDotNet.TestImpact;

public enum DependencyBoundary { Closed, Unresolved, VirtualDispatch, Reflection, External, Native }
public enum SelectionReason { ChangedDependency, NewTest, OpenDependency, SharedState, GroupDependency, UnmappedChange }
public sealed record DependencyNode(string Id, string[] Calls, string[] SharedState, DependencyBoundary Boundary);
public sealed record TestDependencyRoots(string MethodId, string[] Roots);
public sealed record DependencySnapshot(DependencyNode[] Nodes, TestDependencyRoots[] Tests, string[] GroupRoots);
public sealed record SelectedMethod(string MethodId, SelectionReason Reason);

// Pure conservative closure, not a trusted evidence importer. The caller must
// supply complete old/new graphs, independently discovered test roots, and ALL
// changed inputs. An omitted change cannot be detected by this algorithm.
// Source/configuration changes without a mapped node require unmappedChange=true.
public static class DependencySelection
{
    private sealed record Closure(HashSet<string> Nodes, HashSet<string> State, bool Open);
    private sealed record Indexed(Dictionary<string, DependencyNode> Nodes, Dictionary<string, TestDependencyRoots> Tests);

    public static IReadOnlyList<SelectedMethod> Select(DependencySnapshot before, DependencySnapshot after,
        string[] changedNodes, bool unmappedChange)
    {
        Indexed old = Index(before);
        Indexed current = Index(after);
        HashSet<string> changed = Unique(changedNodes);
        // This is not a successful zero-test execution certificate. The execution
        // protocol separately rejects an empty runtime plan.
        if (unmappedChange || changed.Any(id => !old.Nodes.ContainsKey(id) && !current.Nodes.ContainsKey(id)))
            return All(current, SelectionReason.UnmappedChange);
        Closure oldGroup = Walk(old, before.GroupRoots);
        Closure newGroup = Walk(current, after.GroupRoots);
        if (Impacted(oldGroup, changed) || Impacted(newGroup, changed))
            return All(current, SelectionReason.GroupDependency);

        var closures = new Dictionary<string, Closure>(StringComparer.Ordinal);
        var selected = new Dictionary<string, SelectionReason>(StringComparer.Ordinal);
        foreach ((string id, TestDependencyRoots test) in current.Tests)
        {
            Closure next = Walk(current, test.Roots);
            Closure previous = old.Tests.TryGetValue(id, out TestDependencyRoots? prior)
                ? Walk(old, prior.Roots) : new(new(StringComparer.Ordinal), new(StringComparer.Ordinal), false);
            // Union old and new edges: a deleted call or removed node still
            // affects its previous callers; a newly introduced branch also counts.
            next.Nodes.UnionWith(previous.Nodes);
            next.State.UnionWith(previous.State);
            next = next with { Open = next.Open || previous.Open };
            closures.Add(id, next);
            if (prior is null) selected.Add(id, SelectionReason.NewTest);
            else if (next.Open) selected.Add(id, SelectionReason.OpenDependency);
            else if (next.Nodes.Overlaps(changed)) selected.Add(id, SelectionReason.ChangedDependency);
        }

        // A selected test can mutate state consumed by another test. Close over
        // shared state until stable, across both revisions and indirect callers.
        var affectedState = new HashSet<string>(StringComparer.Ordinal);
        foreach ((string id, TestDependencyRoots removed) in old.Tests)
        {
            if (current.Tests.ContainsKey(id)) continue;
            Closure previous = Walk(old, removed.Roots);
            if (previous.Open) return All(current, SelectionReason.OpenDependency);
            affectedState.UnionWith(previous.State);
        }
        bool expanded;
        do
        {
            foreach (string id in selected.Keys) affectedState.UnionWith(closures[id].State);
            if (oldGroup.State.Overlaps(affectedState) || newGroup.State.Overlaps(affectedState))
                return All(current, SelectionReason.SharedState);
            expanded = false;
            foreach ((string id, Closure closure) in closures)
                if (!selected.ContainsKey(id) && closure.State.Overlaps(affectedState))
                {
                    selected.Add(id, SelectionReason.SharedState);
                    expanded = true;
                }
        } while (expanded);
        return Array.AsReadOnly(selected.OrderBy(pair => pair.Key, StringComparer.Ordinal)
            .Select(pair => new SelectedMethod(pair.Key, pair.Value)).ToArray());
    }

    private static bool Impacted(Closure closure, HashSet<string> changed) =>
        closure.Open || closure.Nodes.Overlaps(changed);

    private static IReadOnlyList<SelectedMethod> All(Indexed graph, SelectionReason reason) => Array.AsReadOnly(
        graph.Tests.Keys.Order(StringComparer.Ordinal).Select(id => new SelectedMethod(id, reason)).ToArray());

    private static Closure Walk(Indexed graph, string[] roots)
    {
        var nodes = new HashSet<string>(StringComparer.Ordinal);
        var state = new HashSet<string>(StringComparer.Ordinal);
        var pending = new Stack<string>(roots);
        bool open = false;
        while (pending.TryPop(out string? id))
        {
            if (!nodes.Add(id)) continue;
            if (!graph.Nodes.TryGetValue(id, out DependencyNode? node)) { open = true; continue; }
            open |= node.Boundary != DependencyBoundary.Closed;
            state.UnionWith(node.SharedState);
            foreach (string call in node.Calls) pending.Push(call);
        }
        return new(nodes, state, open);
    }

    private static Indexed Index(DependencySnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        ArgumentNullException.ThrowIfNull(snapshot.Nodes);
        ArgumentNullException.ThrowIfNull(snapshot.Tests);
        var nodes = new Dictionary<string, DependencyNode>(StringComparer.Ordinal);
        var tests = new Dictionary<string, TestDependencyRoots>(StringComparer.Ordinal);
        foreach (DependencyNode node in snapshot.Nodes)
        {
            ArgumentNullException.ThrowIfNull(node);
            Text(node.Id);
            if (!Enum.IsDefined(node.Boundary) || !nodes.TryAdd(node.Id, node))
                throw new InvalidDataException("Invalid or duplicate dependency node.");
            Unique(node.Calls);
            Unique(node.SharedState);
        }
        foreach (TestDependencyRoots test in snapshot.Tests)
        {
            ArgumentNullException.ThrowIfNull(test);
            Text(test.MethodId);
            if (!tests.TryAdd(test.MethodId, test) || Unique(test.Roots).Count == 0)
                throw new InvalidDataException("Duplicate test identity or missing roots.");
        }
        Unique(snapshot.GroupRoots);
        return new(nodes, tests);
    }

    private static HashSet<string> Unique(string[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        var unique = new HashSet<string>(StringComparer.Ordinal);
        foreach (string value in values)
        {
            Text(value);
            if (!unique.Add(value)) throw new InvalidDataException("Duplicate dependency identity.");
        }
        return unique;
    }

    private static void Text(string value)
    {
        if (string.IsNullOrWhiteSpace(value)) throw new InvalidDataException("Missing dependency identity.");
    }
}
