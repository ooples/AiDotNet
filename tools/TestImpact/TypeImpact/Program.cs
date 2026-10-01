using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using AiDotNet.TestImpact.TypeImpact;

// Test-level impact selection from compiled assemblies.
//
// The coverage map selects whole shards, and every shard executes the shared base classes, so in
// practice it selected 50-164 of 164 shards for every pull request. This reads the Build job's
// own output instead: each changed source file is mapped through the portable PDBs to the types it
// defines, the reverse type-reference graph (base types, signatures, attributes and every token in
// every method body, across src and tests, generated tests included) gives every type the change
// can reach, and the concrete xUnit classes among them are the tests to run. Each shard is then
// narrowed to the selected classes it owns, or dropped when it owns none.
//
// It is sound for statically reachable code. It cannot see a type reached only by reflection over
// a name or by enumerating an assembly; a test class that enumerates types itself is selected for
// every source change, and the nightly full matrix on master is the net for the rest. Anything it
// cannot map - a non-C# file, a source with no compiled type, a deleted non-C# file - makes the
// whole plan unresolved, and the caller keeps its own (coverage or full) selection.
//
// Usage:
//   TypeImpact --repo <root> --bin <dir> [--bin <dir>...] --project <csproj>=<assembly> [...]
//              --changes <git diff --name-status output> --shards <shard-manifest.json>
//              --out <plan.json> [--max-classes <n>]
//   TypeImpact --inventory --repo <root> --bin <dir> [...] --project <csproj>=<assembly> [...]
//              --shards <shard-manifest.json> --out <inventory.json> [--excluded-category <name>...]
//              (exit 1 when a test class is selected by no shard; see ShardInventory.cs)

var options = Options.Parse(args);
var clock = Stopwatch.StartNew();
var index = AssemblyIndex.Load(options.Bins, options.Repo);
Console.WriteLine($"indexed {index.Nodes.Count} types in {index.AssemblyNames.Count} assemblies in {clock.Elapsed.TotalSeconds:F1}s");
if (options.Inventory)
{
    return ShardInventory.Run(index, options);
}

var unresolved = new List<JsonObject>();
var ignored = new List<string>();
var changed = new HashSet<TypeNode>();
var perFile = new JsonArray();
var constLines = options.Diff.Length == 0 ? null : FilesEditingConstLines(options.Diff);
foreach (var (status, path) in ReadChanges(options.Changes))
{
    if (IsDocumentation(path))
    {
        ignored.Add(path);
        continue;
    }

    if (!path.EndsWith(".cs", StringComparison.OrdinalIgnoreCase))
    {
        unresolved.Add(new JsonObject { ["path"] = path, ["reason"] = "not a C# source; its effect on tests cannot be read from the assemblies" });
        continue;
    }

    if (options.Unmappable.Any(prefix => path.StartsWith(prefix, StringComparison.Ordinal)))
    {
        unresolved.Add(new JsonObject { ["path"] = path, ["reason"] = "source generator input: it rewrites generated code in every assembly" });
        continue;
    }

    if (status == 'D')
    {
        // The new assemblies cannot show who depended on it: callers may now bind to another
        // overload or extension method, and a reflection inventory has lost a type.
        unresolved.Add(new JsonObject { ["path"] = path, ["reason"] = "deleted C# source: its former dependents cannot be read from the new assemblies" });
        continue;
    }

    var types = index.TypesInDocument(path);
    if (types is null || types.Count == 0)
    {
        unresolved.Add(new JsonObject { ["path"] = path, ["reason"] = "no compiled type in the loaded assemblies comes from this source" });
        continue;
    }

    // Consumers inline a const's value and keep no reference to its type, so an edited const line
    // in a type that exposes one cannot be followed. Without the diff, any such file is unmappable.
    if (types.Any(t => t.DeclaresVisibleConstant) &&
        (constLines is null || constLines.Contains(path)))
    {
        unresolved.Add(new JsonObject { ["path"] = path, ["reason"] = "edits a non-private const: consumers inline its value and hold no reference to it" });
        continue;
    }

    changed.UnionWith(types);
    perFile.Add(new JsonObject { ["path"] = path, ["types"] = types.Count });
}

// A type that names more than --dispatch-threshold others is a dispatch table (a name-to-type
// registry, a generated catalog): which entry a caller reaches is runtime data, exactly the blind
// spot reflection has. Propagating through one would select every test of every model it lists,
// so a change stops at it unless the table itself changed.
// A catalog also shows as a type that names hundreds of types while only a handful of entry points
// reach it (GeneratedLayerFactories: 370 out, 3 in), unlike a facade such as AiModelBuilder
// (561 out, 137 in) whose references are real calls every caller makes.
var dispatchTables = index.Nodes.Where(n => n.TestClasses.Count == 0 &&
    (n.References.Count >= options.DispatchThreshold ||
     (n.References.Count >= options.CatalogThreshold && n.ReferencedBy.Count <= options.CatalogMaxEntryPoints))).ToHashSet();
if (options.Explain)
{
    foreach (var table in dispatchTables.OrderByDescending(n => n.References.Count))
    {
        Console.WriteLine($"  dispatch table: out {table.References.Count,5} in {table.ReferencedBy.Count,4}  {table.Key}");
    }
}
var affected = ReverseClosure(changed, dispatchTables, out var via);
if (options.Explain)
{

    // The first-found path back to a changed type for each reachable non-test type with the most
    // dependents: the hubs that widen a selection.
    foreach (var hub in affected.Where(n => n.TestClasses.Count == 0).OrderByDescending(n => n.ReferencedBy.Count(affected.Contains)).Take(25))
    {
        var chain = new List<string>();
        for (var step = hub; step is not null; step = via.GetValueOrDefault(step))
        {
            chain.Add(step.FullName);
        }

        Console.WriteLine($"  hub {hub.ReferencedBy.Count(affected.Contains),5} dependents: {string.Join(" <- ", chain.AsEnumerable().Reverse())}");
    }
}

// Every test that reaches type enumeration inside the test assemblies - the enumerating class itself
// or a test-side helper it uses (LayerTestBase sweeps every activation function for all 222 layer
// tests) - depends on the enumerated set, which no signature names. A production-side enumerator is a
// name-to-type registry (DeserializationHelper, ModelTypeRegistry): which entry a caller reaches is
// runtime data, the same blind spot as a catalog, left to the nightly full run.
var testAssemblies = index.TestAssemblies;
var enumerating = ReverseClosure(index.Nodes.Where(n => n.Enumerates && testAssemblies.Contains(n.Assembly)), dispatchTables, out _)
    .Where(n => n.TestClasses.Count > 0)
    .ToHashSet();
// Inventories react to production code: an edit confined to test sources cannot move the set of
// types they enumerate.
bool anySource = changed.Any(n => !index.TestAssemblies.Contains(n.Assembly));
var selectedNodes = new HashSet<TypeNode>(affected.Where(n => n.TestClasses.Count > 0));
if (anySource)
{
    selectedNodes.UnionWith(enumerating);
}

var selectedClasses = selectedNodes.SelectMany(n => n.TestClasses).OrderBy(c => c.VsTestName, StringComparer.Ordinal).ToList();
var allClasses = index.Nodes.SelectMany(n => n.TestClasses).ToList();
Console.WriteLine($"changed {changed.Count} type(s); {affected.Count} type(s) reachable; " +
    $"{selectedClasses.Count} of {allClasses.Count} test class(es) selected ({enumerating.Count} enumerate types by reflection)");

var shardPlans = new JsonArray();
var manifest = JsonNode.Parse(File.ReadAllText(options.Shards))!.AsArray();
foreach (var shard in manifest)
{
    var name = (string)shard!["name"]!;
    var project = (string)shard["project"]!;
    var filterText = (string?)shard["filter"] ?? string.Empty;
    if (!options.Projects.TryGetValue(project, out var assembly))
    {
        shardPlans.Add(new JsonObject { ["name"] = name, ["run"] = true, ["narrowed"] = false, ["reason"] = "project not loaded" });
        continue;
    }

    // A nightlyOnly shard (a sweep or conformance window over every model) runs on a pull request
    // only when the pull request edits one of its own test classes; reachability alone defers it
    // to the nightly full run. This is the rule Get-DeferredNightlyShards applies to coverage routes.
    bool nightlyOnly = shard["nightlyOnly"]?.GetValue<bool>() == true;
    var filter = filterText.Length == 0 ? null : VsTestFilter.Parse(filterText);
    var owned = selectedClasses
        .Where(c => c.Node.Assembly == assembly)
        .Where(c => !nightlyOnly || changed.Contains(c.Node))
        .Where(c => c.Methods.Any(m => filter is null || filter.Evaluate(m) != Tri.False))
        .Select(c => c.VsTestName)
        .ToList();
    if (owned.Count == 0)
    {
        shardPlans.Add(new JsonObject { ["name"] = name, ["run"] = false, ["narrowed"] = false, ["classes"] = 0 });
        continue;
    }

    // A nightlyOnly shard partitions one sweep by environment; it runs whole or not at all.
    bool narrow = !nightlyOnly && owned.Count <= options.MaxClasses;
    var plan = new JsonObject { ["name"] = name, ["run"] = true, ["narrowed"] = narrow, ["classes"] = owned.Count };
    if (narrow)
    {
        var classFilter = string.Join('|', owned.Select(c => "FullyQualifiedName~" + Escape(c) + "."));
        plan["filter"] = filterText.Length == 0 ? classFilter : $"({filterText})&({classFilter})";
        plan["testClasses"] = new JsonArray(owned.Select(c => (JsonNode)c).ToArray());
    }

    shardPlans.Add(plan);
}

var result = new JsonObject
{
    ["schemaVersion"] = 1,
    ["resolved"] = unresolved.Count == 0,
    ["unresolved"] = new JsonArray(unresolved.ToArray<JsonNode>()),
    ["ignored"] = new JsonArray(ignored.Select(p => (JsonNode)p).ToArray()),
    ["changedFiles"] = perFile,
    ["changedTypes"] = changed.Count,
    ["reachableTypes"] = affected.Count,
    ["dispatchTables"] = new JsonArray(dispatchTables.Select(n => (JsonNode)n.Key).OrderBy(k => (string)k!, StringComparer.Ordinal).ToArray()),
    ["totalTestClasses"] = allClasses.Count,
    ["selectedTestClasses"] = selectedClasses.Count,
    ["enumeratingTestClasses"] = new JsonArray(enumerating.SelectMany(n => n.TestClasses).Select(c => (JsonNode)c.VsTestName).ToArray()),
    ["shards"] = shardPlans,
    ["seconds"] = Math.Round(clock.Elapsed.TotalSeconds, 1),
};
File.WriteAllText(options.Out, result.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
int running = shardPlans.Count(s => (bool)s!["run"]!);
Console.WriteLine($"resolved={unresolved.Count == 0}; {running} of {manifest.Count} shard(s) own a selected test; wrote {options.Out}");
foreach (var item in unresolved)
{
    Console.WriteLine($"  unresolved: {item["path"]} - {item["reason"]}");
}

return 0;

static HashSet<TypeNode> ReverseClosure(IEnumerable<TypeNode> roots, HashSet<TypeNode> barriers, out Dictionary<TypeNode, TypeNode> via)
{
    via = [];
    var seen = new HashSet<TypeNode>(roots);
    var queue = new Queue<TypeNode>(seen);
    while (queue.Count > 0)
    {
        var current = queue.Dequeue();
        foreach (var dependent in current.ReferencedBy)
        {
            if (!barriers.Contains(dependent) && seen.Add(dependent))
            {
                via[dependent] = current;
                queue.Enqueue(dependent);
            }
        }
    }

    return seen;
}

// Files whose added or removed lines in a unified diff mention `const`.
static HashSet<string> FilesEditingConstLines(string diffFile)
{
    var result = new HashSet<string>(StringComparer.Ordinal);
    var constWord = new System.Text.RegularExpressions.Regex(@"\bconst\b");
    string? current = null;
    foreach (var line in File.ReadLines(diffFile))
    {
        if (line.StartsWith("+++ ", StringComparison.Ordinal))
        {
            current = line == "+++ /dev/null" ? null : line[4..].TrimStart('b').TrimStart('/');
            continue;
        }

        if (line.StartsWith("--- ", StringComparison.Ordinal))
        {
            // A deleted file has no "+++ b/" side; name it from the old side.
            current = line == "--- /dev/null" ? current : line[4..].TrimStart('a').TrimStart('/');
            continue;
        }

        if (current is not null && line.Length > 0 && line[0] is '+' or '-' && constWord.IsMatch(line))
        {
            result.Add(current);
        }
    }

    return result;
}

static IEnumerable<(char Status, string Path)> ReadChanges(string file)
{
    foreach (var raw in File.ReadAllLines(file))
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            continue;
        }

        var fields = raw.Split('\t');
        if (fields.Length < 2)
        {
            throw new FormatException($"expected 'git diff --name-status' lines, got: {raw}");
        }

        // Renames and copies carry the old path first; the new one is what was built.
        yield return (fields[0][0], fields[^1].Replace('\\', '/'));
    }
}

static bool IsDocumentation(string path)
{
    var extension = Path.GetExtension(path).ToLowerInvariant();
    return extension is ".md" or ".txt" or ".png" or ".jpg" or ".svg" or ".gif"
        && !path.StartsWith("src/", StringComparison.Ordinal) && !path.StartsWith("tests/", StringComparison.Ordinal)
        || path.StartsWith("docs/", StringComparison.Ordinal);
}

static string Escape(string value) => VsTestFilter.Escape(value);

internal sealed class Options
{
    public string Repo { get; private set; } = Directory.GetCurrentDirectory();
    public List<string> Bins { get; } = [];
    public Dictionary<string, string> Projects { get; } = new(StringComparer.Ordinal);
    public string Changes { get; private set; } = string.Empty;
    public string Shards { get; private set; } = string.Empty;
    public string Out { get; private set; } = "type-impact-plan.json";
    public string Diff { get; private set; } = string.Empty;
    public int MaxClasses { get; private set; } = 400;
    public bool Explain { get; private set; }
    public int DispatchThreshold { get; private set; } = 1000;
    public int CatalogThreshold { get; private set; } = 250;
    public int CatalogMaxEntryPoints { get; private set; } = 10;
    public List<string> Unmappable { get; } = [];
    public bool Inventory { get; private set; }
    public List<string> ExcludedCategories { get; } = [];

    public static Options Parse(string[] args)
    {
        var options = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
            switch (args[i])
            {
                case "--repo": options.Repo = Path.GetFullPath(Next()); break;
                case "--bin": options.Bins.Add(Path.GetFullPath(Next())); break;
                case "--project":
                    var pair = Next().Split('=', 2);
                    if (pair.Length != 2) { throw new ArgumentException("--project takes <csproj>=<assembly name>"); }
                    options.Projects[pair[0].Replace('\\', '/')] = pair[1];
                    break;
                case "--changes": options.Changes = Next(); break;
                case "--shards": options.Shards = Next(); break;
                case "--out": options.Out = Next(); break;
                case "--diff": options.Diff = Next(); break;
                case "--explain": options.Explain = true; break;
                case "--unmappable": options.Unmappable.Add(Next().Replace('\\', '/')); break;
                case "--catalog-threshold": options.CatalogThreshold = int.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture); break;
                case "--catalog-max-entry-points": options.CatalogMaxEntryPoints = int.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture); break;
                case "--dispatch-threshold": options.DispatchThreshold = int.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture); break;
                case "--inventory": options.Inventory = true; break;
                case "--excluded-category": options.ExcludedCategories.Add(Next()); break;
                case "--max-classes": options.MaxClasses = int.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture); break;
                default: throw new ArgumentException($"unknown argument {args[i]}");
            }
        }

        if (options.Bins.Count == 0 || options.Shards.Length == 0 || (!options.Inventory && options.Changes.Length == 0))
        {
            throw new ArgumentException("--bin and --shards are required, and --changes unless --inventory");
        }

        return options;
    }
}
