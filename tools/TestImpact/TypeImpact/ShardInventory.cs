using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace AiDotNet.TestImpact.TypeImpact;

/// <summary>
/// The shard manifest's completeness check: every test method in a sharded project is selected by
/// some shard's filter, or carries a category the manifest deliberately keeps out of the PR gate.
/// </summary>
/// <remarks>
/// The shards select by hand-written FullyQualifiedName prefixes, so a test class added under a
/// namespace no prefix names matches nothing, never runs - not on a pull request, not on the full
/// matrix - and nothing reports it. Measured on 2026-09-30: 265 classes (2,500+ tests) were in that
/// state, and running them found real failures. The classes found then live in the
/// <c>Unassigned</c> shards; this check keeps the count at zero by failing the build when a class
/// joins them, with the exact clause that assigns it.
///
/// A category is a deliberate exclusion when the manifest itself says so - some shard filter has
/// <c>Category!=X</c> - or when the caller names it with <c>--excluded-category</c> (the categories
/// Invoke-Shard.ps1 appends to every filter). Deriving the set from the manifest keeps this check
/// from needing its own list to maintain.
///
/// Selection is three-valued like the impact plan: a filter that might select a method (a trait the
/// metadata cannot pin down) counts as selecting it, so this reports only what certainly never runs.
/// </remarks>
internal static class ShardInventory
{
    private static readonly Regex ExcludedCategory = new(@"Category\s*!=\s*([^&|()\s]+)", RegexOptions.None, TimeSpan.FromSeconds(1));

    public static int Run(AssemblyIndex index, Options options)
    {
        var manifest = JsonNode.Parse(File.ReadAllText(options.Shards))?.AsArray()
            ?? throw new InvalidDataException($"{options.Shards} is not a JSON array of shards");

        var report = new JsonArray();
        int unassignedTotal = 0;
        foreach (var (project, assembly) in options.Projects.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var filters = new List<VsTestFilter>();
            var deliberate = new SortedSet<string>(options.ExcludedCategories, StringComparer.OrdinalIgnoreCase);
            bool unfiltered = false;
            foreach (var shard in manifest)
            {
                if (shard is null || (string?)shard["project"] != project)
                {
                    continue;
                }

                var text = (string?)shard["filter"] ?? string.Empty;
                if (text.Length == 0)
                {
                    unfiltered = true;
                    continue;
                }

                filters.Add(VsTestFilter.Parse(text));
                foreach (Match match in ExcludedCategory.Matches(text))
                {
                    deliberate.Add(match.Groups[1].Value);
                }
            }

            var classes = index.Nodes.SelectMany(n => n.TestClasses).Where(c => c.Node.Assembly == assembly).ToList();
            var unassigned = new List<(string Name, int Methods)>();
            int excludedClasses = 0;
            if (!unfiltered)
            {
                foreach (var testClass in classes)
                {
                    int lost = 0;
                    bool anyExcluded = false;
                    foreach (var method in testClass.Methods)
                    {
                        if (filters.Any(f => f.Evaluate(method) != Tri.False))
                        {
                            continue;
                        }

                        if (method.CertainCategories.Concat(method.PossibleCategories).Any(deliberate.Contains))
                        {
                            anyExcluded = true;
                            continue;
                        }

                        lost++;
                    }

                    if (lost > 0)
                    {
                        unassigned.Add((testClass.VsTestName, lost));
                    }
                    else if (anyExcluded)
                    {
                        excludedClasses++;
                    }
                }
            }

            unassigned.Sort((a, b) => StringComparer.Ordinal.Compare(a.Name, b.Name));
            unassignedTotal += unassigned.Count;
            Console.WriteLine($"{project}: {classes.Count} test class(es), {filters.Count} shard filter(s)" +
                (unfiltered ? " (an unfiltered shard runs everything)" : string.Empty) +
                $"; {unassigned.Count} selected by no shard, {excludedClasses} kept out by deliberate categories ({string.Join(", ", deliberate)})");
            foreach (var (name, methods) in unassigned)
            {
                Console.WriteLine($"::error::{name} ({methods} test(s)) is selected by no shard in {project}. " +
                    $"Add 'FullyQualifiedName~{VsTestFilter.Escape(name)}.' to the shard that should own it (or to an 'Unassigned' shard) in .github/test-shards.yml.");
            }

            report.Add(new JsonObject
            {
                ["project"] = project,
                ["testClasses"] = classes.Count,
                ["shardFilters"] = filters.Count,
                ["unfilteredShard"] = unfiltered,
                ["deliberateCategories"] = new JsonArray(deliberate.Select(c => (JsonNode)c).ToArray()),
                ["deliberatelyExcludedClasses"] = excludedClasses,
                ["unassigned"] = new JsonArray(unassigned.Select(u => (JsonNode)new JsonObject { ["class"] = u.Name, ["tests"] = u.Methods }).ToArray()),
            });
        }

        var result = new JsonObject { ["schemaVersion"] = 1, ["unassignedClasses"] = unassignedTotal, ["projects"] = report };
        File.WriteAllText(options.Out, result.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"{unassignedTotal} test class(es) selected by no shard; wrote {options.Out}");
        return unassignedTotal == 0 ? 0 : 1;
    }
}
