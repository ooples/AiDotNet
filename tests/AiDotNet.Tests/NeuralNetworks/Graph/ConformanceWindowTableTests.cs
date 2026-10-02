using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Keeps the committed conformance window table complete. It is reflection only, so it runs on every
/// pull request, where the windows themselves are nightly.
/// </summary>
public class ConformanceWindowTableTests
{
    [Fact]
    public void EveryWindowedModelHasAStableWindow()
    {
        var windowed = ModelContractConformanceTests.DiscoverModels()
            .Where(t => t.Namespace is not null
                        && t.Namespace.Contains(ModelContractConformanceTests.WindowedNamespace, StringComparison.OrdinalIgnoreCase))
            .Select(t => t.FullName ?? t.Name)
            .ToList();
        var table = ModelContractConformanceTests.ConformanceWindows;
        var known = new HashSet<string>(windowed, StringComparer.Ordinal);

        var missing = windowed.Where(name => !table.ContainsKey(name)).ToList();
        var stale = table.Keys.Where(name => !known.Contains(name)).OrderBy(n => n, StringComparer.Ordinal).ToList();
        var sizes = table.Values.GroupBy(w => w).ToDictionary(g => g.Key, g => g.Count());
        var oversized = sizes.Where(kv => kv.Value > ModelContractConformanceTests.WindowSize).Select(kv => kv.Key).OrderBy(w => w).ToList();

        // The lines that place each missing model: the last window while it has room, then new windows.
        var suggestion = new StringBuilder();
        int last = sizes.Count == 0 ? 0 : sizes.Keys.Max();
        int used = sizes.TryGetValue(last, out int n) ? n : 0;
        foreach (var name in missing)
        {
            if (used >= ModelContractConformanceTests.WindowSize) { last++; used = 0; }
            suggestion.AppendLine($"        [\"{name}\"] = {last},");
            used++;
        }

        int windows = sizes.Count == 0 && missing.Count == 0 ? 0 : (missing.Count > 0 ? last : sizes.Keys.Max()) + 1;
        Assert.True(missing.Count == 0 && stale.Count == 0 && oversized.Count == 0,
            $"{missing.Count} windowed model(s) have no window, {stale.Count} listed model(s) no longer exist "
            + $"({string.Join(", ", stale)}), {oversized.Count} window(s) exceed {ModelContractConformanceTests.WindowSize} "
            + $"({string.Join(", ", oversized)}). Append to ConformanceWindows in ModelContractConformanceTests.Windows.cs:"
            + Environment.NewLine + suggestion
            + $"and set WindowCount to {windows}, with one 'Conformance - VisionLanguage' shard per window in .github/test-shards.yml.");

        int highest = sizes.Count == 0 ? -1 : sizes.Keys.Max();
        Assert.Equal(ModelContractConformanceTests.WindowCount, highest + 1);
        Assert.Equal(Enumerable.Range(0, highest + 1), sizes.Keys.OrderBy(w => w));
    }
}