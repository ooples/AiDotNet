using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Agentic.Tools;

/// <summary>
/// Learns which tools tend to help for which kinds of request, so an agent can be biased toward tools that
/// worked well on similar past requests instead of relying only on static tool descriptions.
/// </summary>
/// <remarks>
/// <para>
/// This is a lightweight average-reward bandit over the toolset, keyed by request similarity (token overlap).
/// Each run records an outcome per tool used (<see cref="Record"/>); <see cref="RankTools"/> orders the
/// available tools by their mean reward on similar past requests (unseen tools are ranked optimistically so
/// they still get tried), and <see cref="BuildHint"/> turns that ranking into a system-prompt nudge.
/// </para>
/// <para>The reward table is exposed via <see cref="Export"/> / <see cref="Import"/> so it can be persisted
/// across sessions.</para>
/// <para><b>For Beginners:</b> The agent remembers which tools were useful for questions like this one before,
/// and gently prefers them next time — so it gets better at picking the right tool with experience.</para>
/// </remarks>
public sealed class LearnedToolRouter
{
    private const double OptimisticReward = 1.0; // unseen tools are assumed good so they still get tried.

    // key -> (toolName -> (sumReward, count))
    private readonly Dictionary<string, Dictionary<string, (double Sum, int Count)>> _table = new();

    /// <summary>Records the outcome of using a tool for a request.</summary>
    /// <param name="request">The user request the tool was used for.</param>
    /// <param name="toolName">The tool that was used.</param>
    /// <param name="reward">The outcome reward in [0, 1] (1 = clearly helpful, 0 = not).</param>
    public void Record(string request, string toolName, double reward)
    {
        if (string.IsNullOrWhiteSpace(toolName)) return;
        double clamped = reward < 0 ? 0 : reward > 1 ? 1 : reward;
        string key = KeyOf(request);

        if (!_table.TryGetValue(key, out var perTool))
        {
            perTool = new Dictionary<string, (double, int)>();
            _table[key] = perTool;
        }

        var prior = perTool.TryGetValue(toolName, out var e) ? e : (Sum: 0.0, Count: 0);
        perTool[toolName] = (prior.Sum + clamped, prior.Count + 1);
    }

    /// <summary>Gets the mean reward observed for a tool on requests similar to <paramref name="request"/>.</summary>
    /// <returns>The mean reward, or <see cref="OptimisticReward"/> when the tool has no history for this request.</returns>
    public double ScoreFor(string request, string toolName)
    {
        string key = KeyOf(request);
        if (_table.TryGetValue(key, out var perTool) && perTool.TryGetValue(toolName, out var e) && e.Count > 0)
        {
            return e.Sum / e.Count;
        }

        return OptimisticReward;
    }

    /// <summary>Orders the available tools by learned preference for the request (highest mean reward first).</summary>
    public IReadOnlyList<string> RankTools(string request, IEnumerable<string> availableToolNames)
    {
        if (availableToolNames is null) throw new ArgumentNullException(nameof(availableToolNames));
        return availableToolNames
            .Distinct()
            .OrderByDescending(t => ScoreFor(request, t))
            .ThenBy(t => t, StringComparer.Ordinal)
            .ToList();
    }

    /// <summary>
    /// Builds a short system-prompt hint that nudges the model toward the tools that worked best for similar
    /// requests, or <c>null</c> when there is no signal yet (nothing to nudge with).
    /// </summary>
    public string? BuildHint(string request, IEnumerable<string> availableToolNames)
    {
        string key = KeyOf(request);
        if (!_table.TryGetValue(key, out _)) return null; // no history for this kind of request.

        var ranked = RankTools(request, availableToolNames);
        var preferred = ranked
            .Where(t => ScoreFor(request, t) > 0.5)
            .Take(3)
            .ToList();
        if (preferred.Count == 0) return null;

        return "Based on similar past requests, these tools have been most useful: " +
               string.Join(", ", preferred) + ". Prefer them when appropriate.";
    }

    /// <summary>Exports the learned reward table for persistence.</summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, (double Sum, int Count)>> Export()
        => _table.ToDictionary(
            kv => kv.Key,
            kv => (IReadOnlyDictionary<string, (double, int)>)kv.Value.ToDictionary(t => t.Key, t => t.Value));

    /// <summary>Imports a previously exported reward table, replacing any current state.</summary>
    public void Import(IReadOnlyDictionary<string, IReadOnlyDictionary<string, (double Sum, int Count)>> table)
    {
        if (table is null) throw new ArgumentNullException(nameof(table));
        _table.Clear();
        foreach (var kv in table)
        {
            _table[kv.Key] = kv.Value.ToDictionary(t => t.Key, t => t.Value);
        }
    }

    /// <summary>Buckets a request into a similarity key from its salient tokens (order-independent).</summary>
    private static string KeyOf(string request)
    {
        if (string.IsNullOrWhiteSpace(request)) return string.Empty;

        var tokens = request
            .ToLowerInvariant()
            .Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '?', '!', ':', ';', '(', ')', '"', '\'' },
                StringSplitOptions.RemoveEmptyEntries)
            .Where(w => w.Length > 3) // drop short stop-words so similar requests share a key.
            .Distinct()
            .OrderBy(w => w, StringComparer.Ordinal)
            .Take(6);

        return string.Join(" ", tokens);
    }
}
