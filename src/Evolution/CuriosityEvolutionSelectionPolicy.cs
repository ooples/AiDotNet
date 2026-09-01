using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.Evolution;

/// <summary>
/// Samples parents according to bounded curiosity scores and rewards parents whose offspring improve an archive.
/// </summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class CuriosityEvolutionSelectionPolicy<TGenome> : IOutcomeAwareEvolutionSelectionPolicy<TGenome>
{
    private const int MaximumTrackedScores = 1_000_000;
    private readonly object _gate = new();
    private readonly Dictionary<string, double> _scores = new(StringComparer.Ordinal);

    /// <inheritdoc/>
    public string Id => "curiosity-selection";

    /// <inheritdoc/>
    public string VersionHash => "curiosity-selection-v1";

    /// <summary>Gets a deterministic snapshot of current scores.</summary>
    public IReadOnlyDictionary<string, double> Scores
    {
        get
        {
            lock (_gate) return new System.Collections.ObjectModel.ReadOnlyDictionary<string, double>(
                new Dictionary<string, double>(_scores, StringComparer.Ordinal));
        }
    }

    /// <inheritdoc/>
    public EvolutionSelection<TGenome>? Select(IEvolutionArchive<TGenome> archive, StableRandom random, int inspirationCount)
    {
        Guard.NotNull(archive);
        Guard.NotNull(random);
        if (inspirationCount < 0) throw new ArgumentOutOfRangeException(nameof(inspirationCount));
        EvolutionArchiveEntry<TGenome>[] entries = archive.Entries.ToArray();
        if (entries.Length == 0) return null;

        lock (_gate)
        {
            EvolutionArchiveEntry<TGenome> parent = WeightedSample(entries, random);
            var remaining = entries.Where(entry => entry.Evaluation.GenomeId != parent.Evaluation.GenomeId).ToList();
            var inspirations = new List<EvolutionArchiveEntry<TGenome>>();
            while (inspirations.Count < inspirationCount && remaining.Count > 0)
            {
                EvolutionArchiveEntry<TGenome> selected = WeightedSample(remaining.ToArray(), random);
                inspirations.Add(selected);
                remaining.Remove(selected);
            }
            return new EvolutionSelection<TGenome>(parent, inspirations.AsReadOnly());
        }
    }

    /// <inheritdoc/>
    public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult)
    {
        Guard.NotNull(evaluation);
        double delta = insertionResult == EvolutionArchiveInsertionResult.Inserted ||
                       insertionResult == EvolutionArchiveInsertionResult.Replaced ||
                       insertionResult == EvolutionArchiveInsertionResult.InsertedWithEviction
            ? 1.0
            : evaluation.Status == EvolutionEvaluationStatus.Completed ? -0.1 : -0.25;
        lock (_gate)
        {
            foreach (string parentId in evaluation.Lineage.ParentIds)
            {
                if (!_scores.TryGetValue(parentId, out double current))
                {
                    if (_scores.Count >= MaximumTrackedScores) continue;
                    current = 1.0;
                }
                _scores[parentId] = Math.Max(0.05, Math.Min(100.0, current + delta));
            }
        }
    }

    /// <inheritdoc/>
    public string CaptureState()
    {
        lock (_gate)
        {
            var ordered = _scores.OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
            return JsonConvert.SerializeObject(ordered, Formatting.None);
        }
    }

    /// <inheritdoc/>
    public void RestoreState(string state)
    {
        Guard.NotNull(state);
        Dictionary<string, double>? restored;
        try { restored = JsonConvert.DeserializeObject<Dictionary<string, double>>(state); }
        catch (JsonException exception) { throw new InvalidDataException("Curiosity state is invalid.", exception); }
        if (restored is null) throw new InvalidDataException("Curiosity state is empty.");
        if (restored.Count > MaximumTrackedScores) throw new InvalidDataException("Curiosity state exceeds its safety limit.");
        if (restored.Any(pair => string.IsNullOrWhiteSpace(pair.Key) ||
                                 !EvolutionDescriptorDefinition.IsFinite(pair.Value) || pair.Value < 0.05 || pair.Value > 100))
            throw new InvalidDataException("Curiosity state contains an invalid score.");
        lock (_gate)
        {
            _scores.Clear();
            foreach (KeyValuePair<string, double> pair in restored.OrderBy(item => item.Key, StringComparer.Ordinal))
                _scores.Add(pair.Key, pair.Value);
        }
    }

    private EvolutionArchiveEntry<TGenome> WeightedSample(EvolutionArchiveEntry<TGenome>[] entries, StableRandom random)
    {
        double total = entries.Sum(entry => Score(entry.Evaluation.GenomeId));
        double target = random.NextDouble() * total;
        foreach (EvolutionArchiveEntry<TGenome> entry in entries)
        {
            target -= Score(entry.Evaluation.GenomeId);
            if (target <= 0) return entry;
        }
        return entries[entries.Length - 1];
    }

    private double Score(string genomeId) => _scores.TryGetValue(genomeId, out double value) ? value : 1.0;
}
