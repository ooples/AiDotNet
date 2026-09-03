using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Evolution;

public sealed partial class EvolutionEngine<TGenome>
{
    /// <summary>The largest diagnostic count one merged cascade result may carry, matching the task-result bound.</summary>
    private const int MaximumCascadeDiagnostics = 64;

    /// <summary>The upper bound applied to a computed retry delay so a misconfigured backoff cannot stall a run.</summary>
    private static readonly TimeSpan MaximumRetryDelay = TimeSpan.FromMinutes(1);

    /// <summary>
    /// Runs the ordered cascade stages, gating each non-final stage on its configured threshold and merging the
    /// descriptors, metrics, artifacts, diagnostics, and cost units of every stage that actually executed.
    /// </summary>
    private async Task<EvolutionTaskResult> EvaluateCascadeAsync(WorkItem item, ICascadeEvolutionTask<TGenome> cascade,
        CancellationToken cancellationToken)
    {
        if (item.Candidate is null) return EvolutionTaskResult.Failed("missing_candidate", "The evaluator candidate was unavailable.");
        int stageCount = cascade.StageCount;
        if (stageCount != _cascadeStageCount)
            return EvolutionTaskResult.Failed("cascade_stage_count_changed",
                "The cascade task changed its stage count after the engine validated it.");

        var stageCosts = new List<double>(stageCount);
        var descriptors = new Dictionary<string, double>(StringComparer.Ordinal);
        var metrics = new Dictionary<string, double>(StringComparer.Ordinal);
        var artifacts = new List<EvolutionArtifact>();
        var diagnostics = new List<EvolutionDiagnostic>();
        double totalCost = 0;
        double quality = 0;
        EvolutionOptimizationDirection direction = _islands[0].Direction;

        for (int stage = 0; stage < stageCount; stage++)
        {
            EvolutionCandidate<TGenome> candidate = item.Candidate;
            int stageIndex = stage;
            EvolutionTaskResult stageResult = await InvokeEvaluatorAsync(item, stageIndex,
                StageTimeout(stageIndex),
                (context, token) => cascade.EvaluateStageAsync(stageIndex, candidate, context, token),
                cancellationToken).ConfigureAwait(false);

            stageCosts.Add(stageResult.CostUnits);
            totalCost = SaturatingAdd(totalCost, stageResult.CostUnits, diagnostics);
            foreach (KeyValuePair<string, double> descriptor in stageResult.Descriptors) descriptors[descriptor.Key] = descriptor.Value;
            foreach (KeyValuePair<string, double> metric in stageResult.Metrics) metrics[metric.Key] = metric.Value;
            artifacts.AddRange(stageResult.Artifacts);
            foreach (EvolutionDiagnostic diagnostic in stageResult.Diagnostics) AddCascadeDiagnostic(diagnostics, diagnostic);
            item.StageCostUnits = stageCosts.ToArray();

            if (stageResult.Status != EvolutionEvaluationStatus.Completed)
            {
                item.CascadeRejectedStage = null;
                AddCascadeDiagnostic(diagnostics, new EvolutionDiagnostic("cascade_stage_ended",
                    $"Cascade stage {stageIndex} ended with status {stageResult.Status}.",
                    isRedacted: false, data: StageData(stageIndex, item.AttemptCount, stageResult.Status.ToString())));
                return new EvolutionTaskResult(stageResult.Status, stageResult.Quality, stageResult.Direction,
                    descriptors, stageResult.Objectives, stageResult.ConstraintViolations, totalCost, diagnostics,
                    metrics, BoundArtifacts(artifacts));
            }

            quality = stageResult.Quality ?? 0;
            direction = stageResult.Direction;
            if (stageIndex == stageCount - 1) break;

            double threshold = _options.Cascade.Thresholds[stageIndex];
            if (!PassesThreshold(quality, threshold, direction))
            {
                item.CascadeRejectedStage = stageIndex;
                AddCascadeDiagnostic(diagnostics, new EvolutionDiagnostic("cascade_stage_rejected",
                    $"Cascade stage {stageIndex} scored {quality.ToString("R", CultureInfo.InvariantCulture)} " +
                    $"against threshold {threshold.ToString("R", CultureInfo.InvariantCulture)}.",
                    isRedacted: false, data: RejectionData(stageIndex, threshold, quality, direction)));
                return new EvolutionTaskResult(EvolutionEvaluationStatus.Skipped, null, direction,
                    descriptors, Array.Empty<double>(), Array.Empty<double>(), totalCost, diagnostics,
                    metrics, BoundArtifacts(artifacts));
            }
        }

        item.CascadeRejectedStage = null;
        return new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, quality, direction, descriptors,
            Array.Empty<double>(), Array.Empty<double>(), totalCost, diagnostics, metrics, BoundArtifacts(artifacts));
    }

    /// <summary>Returns whether a stage's quality clears its gate in the stage's own optimization direction.</summary>
    private static bool PassesThreshold(double quality, double threshold, EvolutionOptimizationDirection direction) =>
        direction == EvolutionOptimizationDirection.Maximize ? quality >= threshold : quality <= threshold;

    /// <summary>Returns the configured timeout for one stage, falling back to the per-attempt evaluation timeout.</summary>
    private TimeSpan? StageTimeout(int stage)
    {
        IList<TimeSpan> timeouts = _options.Cascade.StageTimeouts;
        return timeouts.Count == 0 ? _options.EvaluationTimeout : timeouts[stage];
    }

    /// <summary>Adds two cost values, saturating at the finite maximum and recording a diagnostic when it saturates.</summary>
    private static double SaturatingAdd(double accumulated, double addition, List<EvolutionDiagnostic> diagnostics)
    {
        if (addition > double.MaxValue - accumulated)
        {
            AddCascadeDiagnostic(diagnostics, new EvolutionDiagnostic("cost_units_saturated",
                "Accumulated evaluator cost exceeded the finite numeric range."));
            return double.MaxValue;
        }
        return accumulated + addition;
    }

    /// <summary>Appends a diagnostic, replacing the last slot with a truncation marker once the public bound is reached.</summary>
    private static void AddCascadeDiagnostic(List<EvolutionDiagnostic> diagnostics, EvolutionDiagnostic diagnostic)
    {
        if (diagnostics.Count < MaximumCascadeDiagnostics)
        {
            diagnostics.Add(diagnostic);
            return;
        }
        if (diagnostics[MaximumCascadeDiagnostics - 1].Code != "diagnostics_truncated")
        {
            diagnostics[MaximumCascadeDiagnostics - 1] = new EvolutionDiagnostic(
                "diagnostics_truncated", "Additional cascade diagnostics were omitted to preserve the public bound.");
        }
    }

    /// <summary>Builds the structured context attached to a stage that ended without completing.</summary>
    private static Dictionary<string, string> StageData(int stage, int attempt, string status) => new(StringComparer.Ordinal)
    {
        ["stage"] = stage.ToString(CultureInfo.InvariantCulture),
        ["attempt"] = attempt.ToString(CultureInfo.InvariantCulture),
        ["status"] = status
    };

    /// <summary>Builds the structured context attached to a stage that failed its threshold.</summary>
    private static Dictionary<string, string> RejectionData(int stage, double threshold, double quality,
        EvolutionOptimizationDirection direction) => new(StringComparer.Ordinal)
    {
        ["stage"] = stage.ToString(CultureInfo.InvariantCulture),
        ["threshold"] = threshold.ToString("R", CultureInfo.InvariantCulture),
        ["quality"] = quality.ToString("R", CultureInfo.InvariantCulture),
        ["direction"] = direction.ToString()
    };

    /// <summary>
    /// Sanitizes, truncates, and caps evaluator artifacts against the configured budgets, returning an empty list
    /// when artifact retention is disabled.
    /// </summary>
    private IReadOnlyList<EvolutionArtifact> BoundArtifacts(IReadOnlyList<EvolutionArtifact> artifacts)
    {
        if (!_options.Artifacts.Enabled || artifacts.Count == 0) return Array.Empty<EvolutionArtifact>();
        var retained = new List<EvolutionArtifact>(Math.Min(artifacts.Count, _options.Artifacts.MaxArtifactsPerEvaluation));
        long totalBytes = 0;
        foreach (EvolutionArtifact artifact in artifacts)
        {
            if (retained.Count >= _options.Artifacts.MaxArtifactsPerEvaluation) break;
            string text = artifact.Text;
            bool redacted = artifact.IsRedacted;
            if (_options.Artifacts.SanitizeSecrets)
            {
                string sanitized = EvolutionArtifactSanitizer.Sanitize(text);
                redacted |= !string.Equals(sanitized, text, StringComparison.Ordinal);
                text = sanitized;
            }
            text = TruncateToBytes(text, _options.Artifacts.MaxArtifactBytes, out bool truncated);
            var bounded = new EvolutionArtifact(artifact.Key, text, artifact.IsTruncated || truncated, redacted);
            if (totalBytes + bounded.SizeBytes > _options.Artifacts.MaxBytesPerEvaluation) break;
            totalBytes += bounded.SizeBytes;
            retained.Add(bounded);
        }
        return retained.Count == 0 ? Array.Empty<EvolutionArtifact>() : Array.AsReadOnly(retained.ToArray());
    }

    /// <summary>Cuts text at a code-point boundary so its UTF-8 encoding fits the supplied byte budget.</summary>
    private static string TruncateToBytes(string text, int maxBytes, out bool truncated)
    {
        int used = 0;
        int index = 0;
        while (index < text.Length)
        {
            char current = text[index];
            int step = 1;
            int codePoint = current;
            if (char.IsHighSurrogate(current) && index + 1 < text.Length && char.IsLowSurrogate(text[index + 1]))
            {
                codePoint = char.ConvertToUtf32(current, text[index + 1]);
                step = 2;
            }
            int size = codePoint < 0x80 ? 1 : codePoint < 0x800 ? 2 : codePoint < 0x10000 ? 3 : 4;
            if (used + size > maxBytes) break;
            used += size;
            index += step;
        }
        truncated = index < text.Length;
        return truncated ? text.Substring(0, index) : text;
    }

    /// <summary>
    /// Queues an evaluation's artifacts under the identity whose next proposal should see them: the candidate itself
    /// when it completed, and otherwise its parent, so a failure note reaches the next sibling proposal from that
    /// lineage rather than a candidate that will never be selected as a parent.
    /// </summary>
    private void QueueLineageArtifacts(WorkItem item, EvolutionEvaluation evaluation)
    {
        if (evaluation.Artifacts.Count == 0) return;
        string? genomeId = evaluation.Status == EvolutionEvaluationStatus.Completed
            ? item.Candidate?.CanonicalGenome.Id
            : item.Lineage.ParentIds.Count > 0 ? item.Lineage.ParentIds[0] : item.Candidate?.CanonicalGenome.Id;
        if (genomeId is null || !_seen.Contains(genomeId)) return;
        QueueArtifactsForDelivery(genomeId, evaluation.Artifacts);
    }

    /// <summary>Stores artifacts against a genome identity, evicting the oldest entry when the queue is full.</summary>
    private void QueueArtifactsForDelivery(string genomeId, IReadOnlyList<EvolutionArtifact> artifacts)
    {
        if (!_options.Artifacts.Enabled || !_options.Artifacts.DeliverToNextProposal || artifacts.Count == 0) return;
        if (!_pendingArtifacts.ContainsKey(genomeId)) _pendingArtifactOrder.Add(genomeId);
        _pendingArtifacts[genomeId] = artifacts.ToArray();
        while (_pendingArtifactOrder.Count > _options.Artifacts.MaxPendingCandidates)
        {
            string oldest = _pendingArtifactOrder[0];
            _pendingArtifactOrder.RemoveAt(0);
            _pendingArtifacts.Remove(oldest);
        }
    }

    /// <summary>Removes and returns the artifacts queued for a genome, so each note informs exactly one proposal.</summary>
    private IReadOnlyList<EvolutionArtifact> ConsumeArtifacts(string genomeId)
    {
        if (!_pendingArtifacts.TryGetValue(genomeId, out EvolutionArtifact[]? artifacts))
            return Array.Empty<EvolutionArtifact>();
        _pendingArtifacts.Remove(genomeId);
        _pendingArtifactOrder.Remove(genomeId);
        return Array.AsReadOnly(artifacts);
    }

    /// <summary>Returns the pending artifact queue as an immutable view for the run result.</summary>
    private IReadOnlyDictionary<string, IReadOnlyList<EvolutionArtifact>> PendingArtifactView()
    {
        var view = new Dictionary<string, IReadOnlyList<EvolutionArtifact>>(StringComparer.Ordinal);
        foreach (string genomeId in _pendingArtifactOrder)
            if (_pendingArtifacts.TryGetValue(genomeId, out EvolutionArtifact[]? artifacts))
                view[genomeId] = Array.AsReadOnly(artifacts);
        return view;
    }

    /// <summary>Returns whether the configured target quality has been reached in the archive's direction.</summary>
    private bool IsTargetReached()
    {
        if (!_options.TargetQuality.HasValue) return false;
        double? best = BestQualityAcrossIslands();
        if (!best.HasValue) return false;
        return _islands[0].Direction == EvolutionOptimizationDirection.Maximize
            ? best.Value >= _options.TargetQuality.Value
            : best.Value <= _options.TargetQuality.Value;
    }

    /// <summary>Returns the best elite quality across every island, or <c>null</c> when every island is empty.</summary>
    private double? BestQualityAcrossIslands()
    {
        double? best = null;
        bool maximize = _islands[0].Direction == EvolutionOptimizationDirection.Maximize;
        foreach (IEvolutionArchive<TGenome> archive in _islands)
        {
            double? quality = archive.Best?.Evaluation.Quality;
            if (!quality.HasValue) continue;
            if (!best.HasValue || (maximize ? quality.Value > best.Value : quality.Value < best.Value)) best = quality;
        }
        return best;
    }

    /// <summary>
    /// Updates the early-stopping plateau counters after a committed batch, resetting them when the configured
    /// metric improved by at least the configured minimum and otherwise charging the batch's evaluations to patience.
    /// </summary>
    private void UpdateEarlyStopping(long committedEvaluations)
    {
        if (_options.EarlyStopping.PatienceEvaluations <= 0) return;
        double? metric = CurrentEarlyStoppingMetric();
        if (metric.HasValue &&
            (!_earlyStoppingBest.HasValue || metric.Value - _earlyStoppingBest.Value >= _options.EarlyStopping.MinimumImprovement))
        {
            _earlyStoppingBest = metric;
            _evaluationsSinceImprovement = 0;
            return;
        }
        _evaluationsSinceImprovement += committedEvaluations;
    }

    /// <summary>Returns whether the early-stopping patience has been exhausted.</summary>
    private bool IsEarlyStopped() => _options.EarlyStopping.PatienceEvaluations > 0 &&
        _evaluationsSinceImprovement >= _options.EarlyStopping.PatienceEvaluations;

    /// <summary>
    /// Computes the configured early-stopping metric across every island, normalized so that a larger value is always
    /// better, including under minimization.
    /// </summary>
    private double? CurrentEarlyStoppingMetric()
    {
        bool maximize = _islands[0].Direction == EvolutionOptimizationDirection.Maximize;

        // A named evaluator metric wins over the three built-in views of the search, because a run often plateaus
        // on something only the evaluator can see. Absent from every evaluation, it yields null and the run simply
        // never stops early, which is safer than treating "not reported" as "no progress".
        if (_options.EarlyStopping.MetricName is { } watched)
        {
            double best = 0;
            bool any = false;
            foreach (IEvolutionArchive<TGenome> archive in _islands)
            {
                foreach (EvolutionArchiveEntry<TGenome> entry in archive.Entries)
                {
                    if (!entry.Evaluation.Metrics.TryGetValue(watched, out double value)) continue;
                    if (!EvolutionDescriptorDefinition.IsFinite(value)) continue;

                    double normalized = maximize ? value : -value;
                    if (!any || normalized > best)
                    {
                        best = normalized;
                        any = true;
                    }
                }
            }

            return any ? best : (double?)null;
        }

        switch (_options.EarlyStopping.Metric)
        {
            case EvolutionEarlyStoppingMetric.Coverage:
            {
                long occupied = 0;
                long total = 0;
                foreach (IEvolutionArchive<TGenome> archive in _islands)
                {
                    occupied += archive.Count;
                    long cells = TotalGridCells(archive);
                    total = cells > long.MaxValue - total ? long.MaxValue : total + cells;
                }
                return total == 0 ? null : occupied / (double)total;
            }
            case EvolutionEarlyStoppingMetric.QdScore:
            {
                double score = 0;
                bool any = false;
                foreach (IEvolutionArchive<TGenome> archive in _islands)
                {
                    foreach (EvolutionArchiveEntry<TGenome> entry in archive.Entries)
                    {
                        if (!entry.Evaluation.Quality.HasValue) continue;
                        any = true;
                        double contribution = maximize ? entry.Evaluation.Quality.Value : -entry.Evaluation.Quality.Value;
                        if (contribution > double.MaxValue - score) return double.MaxValue;
                        score += contribution;
                    }
                }
                return any ? score : (double?)null;
            }
            default:
            {
                double? best = BestQualityAcrossIslands();
                return best.HasValue ? (maximize ? best.Value : -best.Value) : (double?)null;
            }
        }
    }

    /// <summary>Returns the deterministic pause applied before a retry round, or zero when backoff is disabled.</summary>
    private TimeSpan RetryDelayForAttempt(int attemptCount)
    {
        if (_options.RetryBaseDelay <= TimeSpan.Zero || attemptCount <= 1) return TimeSpan.Zero;
        double factor = Math.Pow(_options.RetryBackoffMultiplier, attemptCount - 2);
        double milliseconds = _options.RetryBaseDelay.TotalMilliseconds * factor;
        if (!EvolutionDescriptorDefinition.IsFinite(milliseconds) || milliseconds >= MaximumRetryDelay.TotalMilliseconds)
            return MaximumRetryDelay;
        return TimeSpan.FromMilliseconds(milliseconds);
    }
}
