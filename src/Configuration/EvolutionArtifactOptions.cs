using System.Globalization;
using AiDotNet.Evolution;
using AiDotNet.Validation;

namespace AiDotNet.Configuration;

/// <summary>Bounds how much untrusted evaluator text an evolution run retains, sanitizes, and replays.</summary>
/// <remarks>
/// <para>
/// Artifacts are off by default, so an engine stores zero artifact bytes until <see cref="Enabled"/> is set. Once
/// enabled, every artifact an <see cref="EvolutionTaskResult"/> carries is sanitized (unless
/// <see cref="SanitizeSecrets"/> is cleared), truncated to <see cref="MaxArtifactBytes"/>, and then admitted in order
/// until either <see cref="MaxArtifactsPerEvaluation"/> artifacts or <see cref="MaxBytesPerEvaluation"/> total bytes
/// have been kept. The retained set is attached to <see cref="EvolutionEvaluation.Artifacts"/>, travels into the
/// checkpoint with its archive entry, and is folded into the run state hash, so two runs with the same seed produce the
/// same artifacts.
/// </para>
/// <para>
/// When <see cref="DeliverToNextProposal"/> is set, the retained artifacts are additionally queued for the next
/// variation proposal of the same lineage and reach it through
/// <see cref="EvolutionVariationContext{TGenome}.ParentArtifacts"/>. A completed candidate's artifacts are queued under
/// its own identity, because it can itself become a parent; a candidate that failed, timed out, or was skipped queues
/// them under its parent instead, because it will never enter an archive and so would never be selected again - which
/// is exactly the case where the note matters most. Delivery is consume-once: the queue entry is removed when it is
/// handed over, so a failure note informs exactly one follow-up proposal instead of being replayed forever. The queue
/// is capped at <see cref="MaxPendingCandidates"/> entries and evicts the oldest, which bounds memory whatever the
/// evaluator does. OpenEvolve's equivalent is a per-program dictionary popped by the worker after evaluation
/// (evaluator.py:63,319-329) with no cap of its own, and its snapshot path ships only the oldest 100 programs'
/// artifacts to each worker (process_parallel.py:509-524).
/// </para>
/// <para><b>For Beginners:</b> Turning this on lets the engine keep the error output of a failed evaluation and show it
/// to whatever proposes the next candidate, which is how a search learns from a compiler error instead of repeating it.
/// The numbers here are all safety valves: how big one piece of output may be, how many pieces you keep per candidate,
/// how many bytes in total, and how many candidates may have output waiting. Leave <see cref="SanitizeSecrets"/> on
/// unless you completely control the evaluated code, because program output routinely contains environment variables
/// and connection strings that you do not want written into a checkpoint file.</para>
/// </remarks>
public sealed class EvolutionArtifactOptions
{
    /// <summary>Gets or sets whether evaluator artifacts are retained at all.</summary>
    public bool Enabled { get; set; }

    /// <summary>Gets or sets the maximum UTF-8 byte length kept for one artifact; longer text is truncated.</summary>
    public int MaxArtifactBytes { get; set; } = 32 * 1024;

    /// <summary>Gets or sets the maximum number of artifacts kept for one evaluation.</summary>
    public int MaxArtifactsPerEvaluation { get; set; } = 8;

    /// <summary>Gets or sets the maximum total UTF-8 byte length kept across one evaluation's artifacts.</summary>
    public int MaxBytesPerEvaluation { get; set; } = 128 * 1024;

    /// <summary>Gets or sets whether credential-shaped content is removed before an artifact is stored.</summary>
    public bool SanitizeSecrets { get; set; } = true;

    /// <summary>Gets or sets whether retained artifacts are queued for the next proposal that uses the candidate as a parent.</summary>
    public bool DeliverToNextProposal { get; set; } = true;

    /// <summary>Gets or sets the maximum number of candidates whose artifacts may await delivery.</summary>
    public int MaxPendingCandidates { get; set; } = 256;

    /// <summary>Validates every value and returns an independent copy.</summary>
    /// <returns>A defensive copy that later mutation of this instance cannot affect.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// A bound is not positive, or <see cref="MaxArtifactsPerEvaluation"/> exceeds
    /// <see cref="EvolutionTaskResult.MaximumArtifacts"/>.
    /// </exception>
    internal EvolutionArtifactOptions SnapshotAndValidate()
    {
        Guard.Positive(MaxArtifactBytes);
        Guard.Positive(MaxArtifactsPerEvaluation);
        Guard.Positive(MaxBytesPerEvaluation);
        Guard.Positive(MaxPendingCandidates);
        if (MaxArtifactsPerEvaluation > EvolutionTaskResult.MaximumArtifacts)
            throw new ArgumentOutOfRangeException(nameof(MaxArtifactsPerEvaluation),
                $"At most {EvolutionTaskResult.MaximumArtifacts} artifacts may be attached to one evaluation.");

        return new EvolutionArtifactOptions
        {
            Enabled = Enabled,
            MaxArtifactBytes = MaxArtifactBytes,
            MaxArtifactsPerEvaluation = MaxArtifactsPerEvaluation,
            MaxBytesPerEvaluation = MaxBytesPerEvaluation,
            SanitizeSecrets = SanitizeSecrets,
            DeliverToNextProposal = DeliverToNextProposal,
            MaxPendingCandidates = MaxPendingCandidates
        };
    }

    /// <summary>Returns a stable, culture-independent representation suitable for compatibility hashes.</summary>
    /// <returns>The canonical text form of every value that changes artifact behaviour.</returns>
    internal string ToCanonicalString() => string.Join("|", new[]
    {
        Enabled ? "artifacts" : "no-artifacts",
        MaxArtifactBytes.ToString(CultureInfo.InvariantCulture),
        MaxArtifactsPerEvaluation.ToString(CultureInfo.InvariantCulture),
        MaxBytesPerEvaluation.ToString(CultureInfo.InvariantCulture),
        SanitizeSecrets ? "sanitize" : "raw",
        DeliverToNextProposal ? "deliver" : "no-deliver",
        MaxPendingCandidates.ToString(CultureInfo.InvariantCulture)
    });
}
