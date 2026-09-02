namespace AiDotNet.Configuration;

/// <summary>Configures how much of each language-model request and answer is kept as provenance.</summary>
/// <remarks>
/// <para>
/// Provenance is only useful if it is affordable and safe. These options set both budgets. The text budgets bound
/// what one record can cost on disk, so a run that asks a model ten thousand times cannot fill a volume with
/// quoted program source; <see cref="FlushEveryRecords"/> and <see cref="MaxSegmentBytes"/> bound how much sits
/// in memory and how large one crash-safe file segment grows.
/// </para>
/// <para>
/// Redaction is not configurable and always runs. Prompt and answer text is model output and program output —
/// untrusted by definition — so it passes through the same credential-stripping and control-character-stripping
/// pass the prompt builder uses before any of it reaches a file. Turning the text off entirely with
/// <see cref="IncludePromptText"/> and <see cref="IncludeResponseText"/> still leaves the identifiers, hashes,
/// token counts, timings, and outcomes, which is enough to reconstruct a lineage and cost a run without storing a
/// single quoted character.
/// </para>
/// <para><b>For Beginners:</b> These settings decide how much detail the library keeps about each conversation it
/// has with the AI. The defaults keep a generous excerpt of both sides, trimmed to a few kilobytes each, and write
/// them to disk in small batches. If your prompts contain anything you would rather not store, switch
/// <see cref="IncludePromptText"/> off — you keep the who, when, and how much, and lose only the what.</para>
/// </remarks>
public sealed class ProposalProvenanceOptions
{
    /// <summary>Gets or sets whether provenance is recorded at all.</summary>
    /// <remarks>Set to <c>false</c> to keep a configured sink attached but stop feeding it, without rewiring code.</remarks>
    public bool Enabled { get; set; } = true;

    /// <summary>Gets or sets whether failed attempts are recorded as well as accepted ones.</summary>
    /// <remarks>
    /// On by default, and worth leaving on: how often the model answered unusably is the single most diagnostic
    /// number a run produces, and it is invisible in the archive.
    /// </remarks>
    public bool RecordFailedAttempts { get; set; } = true;

    /// <summary>Gets or sets whether the redacted prompt text is kept.</summary>
    public bool IncludePromptText { get; set; } = true;

    /// <summary>Gets or sets whether the redacted answer text is kept.</summary>
    public bool IncludeResponseText { get; set; } = true;

    /// <summary>Gets or sets the maximum UTF-8 bytes of prompt text kept per record.</summary>
    public int MaxPromptBytes { get; set; } = 8_192;

    /// <summary>Gets or sets the maximum UTF-8 bytes of answer text kept per record.</summary>
    public int MaxResponseBytes { get; set; } = 8_192;

    /// <summary>Gets or sets how many records are buffered before a segment is written to disk.</summary>
    public int FlushEveryRecords { get; set; } = 16;

    /// <summary>Gets or sets the byte size at which a buffered batch is written even if it is not yet full.</summary>
    public long MaxSegmentBytes { get; set; } = 4L * 1024L * 1024L;

    /// <summary>Creates an independent copy of these options.</summary>
    /// <returns>A copy that shares no mutable state with this instance.</returns>
    public ProposalProvenanceOptions Clone() => new()
    {
        Enabled = Enabled,
        RecordFailedAttempts = RecordFailedAttempts,
        IncludePromptText = IncludePromptText,
        IncludeResponseText = IncludeResponseText,
        MaxPromptBytes = MaxPromptBytes,
        MaxResponseBytes = MaxResponseBytes,
        FlushEveryRecords = FlushEveryRecords,
        MaxSegmentBytes = MaxSegmentBytes
    };

    /// <summary>Validates the option values.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A budget is negative, or a positive-only budget is not positive.</exception>
    public void Validate()
    {
        if (MaxPromptBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxPromptBytes), MaxPromptBytes, "Value cannot be negative.");
        }

        if (MaxResponseBytes < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxResponseBytes), MaxResponseBytes, "Value cannot be negative.");
        }

        if (FlushEveryRecords <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(FlushEveryRecords), FlushEveryRecords, "Value must be positive.");
        }

        if (MaxSegmentBytes <= 0L)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxSegmentBytes), MaxSegmentBytes, "Value must be positive.");
        }
    }
}
